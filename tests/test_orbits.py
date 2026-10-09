import warnings

import h5py
import numpy as np
import pytest

from nba.orbits import iter_orbit, orbit

N = 5000
CENTERS = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])]
VEL = np.array([5.0, 0.0, -5.0])
# The test halos are unit Gaussians, i.e. cored: stop the shrinking sphere at r = 4 * SOFTENING = 1
SOFTENING = 0.25


def write_snap(path, center, time=None, scale=1.0, units=False):
    """Unit Gaussian halo at `center`, with header time center[0] unless given."""
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(N, 3)) * scale + center
    with h5py.File(path, "w") as f:
        f.create_group("Header").attrs["Time"] = float(center[0] if time is None else time)
        if units:  # Gadget's kpc, km/s and 1e10 Msun
            par = f.create_group("Parameters").attrs
            par["UnitLength_in_cm"] = 3.085678e21
            par["UnitVelocity_in_cm_per_s"] = 1e5
            par["UnitMass_in_g"] = 1.989e43
        for group, mass in (("PartType1", 1.0), ("PartType2", 0.1)):
            g = f.create_group(group)
            g["Coordinates"] = pos
            g["Velocities"] = rng.normal(size=(N, 3)) * 0.1 + VEL
            g["Masses"] = np.full(N, mass)
            g["ParticleIDs"] = np.arange(N)
            # potential minimum at the center of the distribution
            g["Potential"] = np.linalg.norm(pos - center, axis=1)


@pytest.fixture
def snaps(tmp_path):
    for k, c in enumerate(CENTERS):
        write_snap(tmp_path / f"sim_{k:03d}.hdf5", c)
    return str(tmp_path)


@pytest.mark.parametrize("method", ["shrinking", "diskpot", "mean", "mean_pos", "shrinking_sphere",
                                    "shrinking_sphere_numba", "min_potential"])
def test_orbit_recovers_centers(snaps, method):
    pos, vel = orbit(snaps, "sim_{:03d}.hdf5", [0, 1], halo="MW", com_method=method,
                     softening=SOFTENING)
    assert pos.shape == vel.shape == (2, 3)
    np.testing.assert_allclose(pos, CENTERS, atol=0.3)
    np.testing.assert_allclose(vel, [VEL, VEL], atol=0.3)


def test_orbit_several_methods(snaps):
    methods = ["mean_pos", "shrinking_sphere", "shrinking_sphere_numba", "min_potential"]
    result = orbit(snaps, "sim_{:03d}.hdf5", [0, 1], halo="MW", com_method=methods,
                    softening=SOFTENING)
    assert list(result) == methods
    for pos, vel in result.values():
        np.testing.assert_allclose(pos, CENTERS, atol=0.3)
        np.testing.assert_allclose(vel, [VEL, VEL], atol=0.3)


def test_iter_orbit_reads_each_snapshot_once(snaps, monkeypatch):
    from nba.ios import snap_reader

    calls = []
    original = snap_reader.ReadGadgetSim.read_snapshot

    def counting(self, quantity, ptype, snapformat=3):
        calls.append((self.snapname, ptype))
        return original(self, quantity, ptype, snapformat)

    monkeypatch.setattr(snap_reader.ReadGadgetSim, "read_snapshot", counting)
    steps = list(iter_orbit(snaps, "sim_{:03d}.hdf5", [0, 1], halo="MW",
                            com_method=["mean_pos", "shrinking_sphere", "min_potential"],
                            softening=SOFTENING))

    assert [s[0] for s in steps] == [0, 1]
    assert [s[1] for s in steps] == [CENTERS[0][0], CENTERS[1][0]]  # time from the header
    assert calls == [("sim_000.hdf5", "dm"), ("sim_001.hdf5", "dm")]


def test_orbit_invalid_method(snaps):
    with pytest.raises(ValueError):
        orbit(snaps, "sim_{:03d}.hdf5", [0], com_method="nope")


def test_diskpot_requires_mw(snaps):
    with pytest.raises(ValueError):
        orbit(snaps, "sim_{:03d}.hdf5", [0], halo="LMC", com_method="diskpot")


def test_potential_methods_require_mw(snaps):
    with pytest.raises(ValueError, match="shrinking sphere"):
        orbit(snaps, "sim_{:03d}.hdf5", [0], halo="LMC", com_method="min_potential")


def test_orbit_warns_without_softening(snaps):
    with pytest.warns(UserWarning, match="softening"):
        orbit(snaps, "sim_{:03d}.hdf5", [0], halo="MW", com_method="shrinking")


def test_iter_orbit_warns_on_jump(snaps):
    # the centers move by sqrt(3) in dt = 1 with |v| ~ 7
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        list(iter_orbit(snaps, "sim_{:03d}.hdf5", [0, 1], com_method="mean_pos"))
    with pytest.warns(UserWarning, match="moved"):
        list(iter_orbit(snaps, "sim_{:03d}.hdf5", [0, 1], com_method="mean_pos", jump_factor=0.1))


def test_iter_orbit_time_offset(snaps):
    with pytest.warns(UserWarning, match="time decreases"):
        steps = list(iter_orbit(snaps, "sim_{:03d}.hdf5", [1, 0], com_method="mean_pos"))
    assert [s[1] for s in steps] == [2.0, 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        steps = list(iter_orbit(snaps, "sim_{:03d}.hdf5", [1, 0], com_method="mean_pos",
                                time_offset=lambda k: 10.0 if k == 0 else 0.0, jump_factor=None))
    assert [s[1] for s in steps] == [2.0, 11.0]


def test_iter_orbit_return_info(snaps):
    steps = list(iter_orbit(snaps, "sim_{:03d}.hdf5", [0], com_method=["shrinking", "min_potential"],
                            softening=SOFTENING, rvel_factor=2.0, return_info=True))
    _, _, centers = steps[0]
    _, _, info = centers["shrinking"]
    assert info["stop"] == "softening" and info["nvel"] > 0
    _, _, info = centers["min_potential"]
    assert info["npart"] > 0


def test_read_halo_split_and_sample(tmp_path, monkeypatch):
    from nba.ios import ReadGC21

    rng = np.random.default_rng(2)
    ids = rng.permutation(N) + 1000  # MW: the 3000 lowest IDs
    with h5py.File(tmp_path / "split.hdf5", "w") as f:
        g = f.create_group("PartType1")
        g["Coordinates"] = np.zeros((N, 3))
        g["ParticleIDs"] = ids
    monkeypatch.setattr(ReadGC21, "npart_mw", 3000)
    reader = ReadGC21(str(tmp_path), "split.hdf5")
    mw = reader.read_halo("pos", halo="MW", ptype="dm")
    lmc = reader.read_halo("pos", halo="LMC", ptype="dm")
    np.testing.assert_array_equal(mw["pid"], ids[ids < 4000])
    np.testing.assert_array_equal(lmc["pid"], ids[ids >= 4000])
    sample = reader.read_halo("pos", halo="LMC", ptype="dm", randomsample=1500, seed=0)
    assert len(np.unique(sample["pid"])) == 1500  # exactly n distinct particles
    assert np.all(sample["pid"] >= 4000)


@pytest.mark.parametrize("method", ["shrinking_sphere", "shrinking_sphere_numba"])
def test_orbit_tracks_previous_center(snaps, method, monkeypatch):
    from nba.com import CenterHalo

    starts = []
    original = getattr(CenterHalo, method)

    def recording(self, *args, center0=None, r0=None, **kwargs):
        starts.append((None if center0 is None else center0.copy(), r0))
        return original(self, *args, center0=center0, r0=r0, **kwargs)

    monkeypatch.setattr(CenterHalo, method, recording)
    pos, _ = orbit(snaps, "sim_{:03d}.hdf5", [0, 1], halo="MW", com_method=method, r0=5.0,
                   softening=SOFTENING)
    np.testing.assert_allclose(pos, CENTERS, atol=0.3)
    assert starts[0] == (None, None)  # first snapshot: all particles
    np.testing.assert_allclose(starts[1][0], pos[0])  # then the previous center
    assert starts[1][1] == 5.0


def moving_snaps(tmp_path, centers, scales=None):
    """Snapshots at t = 0, 1, 2... with the halo at `centers` (spread `scales`)."""
    scales = scales or [1.0] * len(centers)
    for k, (c, scale) in enumerate(zip(centers, scales)):
        write_snap(tmp_path / f"mov_{k:03d}.hdf5", np.asarray(c, dtype=float), time=float(k),
                   scale=scale, units=True)
    return str(tmp_path), "mov_{:03d}.hdf5", list(range(len(centers)))


def test_velocity_check(tmp_path):
    # Centers that move with the particles' velocity VEL pass the check
    path, name, snaps = moving_snaps(tmp_path, [VEL * k for k in range(5)])
    log = []
    steps = list(iter_orbit(path, name, snaps, com_method="shrinking", softening=SOFTENING,
                            rvel_factor=2.0, return_info=True, warning_log=log))
    assert log == []
    errors = [s[2]["shrinking"][2]["velocity_error"] for s in steps]
    assert np.isnan(errors[0]) and max(errors[1:]) < 0.1

    # Centers that wander against their velocity are flagged after velocity_window steps
    wander = [[0, 0, 0], [0, 6, 0], [0, 0, 6], [0, -6, 0], [0, 0, -6], [0, 6, 0]]
    (tmp_path / "w").mkdir()
    path, name, snaps = moving_snaps(tmp_path / "w", wander)
    log = []
    with pytest.warns(UserWarning, match="inconsistently with its velocity"):
        list(iter_orbit(path, name, snaps, com_method="shrinking", softening=SOFTENING,
                        rvel_factor=2.0, jump_factor=None, warning_log=log))
    assert [(k, m) for k, m, msg in log if "inconsistently" in msg] == [(3, "shrinking")]

    # Not applied when the velocity comes from the fixed rcut_vel sphere
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        list(iter_orbit(path, name, snaps, com_method="shrinking", softening=SOFTENING,
                        jump_factor=None))


def test_min_density_ratio(tmp_path):
    # The halo spreads out: its central density falls by 8 and then 27
    path, name, snaps = moving_snaps(tmp_path, [[0, 0, 0]] * 3, scales=[1.0, 2.0, 3.0])
    log = []
    with pytest.warns(UserWarning, match="density in the final sphere"):
        steps = list(iter_orbit(path, name, snaps, com_method="shrinking", softening=SOFTENING,
                                min_density_ratio=0.5, return_info=True, warning_log=log))
    ratios = [s[2]["shrinking"][2]["density_ratio"] for s in steps]
    assert ratios[0] == 1 and ratios[1] < 0.5 and ratios[2] < ratios[1]
    assert [(k, m) for k, m, msg in log if "density" in msg] == [(1, "shrinking")]  # warned once


def test_orbit_writes_ecsv(tmp_path):
    import astropy.units as u
    from nba.orbits import read_orbit

    path, name, snaps = moving_snaps(tmp_path, [VEL * k for k in range(3)])
    out = str(tmp_path / "orbit_{method}.ecsv")
    pos, vel = orbit(path, name, snaps, com_method="shrinking", softening=SOFTENING, r0=5.0,
                     rvel_factor=2.0, time_offset=lambda k: 10.0 if k == 2 else 0.0,
                     outfile=out, simulation={"name": "test"}, notes="a test")
    table = read_orbit(out.format(method="shrinking"))
    np.testing.assert_array_equal(table["snap"], snaps)
    np.testing.assert_array_equal(np.c_[table["x"], table["y"], table["z"]], pos)
    np.testing.assert_array_equal(np.c_[table["vx"], table["vy"], table["vz"]], vel)
    assert table["x"].unit == u.kpc and table["vx"].unit == u.km / u.s
    np.testing.assert_array_equal(table["t_code"], [0.0, 1.0, 12.0])
    np.testing.assert_array_equal(table["time_offset"], [0.0, 0.0, 10.0])
    np.testing.assert_allclose(table["t"].quantity.to_value(u.Gyr), np.array([0.0, 1.0, 12.0]) * 0.9778, rtol=1e-4)
    for col in ("radius", "npart", "nmin", "niter", "stop", "density", "density_ratio", "nvel",
                "velocity_error"):
        assert col in table.colnames, col
    meta = table.meta
    assert meta["method"] == "shrinking_sphere_numba"
    assert meta["parameters"]["r0"] == 5.0 and meta["parameters"]["min_npart"] == 1000
    assert "2 times the final radius" in meta["parameters"]["velocity_region"]
    assert meta["simulation"]["name"] == "test" and meta["notes"] == "a test"
    assert meta["simulation"]["units"]["time_code_to_Gyr"] == pytest.approx(0.9778, rel=1e-4)
    assert meta["provenance"]["nba_version"]
    assert meta["warnings"] == []
    # numpy structured array without units
    assert read_orbit(out.format(method="shrinking"), as_array=True)["x"].shape == (3,)


def test_orbit_records_warnings(snaps, tmp_path):
    out = str(tmp_path / "o_{method}.ecsv")
    with pytest.warns(UserWarning):
        orbit(snaps, "sim_{:03d}.hdf5", [1, 0], com_method=["mean_pos", "shrinking"], outfile=out)
    from nba.orbits import read_orbit
    messages = [w["message"] for w in read_orbit(out.format(method="shrinking")).meta["warnings"]]
    assert any("softening" in m for m in messages) and any("time decreases" in m for m in messages)
    # mean_pos gets the warnings that concern every method, not the shrinking sphere's
    messages = [w["message"] for w in read_orbit(out.format(method="mean_pos")).meta["warnings"]]
    assert any("time decreases" in m for m in messages)
    # no Parameters group: times in code units
    assert read_orbit(out.format(method="mean_pos"))["t"].unit is None


def test_orbit_outfile_needs_method_field(snaps, tmp_path):
    with pytest.raises(ValueError, match="method"):
        orbit(snaps, "sim_{:03d}.hdf5", [0], com_method=["mean_pos", "shrinking"],
              outfile=str(tmp_path / "o.ecsv"))
