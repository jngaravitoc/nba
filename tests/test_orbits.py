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


def write_snap(path, center):
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(N, 3)) + center
    with h5py.File(path, "w") as f:
        f.create_group("Header").attrs["Time"] = float(center[0])
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
