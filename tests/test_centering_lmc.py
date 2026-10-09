"""
Tests of the nba centering methods on real simulation data: a 10% random
subsample (1.5M particles) of the LMC dark matter particles of GC21
MWLMC5_b0 at snapshot 150 (t = 2.93 Gyr), in simulation coordinates.
At this time the LMC core is intact but surrounded by stripped debris, so a
good centre and the plain mean position differ by tens of kpc.

Data files:
- MWLMC5_b0_snap150_lmc_subsample.npz (42 MB, not in the repository): pos
  [kpc], vel [km/s] (float32), pid, mass (one value, 1e10 Msun) and JSON
  metadata. Looked for in the folder given by the NBA_TEST_DATA environment
  variable, else in tests/data/. The tests are skipped without it.
- tests/data/MWLMC5_b0_snap150_lmc_centers.json: reference centres computed
  with nba (commit in meta.provenance, later changes in meta.updates) on the
  subsample and on all 15M LMC particles, and the full-data orbit around
  snapshot 150.
Both were made by make_centering_test_data.py in the nba_centering project
(see tests/data/README.md).

Two kinds of tests:
- Regression: the shrinking sphere on the subsample reproduces the stored
  centres, velocities and info, for each stored setting. A failure means
  that the result changed: check whether the change was intended.
- Physics: properties that hold whatever the implementation details. The
  centre found from 10% of the particles agrees with the full-data centre,
  the velocity agrees with the rate of change of the full-data orbit,
  tracking gives the same centre, and mean_pos is pulled away by the debris.

Run with: pytest tests/test_centering_lmc.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from nba.com import CenterHalo
from nba.com.com_methods import ssphere_numba

DATA = Path(__file__).parent / "data"
SUBSAMPLE = Path(os.environ.get("NBA_TEST_DATA", DATA)) / "MWLMC5_b0_snap150_lmc_subsample.npz"
CENTERS = DATA / "MWLMC5_b0_snap150_lmc_centers.json"

INT_INFO = ("npart", "nmin", "niter", "nvel")

# Regression tolerances: the results are deterministic, so these only allow
# for a different summation order (e.g. another numpy version)
ATOL_POS = 1e-6  # kpc
ATOL_VEL = 1e-5  # km/s

# Physical tolerances, about 2-4x the values observed with nba 8c1c3a1
TOL_FULL_POS = 0.3  # kpc, subsample vs full-data centre (observed <= 0.071)
TOL_FULL_VEL = 3.0  # km/s, subsample vs full-data velocity, rvel_factor/nvel (observed <= 1.6)
TOL_FD_VEL = 4.0  # km/s, velocity vs finite difference of the full-data orbit (observed <= 1.9)
TOL_TRACKED = 0.05  # kpc, tracked vs untracked: the final 1000-particle sphere (R ~ 0.7 kpc)
                    # depends slightly on the starting sphere (observed 0.004 kpc)
MIN_MEAN_POS_OFFSET = 30.0  # kpc, mean_pos vs centre: the debris pulls it away (observed 66)

pytestmark = pytest.mark.skipif(not (SUBSAMPLE.exists() and CENTERS.exists()),
                                reason="LMC test data files not found")


@pytest.fixture(scope="module")
def ref():
    with open(CENTERS) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def halo():
    with np.load(SUBSAMPLE) as d:
        pos, vel, mass = d["pos"], d["vel"], d["mass"]
        meta = json.loads(str(d["meta"]))
    return {"pos": pos, "vel": vel, "mass": np.full(len(pos), mass, dtype=np.float32)}, meta


def kwargs_of(ref, name):
    kwargs = dict(ref["meta"]["configs"][name])
    if "center0" in kwargs:
        kwargs["center0"] = np.array(kwargs["center0"])
    return kwargs


@pytest.fixture(scope="module")
def computed(halo, ref):
    """Run every stored setting once: {(config, method): (pos, vel, info)}."""
    data, _ = halo
    out = {}
    for name, method in CASES:
        out[name, method] = getattr(CenterHalo(data), method)(return_info=True,
                                                                  **kwargs_of(ref, name))
    return out


CONFIGS = ("default", "recommended", "nvel", "uncapped", "tracked")
# The NumPy version takes ~20 s per setting on 1.5M particles: run it for two
NUMPY_CONFIGS = ("default", "tracked")
CASES = [(c, "shrinking_sphere_numba") for c in CONFIGS] + [(c, "shrinking_sphere") for c in NUMPY_CONFIGS]


# --- the data files -----------------------------------------------------------

def test_subsample_file(halo, ref):
    data, meta = halo
    n = meta["n_subsample"]
    assert data["pos"].shape == data["vel"].shape == (n, 3)
    assert data["pos"].dtype == data["vel"].dtype == np.float32
    assert n == ref["meta"]["n_subsample"] == 1_500_000
    with np.load(SUBSAMPLE) as d:
        pid = d["pid"]
    assert len(np.unique(pid)) == n  # drawn without replacement
    assert pid.min() >= 107180001  # LMC IDs only
    # Not recentred: the LMC is ~85 kpc from the simulation origin at this time
    assert np.linalg.norm(ref["subsample"]["recommended/shrinking_sphere_numba"]["pos"]) > 50


# --- regression against the stored nba results --------------------------------

@pytest.mark.parametrize("name,method", CASES)
def test_reference_centres(computed, ref, name, method):
    pos, vel, info = computed[name, method]
    r = ref["subsample"][f"{name}/{method}"]
    np.testing.assert_allclose(pos, r["pos"], rtol=0, atol=ATOL_POS)
    np.testing.assert_allclose(vel, r["vel"], rtol=0, atol=ATOL_VEL)
    for key in INT_INFO:
        assert info[key] == r["info"][key], key
    assert info["stop"] == r["info"]["stop"]
    np.testing.assert_allclose(info["radius"], r["info"]["radius"], rtol=1e-9)
    np.testing.assert_allclose(info["density"], r["info"]["density"], rtol=1e-6)


@pytest.mark.parametrize("name", NUMPY_CONFIGS)
def test_numpy_and_numba_agree(computed, name):
    a, b = computed[name, "shrinking_sphere"], computed[name, "shrinking_sphere_numba"]
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_ssphere_numba_function(halo, computed, ref):
    data, _ = halo
    pos, vel, info = ssphere_numba(data["pos"], data["vel"], data["mass"], return_info=True,
                                   **kwargs_of(ref, "recommended"))
    np.testing.assert_array_equal(pos, computed["recommended", "shrinking_sphere_numba"][0])
    np.testing.assert_array_equal(vel, computed["recommended", "shrinking_sphere_numba"][1])
    assert info == computed["recommended", "shrinking_sphere_numba"][2]


# --- the info dict ------------------------------------------------------------

@pytest.mark.parametrize("name", ["default", "recommended", "nvel", "uncapped"])
def test_info_consistent_with_particles(halo, computed, ref, name):
    data, _ = halo
    pos, _, info = computed[name, "shrinking_sphere_numba"]
    kwargs = ref["meta"]["configs"][name]
    r = np.sqrt(np.sum((data["pos"].astype(np.float64) - pos)**2, axis=1))
    inside = r < info["radius"]
    assert info["npart"] >= info["nmin"]
    assert info["npart"] == np.count_nonzero(inside)
    volume = 4 / 3 * np.pi * info["radius"]**3
    np.testing.assert_allclose(info["density"], data["mass"][inside].sum(dtype=np.float64) / volume,
                               rtol=1e-6)
    if "rvel_factor" in kwargs:
        assert info["nvel"] == np.count_nonzero(r < kwargs["rvel_factor"] * info["radius"])
    elif "nvel" in kwargs:
        assert info["nvel"] == kwargs["nvel"]
    else:
        assert info["nvel"] == np.count_nonzero(r < 20.0)


@pytest.mark.parametrize("name", ["default", "uncapped", "tracked"])
def test_info_npart_and_density_describe_the_same_sphere(halo, computed, name):
    data, _ = halo
    pos, _, info = computed[name, "shrinking_sphere_numba"]
    volume = 4 / 3 * np.pi * info["radius"]**3
    assert round(info["density"] * volume / float(data["mass"][0])) == info["npart"]


def test_nmin(computed):
    n = 1_500_000
    assert computed["default", "shrinking_sphere_numba"][2]["nmin"] == min(1000, int(0.01 * n))
    assert computed["uncapped", "shrinking_sphere_numba"][2]["nmin"] == 10_000


def test_softening_floor(computed):
    """With softening the sphere never shrinks below 4 * softening."""
    for name in ("recommended", "nvel", "uncapped", "tracked"):
        info = computed[name, "shrinking_sphere_numba"][2]
        assert info["radius"] >= 4 * 0.08 or info["stop"] != "softening"
        assert info["radius"] * 0.975 >= 4 * 0.08 or info["stop"] == "softening"


# --- physics ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["default", "recommended", "nvel", "uncapped"])
def test_subsample_centre_matches_full_data(computed, ref, name):
    """10% of the particles give the centre found from all 15M particles."""
    pos = computed[name, "shrinking_sphere_numba"][0]
    full = ref["full"][f"{name}/shrinking_sphere_numba"]["pos"]
    assert np.linalg.norm(pos - full) < TOL_FULL_POS


@pytest.mark.parametrize("name", ["recommended", "nvel"])
def test_subsample_velocity_matches_full_data(computed, ref, name):
    vel = computed[name, "shrinking_sphere_numba"][1]
    full = ref["full"][f"{name}/shrinking_sphere_numba"]["vel"]
    assert np.linalg.norm(vel - full) < TOL_FULL_VEL


@pytest.mark.parametrize("name", ["recommended", "nvel"])
def test_velocity_matches_orbit(computed, ref, name):
    """The centre's velocity equals the rate of change of the (full-data) centre."""
    vel = computed[name, "shrinking_sphere_numba"][1]
    assert np.linalg.norm(vel - ref["orbit"]["vel_finite_difference"]) < TOL_FD_VEL


def test_tracking_gives_the_same_centre(computed):
    """Starting from the previous snapshot's centre with r0 = 15 kpc finds the same centre."""
    tracked = computed["tracked", "shrinking_sphere_numba"]
    untracked = computed["recommended", "shrinking_sphere_numba"]
    assert np.linalg.norm(tracked[0] - untracked[0]) < TOL_TRACKED
    assert tracked[2]["niter"] < untracked[2]["niter"]  # and starts much closer


def test_mean_pos_is_pulled_by_debris(halo, computed, ref):
    data, _ = halo
    pos, vel = CenterHalo(data).mean_pos()
    np.testing.assert_allclose(pos, ref["subsample"]["mean_pos"]["pos"], rtol=0, atol=ATOL_POS)
    np.testing.assert_allclose(vel, ref["subsample"]["mean_pos"]["vel"], rtol=0, atol=ATOL_VEL)
    centre = computed["recommended", "shrinking_sphere_numba"][0]
    assert np.linalg.norm(pos - centre) > MIN_MEAN_POS_OFFSET
