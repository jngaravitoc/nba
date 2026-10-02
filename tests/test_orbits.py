import h5py
import numpy as np
import pytest

from nba.orbits import orbit

N = 5000
CENTERS = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])]
VEL = np.array([5.0, 0.0, -5.0])


def write_snap(path, center):
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(N, 3)) + center
    with h5py.File(path, "w") as f:
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


@pytest.mark.parametrize("method", ["shrinking", "diskpot", "mean"])
def test_orbit_recovers_centers(snaps, method):
    pos, vel = orbit(snaps, "sim_{:03d}.hdf5", [0, 1], halo="MW", com_method=method)
    assert pos.shape == vel.shape == (2, 3)
    np.testing.assert_allclose(pos, CENTERS, atol=0.3)
    np.testing.assert_allclose(vel, [VEL, VEL], atol=0.3)


def test_orbit_invalid_method(snaps):
    with pytest.raises(ValueError):
        orbit(snaps, "sim_{:03d}.hdf5", [0], com_method="nope")


def test_diskpot_requires_mw(snaps):
    with pytest.raises(ValueError):
        orbit(snaps, "sim_{:03d}.hdf5", [0], halo="LMC", com_method="diskpot")
