import numpy as np
import pytest

from nba.com import CenterHalo


@pytest.fixture
def halo():
    rng = np.random.default_rng(0)
    n = 20000
    offset = np.array([5.0, -3.0, 2.0])
    voffset = np.array([10.0, 20.0, -5.0])
    return {
        "pos": rng.normal(size=(n, 3)) + offset,
        "vel": rng.normal(size=(n, 3)) + voffset,
        "mass": np.ones(n),
    }, offset, voffset


def test_mean_pos(halo):
    data, offset, voffset = halo
    com, vcom = CenterHalo(data).mean_pos()
    np.testing.assert_allclose(com, offset, atol=0.05)
    np.testing.assert_allclose(vcom, voffset, atol=0.05)


def test_mean_pos_bad_radii(halo):
    data, _, _ = halo
    with pytest.raises(ValueError):
        CenterHalo(data).mean_pos(rmin=-1, rmax=1)
    with pytest.raises(ValueError):
        CenterHalo(data).mean_pos(rmin=5, rmax=1)


def test_shrinking_sphere(halo):
    data, offset, voffset = halo
    com, vcom = CenterHalo(data).shrinking_sphere()
    np.testing.assert_allclose(com, offset, atol=0.3)
    np.testing.assert_allclose(vcom, voffset, atol=0.3)


def test_shrinking_sphere_numba(halo):
    data, offset, _ = halo
    com, _ = CenterHalo(data).shrinking_sphere_numba()
    np.testing.assert_allclose(com, offset, atol=0.3)
