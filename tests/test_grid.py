import numpy as np
import pytest

from nba.utils import Grid3D


def test_cartesian_shape():
    g = Grid3D("cartesian", [(0, 1), (0, 1), (0, 1)], num_points=[2, 3, 4])
    assert g.get().shape == (24, 3)


def test_spherical_to_cartesian_radius():
    g = Grid3D("spherical", [(1.0, 2.0), None, None], num_points=[4, 6, 5])
    cart = g.to("cartesian")
    r = np.linalg.norm(cart, axis=1)
    np.testing.assert_allclose(r, g.get()[:, 0])


def test_cylindrical_roundtrip():
    g = Grid3D("cylindrical", [(1.0, 2.0), None, (-1.0, 1.0)], num_points=[3, 6, 3])
    cart = g.to("cartesian")
    back = Grid3D._from_cartesian(g, cart, "cylindrical")
    np.testing.assert_allclose(back[:, 0], g.get()[:, 0])
    np.testing.assert_allclose(back[:, 2], g.get()[:, 2])


def test_invalid_system():
    with pytest.raises(ValueError):
        Grid3D("polar", [(0, 1)] * 3)


def test_invalid_num_points():
    with pytest.raises(ValueError):
        Grid3D("cartesian", [(0, 1)] * 3, num_points=[1, 2])
