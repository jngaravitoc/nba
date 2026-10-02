import numpy as np
import pytest

from nba.structure import Profiles


def uniform_sphere(n, radius, seed=0):
    rng = np.random.default_rng(seed)
    r = radius * rng.random(n) ** (1 / 3)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * r[:, None]


def test_density_uniform_sphere():
    n, radius = 200000, 10.0
    pos = uniform_sphere(n, radius)
    edges = np.linspace(1, 9, 9)
    _, rho = Profiles(pos, edges).density(mass=1.0)
    expected = n / (4 / 3 * np.pi * radius**3)
    np.testing.assert_allclose(rho, expected, rtol=0.05)


def test_enclosed_mass_total():
    pos = uniform_sphere(1000, 5.0)
    edges = np.linspace(0, 6, 13)
    _, menc = Profiles(pos, edges).enclosed_mass(np.ones(1000))
    assert menc[-1] == pytest.approx(1000)
    assert np.all(np.diff(menc) >= 0)


def test_invalid_input():
    with pytest.raises(ValueError):
        Profiles(np.zeros((10, 2)), [0, 1])
    with pytest.raises(ValueError):
        Profiles(np.zeros((10, 3)), [1])
    with pytest.raises(ValueError):
        Profiles(np.zeros((10, 3)), [0, 1]).enclosed_mass(np.ones(5))
