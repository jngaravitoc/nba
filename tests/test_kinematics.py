import numpy as np
import pytest

from nba.kinematics import Kinematics
from nba.structure import Profiles


def sphere(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    r = 100.0 * rng.random(n) ** (1 / 3)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u * r[:, None], u


def test_beta_profile_grid_matches_profiles():
    pos, u = sphere()
    vel = np.random.default_rng(1).normal(size=pos.shape)
    kin = Kinematics(pos, vel)
    nbins, rmin, rmax = 11, 20.0, 90.0
    beta = kin.profiles(nbins=nbins, quantity="beta", rmin=rmin, rmax=rmax)

    r_centers, _ = Profiles(pos, np.linspace(rmin, rmax, nbins)).density()
    assert beta.shape == r_centers.shape
    np.testing.assert_allclose(kin.dr, r_centers)


def test_beta_isotropic_is_zero():
    pos, _ = sphere()
    vel = np.random.default_rng(1).normal(size=pos.shape)
    beta = Kinematics(pos, vel).profiles(nbins=6, quantity="beta", rmin=10, rmax=90)
    np.testing.assert_allclose(beta, 0, atol=0.1)


def test_beta_radial_orbits_is_one():
    pos, u = sphere()
    speed = np.random.default_rng(2).normal(size=len(pos))
    beta = Kinematics(pos, u * speed[:, None]).profiles(nbins=6, quantity="beta", rmin=10, rmax=90)
    np.testing.assert_allclose(beta, 1, atol=1e-6)


def test_profiles_does_not_modify_instance():
    pos, _ = sphere(5000)
    vel = np.random.default_rng(1).normal(size=pos.shape)
    kin = Kinematics(pos, vel)
    kin.profiles(nbins=5, quantity="beta", rmin=10, rmax=90)
    assert kin.pos is pos and kin.vel is vel
