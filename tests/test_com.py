import numpy as np
import pytest

from nba.com import CenterHalo


def cuspy_halo_with_satellite(n=200000, seed=1):
    """Cuspy halo centered at (3, -2, 1) plus a satellite with 10% of its particles at 80 kpc."""
    rng = np.random.default_rng(seed)
    r = rng.pareto(1.5, n) * 5
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    center = np.array([3.0, -2.0, 1.0])
    pos = np.vstack([d * r[:, None] + center, rng.normal(size=(n // 10, 3)) * 3 + [80, 0, 0]])
    return {"pos": pos, "vel": np.zeros_like(pos), "mass": np.ones(len(pos))}, center


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


def test_shrinking_sphere_versions_agree(halo):
    data, _, _ = halo
    com, vcom = CenterHalo(data).shrinking_sphere(min_npart=500)
    com_nb, vcom_nb = CenterHalo(data).shrinking_sphere_numba(min_npart=500)
    np.testing.assert_allclose(com, com_nb)
    np.testing.assert_allclose(vcom, vcom_nb)


def test_mean_pos_center(halo):
    data, offset, voffset = halo
    # The shell around the halo center holds particles; around the origin it does not
    com, vcom = CenterHalo(data).mean_pos(rmin=0, rmax=3, center=offset)
    np.testing.assert_allclose(com, offset, atol=0.05)
    np.testing.assert_allclose(vcom, voffset, atol=0.05)
    with pytest.raises(ValueError):
        CenterHalo(data).mean_pos(rmin=0, rmax=1)


def test_min_potential_disk_pot_override(halo):
    data, offset, _ = halo
    data = dict(data, pot=np.linalg.norm(data["pos"], axis=1))  # minimum near the origin
    disk_pot = np.linalg.norm(data["pos"] - offset, axis=1)  # minimum at the halo center
    com, _ = CenterHalo(data).min_potential(disk_pot=disk_pot, rcut=1.0)
    np.testing.assert_allclose(com, offset, atol=0.1)


def test_min_potential_without_pot(halo):
    data, _, _ = halo
    with pytest.raises(ValueError):
        CenterHalo(data).min_potential()


def test_recenter(halo):
    data, offset, voffset = halo
    pos0 = data["pos"].copy()
    CenterHalo({"pos": data["pos"]}).recenter(offset)  # no velocities
    np.testing.assert_allclose(data["pos"], pos0 - offset)
    vel0 = data["vel"].copy()
    CenterHalo(data).recenter(np.zeros(3))  # no vcom: velocities untouched
    np.testing.assert_allclose(data["vel"], vel0)


@pytest.mark.parametrize("method", ["shrinking_sphere", "shrinking_sphere_numba"])
def test_shrinking_sphere_ignores_satellite(method):
    data, center = cuspy_halo_with_satellite()
    com, _ = getattr(CenterHalo(data), method)()
    # the global mean is ~7 kpc away, pulled by the satellite
    np.testing.assert_allclose(com, center, atol=0.1)


@pytest.mark.parametrize("method", ["shrinking_sphere", "shrinking_sphere_numba"])
def test_shrinking_sphere_initial_sphere(method):
    data, center = cuspy_halo_with_satellite()
    # starting on the satellite finds the satellite, starting near the halo finds the halo
    com, _ = getattr(CenterHalo(data), method)(center0=[80, 0, 0], r0=20, softening=0.75)  # cored satellite, sigma=3
    np.testing.assert_allclose(com, [80, 0, 0], atol=0.3)
    com, _ = getattr(CenterHalo(data), method)(center0=center + 5, r0=50)
    np.testing.assert_allclose(com, center, atol=0.1)


def test_shrinking_sphere_empty_initial_sphere(halo):
    data, _, _ = halo
    with pytest.raises(ValueError):
        CenterHalo(data).shrinking_sphere(center0=[1e3, 0, 0], r0=1)
    with pytest.raises(ValueError):
        CenterHalo(data).shrinking_sphere_numba(r0=-1)


@pytest.mark.parametrize("method", ["shrinking_sphere", "shrinking_sphere_numba"])
def test_shrinking_sphere_info(halo, method):
    data, _, _ = halo
    com, _, info = getattr(CenterHalo(data), method)(min_npart=150, return_info=True)
    assert info["stop"] == "min_npart"
    assert 150 <= info["npart"] < 150 / 0.975**3  # one more step would go below 150
    r = np.linalg.norm(data["pos"] - com, axis=1)
    assert np.count_nonzero(r <= info["radius"]) >= info["npart"]
    assert info["niter"] > 0
    assert info["nvel"] == np.count_nonzero(r < 20)
    _, _, info = getattr(CenterHalo(data), method)(delta=10.0, return_info=True)
    assert info["stop"] == "delta" and info["niter"] == 1


@pytest.mark.parametrize("method", ["shrinking_sphere", "shrinking_sphere_numba"])
def test_shrinking_sphere_nvel(halo, method):
    data, offset, voffset = halo
    com, vcom, info = getattr(CenterHalo(data), method)(nvel=2000, return_info=True)
    assert info["nvel"] == 2000
    np.testing.assert_allclose(vcom, voffset, atol=0.1)
    nearest = np.argsort(np.linalg.norm(data["pos"] - com, axis=1))[:2000]
    np.testing.assert_allclose(vcom, data["vel"][nearest].mean(axis=0))


def test_shrinking_sphere_deprecated_names(halo):
    data, _, _ = halo
    with pytest.warns(DeprecationWarning):
        old = CenterHalo(data).shrinking_sphere_numba(minNpart=500, rcut=5.0)
    new = CenterHalo(data).shrinking_sphere_numba(min_npart=500, rcut_vel=5.0)
    np.testing.assert_allclose(old, new)
    with pytest.warns(DeprecationWarning):
        CenterHalo(data).shrinking_sphere(minNpart=500)
    with pytest.raises(TypeError):
        CenterHalo(data).shrinking_sphere(rcut=5.0)


def test_min_potential_npart(halo):
    data, offset, voffset = halo
    pot = np.linalg.norm(data["pos"] - offset, axis=1)
    com, vcom = CenterHalo(dict(data, pot=pot)).min_potential(npart=1000)
    lowest = np.argsort(pot)[:1000]
    np.testing.assert_allclose(com, data["pos"][lowest].mean(axis=0))
    np.testing.assert_allclose(vcom, data["vel"][lowest].mean(axis=0))
    with pytest.raises(ValueError):
        CenterHalo(dict(data, pot=pot)).min_potential(npart=0)


def test_recenter_copy(halo):
    data, offset, voffset = halo
    pos0, vel0 = data["pos"].copy(), data["vel"].copy()
    center = CenterHalo(data)
    center.recenter(offset, voffset, copy=True)
    np.testing.assert_array_equal(data["pos"], pos0)  # caller's arrays untouched
    np.testing.assert_array_equal(data["vel"], vel0)
    np.testing.assert_allclose(center.pos, pos0 - offset)
    np.testing.assert_allclose(center.vel, vel0 - voffset)


def test_float32_accumulates_in_float64():
    # 2**24 + 1 is not representable in float32: a float32 running sum stalls
    n = 2**24 + 1000
    pos = np.full((n, 3), 1.0, dtype=np.float32)
    pos[0] = 1001.0
    com, _ = CenterHalo({"pos": pos, "vel": pos, "mass": np.ones(n, dtype=np.float32)}).mean_pos()
    assert com.dtype == np.float64
    np.testing.assert_allclose(com, 1.0 + 1000.0 / n, rtol=1e-12)


@pytest.mark.parametrize("method", ["shrinking_sphere", "shrinking_sphere_numba"])
def test_shrinking_sphere_power_npart(halo, method):
    # Power et al. (2003): min_npart or 1% of the halo particles, whichever is smaller
    data, _, _ = halo  # 20000 particles: 1% = 200
    _, _, info = getattr(CenterHalo(data), method)(return_info=True)
    assert 200 <= info["npart"] < 200 / 0.975**3
    _, _, info = getattr(CenterHalo(data), method)(min_npart=100, return_info=True)
    assert 100 <= info["npart"] < 100 / 0.975**3


@pytest.mark.parametrize("method", ["shrinking_sphere", "shrinking_sphere_numba"])
def test_shrinking_sphere_softening(halo, method):
    data, offset, _ = halo
    com, _, info = getattr(CenterHalo(data), method)(softening=0.25, return_info=True)
    assert info["stop"] == "softening"
    assert 1.0 <= info["radius"] < 1.0 / 0.975  # stops before going below 4 * softening
    np.testing.assert_allclose(com, offset, atol=0.1)
    with pytest.raises(ValueError):
        getattr(CenterHalo(data), method)(softening=-1)
