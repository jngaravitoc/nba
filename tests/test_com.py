import numpy as np
import pytest

from nba.com import CenterHalo
from nba.com.com_methods import _initial_sphere, _ssphere_kernel, _ssphere_power


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
    # In float32, 1e8 + 1 rounds back to 1e8, so a float32 sum gives 0 instead of 1/3
    values = np.array([[1e8, 1e8, 1e8],
                       [1.0, 1.0, 1.0],
                       [-1e8, -1e8, -1e8]], dtype=np.float32)
    mass = np.ones(3, dtype=np.float32)
    com, _ = CenterHalo({"pos": values, "vel": values, "mass": mass}).mean_pos()
    assert com.dtype == np.float64
    np.testing.assert_allclose(com, np.full(3, 1.0 / 3.0), rtol=1e-12)


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


def _power_brute_force(pos, mass, center, radius, nmin):
    """Power et al. (2003) shrinking sphere over all particles at every step."""
    while True:
        new_radius = 0.975 * radius
        inside = np.sum((pos - center)**2, axis=1) < new_radius**2
        if np.count_nonzero(inside) < nmin:
            return center, radius
        center = np.average(pos[inside], axis=0, weights=mass[inside])
        radius = new_radius


@pytest.mark.parametrize("method", ["shrinking_sphere", "shrinking_sphere_numba"])
def test_shrinking_sphere_matches_power_brute_force(method):
    data, _ = cuspy_halo_with_satellite(n=20000)
    pos, mass = data["pos"], data["mass"]
    start = np.average(pos, axis=0, weights=mass)
    expected, radius = _power_brute_force(pos, mass, start,
                                          np.max(np.linalg.norm(pos - start, axis=1)), 200)
    com, _, info = getattr(CenterHalo(data), method)(return_info=True)
    # Rounding (here already in the start center) can move a particle across
    # the sphere edge and change the path slightly, so the stop can come a few
    # steps earlier or later: ask for agreement well within the final sphere
    assert info["radius"] == pytest.approx(radius, rel=0.2)
    np.testing.assert_allclose(com, expected, rtol=0, atol=0.1 * radius)


@pytest.mark.parametrize("loop", [_ssphere_power, _ssphere_kernel])
def test_shrinking_sphere_pruning_is_exact(loop):
    # The satellite pulls the start center away, so the center drifts and the
    # sorted reference is reset during the shrinking
    data, _ = cuspy_halo_with_satellite(n=20000)
    pos, mass = data["pos"], data["mass"]
    center, radius = _initial_sphere(pos, mass)
    pruned = loop(pos, mass, center, radius, -1.0, 200, 0.0, True)
    full = loop(pos, mass, center, radius, -1.0, 200, 0.0, False)
    np.testing.assert_array_equal(pruned[0], full[0])
    assert pruned[1:] == full[1:]


@pytest.mark.parametrize("method", ["shrinking_sphere", "shrinking_sphere_numba"])
def test_shrinking_sphere_radius_sequence(halo, method):
    data, offset, _ = halo
    _, _, info = getattr(CenterHalo(data), method)(center0=offset, r0=3.0, return_info=True)
    assert info["niter"] > 0
    assert info["radius"] == pytest.approx(3.0 * 0.975**info["niter"])
