import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
hp = pytest.importorskip("healpy")

from nba.visuals.mollweide import healpix_density_map, plot_mollweide_galactic


def test_density_map_total_counts():
    rng = np.random.default_rng(0)
    l = rng.uniform(0, 360, 5000)
    b = np.degrees(np.arcsin(rng.uniform(-1, 1, 5000)))
    nside = 16
    m = healpix_density_map(l, b, nside, smooth=0)
    assert m.shape == (hp.nside2npix(nside),)
    assert m.sum() * hp.nside2pixarea(nside, degrees=True) == pytest.approx(5000)


def test_density_map_peaks_at_pole():
    l = np.zeros(1000)
    b = np.full(1000, 89.0)
    m = healpix_density_map(l, b, 16, smooth=5)
    assert hp.pix2ang(16, np.argmax(m))[0] < np.radians(10)


def test_plot_saves_figure(tmp_path):
    m = healpix_density_map(np.array([10.0, 20.0]), np.array([0.0, 30.0]), 8, smooth=10)
    out = tmp_path / "map.png"
    plot_mollweide_galactic(m, figname=str(out))
    assert out.exists()
