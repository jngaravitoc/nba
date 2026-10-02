"""
Mollweide sky maps of directions (e.g. orbital poles) using HEALPix.

Requires healpy (``pip install astro-nba[extra]``).
"""

import numpy as np
import matplotlib.pyplot as plt


def healpix_density_map(l, b, nside, smooth=5):
    """
    Number density of directions per square degree on a HEALPix map.

    Parameters
    ----------
    l, b : array_like
        Galactic longitude and latitude in degrees.
    nside : int
        HEALPix resolution parameter.
    smooth : float
        FWHM of the Gaussian smoothing kernel in degrees (0 for no smoothing).

    Returns
    -------
    hpx_map : ndarray
        Density map (counts per square degree) of shape ``(12*nside**2,)``,
        smoothed if ``smooth > 0``.
    """
    import healpy as hp

    pix = hp.ang2pix(nside, np.radians(90.0 - np.asarray(b)), np.radians(l))
    counts = np.bincount(pix, minlength=hp.nside2npix(nside))
    hpx_map = counts / hp.nside2pixarea(nside, degrees=True)
    if smooth > 0:
        hpx_map = hp.smoothing(hpx_map, fwhm=np.radians(smooth))
    return hpx_map


def plot_mollweide_galactic(hpx_map, figname=None, rotation=(180, 0, 0), vmin=None, vmax=None,
                            title="", unit=""):
    """
    Plot a HEALPix map in a Mollweide projection in Galactic coordinates.

    Parameters
    ----------
    hpx_map : ndarray
        HEALPix map, e.g. from :func:`healpix_density_map`.
    figname : str, optional
        If given, the figure is saved here and closed.
    rotation : tuple
        Rotation (lon, lat, psi) of the map center in degrees.
    vmin, vmax : float, optional
        Color scale limits.
    """
    import healpy as hp

    hp.projview(hpx_map, coord=["G"], graticule=True, graticule_labels=True,
                rot=rotation, unit=unit, xlabel="Galactic Longitude (l)",
                ylabel="Galactic Latitude (b)", cb_orientation="horizontal",
                min=vmin, max=vmax, latitude_grid_spacing=45,
                projection_type="mollweide", title=title)
    if figname is not None:
        plt.savefig(figname, bbox_inches="tight")
        plt.close()
