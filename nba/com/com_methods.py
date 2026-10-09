#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings

import numpy as np
from numba import njit

def _weighted_mean(values, weights=None):
    """Mean of `values` along the first axis, accumulated in float64."""
    if weights is None:
        return np.mean(values, axis=0, dtype=np.float64)
    total = np.sum(weights, dtype=np.float64)
    if total == 0:
        raise ValueError("Total mass is zero — cannot compute center of mass")
    return np.sum(values * weights[:, None], axis=0, dtype=np.float64) / total


def _renamed(kwargs, old, new, value):
    """Return the value of a renamed keyword, warning if the old name was used."""
    if old not in kwargs:
        return value
    warnings.warn(f"'{old}' is deprecated, use '{new}'", DeprecationWarning, stacklevel=3)
    return kwargs.pop(old)


def _initial_sphere(xyz, mass, center0=None, r0=None):
    """
    Starting center and radius of the shrinking sphere. Without `r0` the
    sphere is centered on the global center of mass (or `center0`) and
    encloses all particles. With `r0`, it is the COM of the particles within
    `r0` of `center0` (or of the global COM), and the radius is `r0`.
    """
    if center0 is None:
        center0 = _weighted_mean(xyz, mass)
    center0 = np.asarray(center0, dtype=np.float64)
    r2 = np.sum((xyz - center0)**2, axis=1)
    if r0 is None:
        return center0, float(np.sqrt(np.max(r2)))
    if r0 <= 0:
        raise ValueError("r0 must be positive")
    inside = r2 < r0**2
    if not np.any(inside):
        raise ValueError(f"No particles found within r0={r0} of center0={center0}")
    return _weighted_mean(xyz[inside], mass[inside]), float(r0)


# What stopped the shrinking sphere, as returned by the loops
_STOP_NPART, _STOP_DELTA, _STOP_SOFTENING = 0, 1, 2
_STOP_NAMES = ("min_npart", "delta", "softening")


# The Power et al. (2003) shrinking sphere: at each step the sphere is
# centered on the last barycentre and its radius reduced by 2.5%, so the
# radius is radius0 * 0.975**k, and all particles inside it are used,
# including those that were outside an earlier sphere. The loops stop before
# the radius drops below sqrt(r2_min) or the sphere holds fewer than `nmin`
# particles, or when the center moves less than `delta` (no early stop if
# delta < 0). They return the center, the number of particles and radius of
# the final sphere, the number of steps and what stopped it (_STOP_* code).
#
# To avoid looking at every particle at every step, positions and masses are
# copied in order of distance from a reference center. A particle inside a
# sphere of radius R around `c` is within R + |c - ref| of the reference, so
# each step only reads that leading block of the sorted copies (`prune=False`
# reads them all and is only used to test that this changes nothing). The
# reference is reset when the center moves more than R from it.

# Relative margin on the candidate radius, covering rounding in the distances
_PRUNE_MARGIN = 1e-9


def _sort_by_distance(xyz, mass, ref):
    """Positions, float64 masses and distances from `ref`, sorted by distance."""
    d2 = np.sum((xyz - ref)**2, axis=1)
    order = np.argsort(d2)
    return xyz[order], mass[order].astype(np.float64), np.sqrt(d2[order])


def _ssphere_power(xyz, mass, com_pos, radius, delta, nmin, r2_min, prune=True):
    """NumPy version of the Power et al. shrinking sphere (see the comment above)."""
    c = np.array(com_pos, dtype=np.float64)
    ref = c.copy()
    xs, ms, ds = _sort_by_distance(xyz, mass, ref)
    npart = np.searchsorted(ds, radius, side="left")
    niter = 0
    while True:
        new_radius = 0.975 * radius
        if new_radius**2 < r2_min:
            return c, npart, radius, niter, _STOP_SOFTENING

        drift = np.linalg.norm(c - ref)
        if drift > new_radius:
            ref = c.copy()
            xs, ms, ds = _sort_by_distance(xyz, mass, ref)
            drift = 0.0
        n = len(ds)
        if prune:
            n = np.searchsorted(ds, (new_radius + drift) * (1 + _PRUNE_MARGIN), side="right")

        inside = np.sum((xs[:n] - c)**2, axis=1) < new_radius**2
        count = np.count_nonzero(inside)
        if count < nmin:
            return c, npart, radius, niter, _STOP_NPART

        m = ms[:n][inside]
        new_c = np.sum(xs[:n][inside] * m[:, None], axis=0, dtype=np.float64) / np.sum(m)
        shift = np.linalg.norm(new_c - c)
        c, radius, npart = new_c, new_radius, count
        niter += 1
        if shift <= delta:
            return c, npart, radius, niter, _STOP_DELTA


@njit
def _sort_by_distance_numba(xyz, mass, ref):
    """Numba version of `_sort_by_distance`."""
    n = xyz.shape[0]
    d2 = np.empty(n)
    for i in range(n):
        dx = xyz[i, 0] - ref[0]
        dy = xyz[i, 1] - ref[1]
        dz = xyz[i, 2] - ref[2]
        d2[i] = dx*dx + dy*dy + dz*dz
    order = np.argsort(d2)
    xs = np.empty((n, 3), dtype=xyz.dtype)
    ms = np.empty(n)
    ds = np.empty(n)
    for i in range(n):
        k = order[i]
        xs[i, 0] = xyz[k, 0]
        xs[i, 1] = xyz[k, 1]
        xs[i, 2] = xyz[k, 2]
        ms[i] = mass[k]
        ds[i] = np.sqrt(d2[k])
    return xs, ms, ds


@njit
def _ssphere_kernel(xyz, mass, com_pos, radius, delta, nmin, r2_min, prune=True):
    """Numba version of the Power et al. shrinking sphere (see the comment above)."""
    c = com_pos.astype(np.float64)
    ref = c.copy()
    xs, ms, ds = _sort_by_distance_numba(xyz, mass, ref)
    npart = np.searchsorted(ds, radius, side="left")
    niter = 0
    while True:
        new_radius = 0.975 * radius
        if new_radius * new_radius < r2_min:
            return c, npart, radius, niter, 2  # _STOP_SOFTENING

        drift = np.sqrt(np.sum((c - ref)**2))
        if drift > new_radius:
            ref = c.copy()
            xs, ms, ds = _sort_by_distance_numba(xyz, mass, ref)
            drift = 0.0
        n = len(ds)
        if prune:
            n = np.searchsorted(ds, (new_radius + drift) * (1 + _PRUNE_MARGIN), side="right")

        # One pass: count the particles inside the sphere and sum their mass and m * x
        r2_cut = new_radius * new_radius
        count = 0
        msum = 0.0
        sx = 0.0
        sy = 0.0
        sz = 0.0
        for i in range(n):
            dx = xs[i, 0] - c[0]
            dy = xs[i, 1] - c[1]
            dz = xs[i, 2] - c[2]
            if dx*dx + dy*dy + dz*dz < r2_cut:
                m = ms[i]
                count += 1
                msum += m
                sx += xs[i, 0] * m
                sy += xs[i, 1] * m
                sz += xs[i, 2] * m
        if count < nmin:
            return c, npart, radius, niter, 0  # _STOP_NPART

        new_c = np.array([sx / msum, sy / msum, sz / msum])
        shift = np.sqrt(np.sum((new_c - c)**2))
        c = new_c
        radius = new_radius
        npart = count
        niter += 1
        if shift <= delta:
            return c, npart, radius, niter, 1  # _STOP_DELTA


def _com_velocity(vxyz, mass, r2, rcut_vel, nvel):
    """
    Mass-weighted velocity of the `nvel` particles closest to the center, or of
    all particles within `rcut_vel` of it if `nvel` is None, given the squared
    distances `r2` to the center. Returns the velocity and the number of
    particles, and raises a ValueError if no particles are selected.
    """
    if nvel is not None:
        if nvel <= 0:
            raise ValueError("nvel must be positive")
        sel = np.argpartition(r2, nvel - 1)[:nvel] if nvel < len(r2) else slice(None)
    else:
        sel = r2 < rcut_vel**2
    vel, m = vxyz[sel], mass[sel]
    if len(m) == 0:
        raise ValueError(f"No particles found within rcut_vel={rcut_vel} of the center")
    return _weighted_mean(vel, m), len(m)


def _shrinking_sphere(loop, xyz, vxyz, mass, delta, rcut_vel, min_npart, softening,
                      center0, r0, nvel, rvel_factor, npart_frac, return_info):
    """Shrinking sphere driver shared by the NumPy and Numba versions."""
    if nvel is not None and rvel_factor is not None:
        raise ValueError("Give at most one of nvel and rvel_factor")
    if rvel_factor is not None and rvel_factor <= 0:
        raise ValueError("rvel_factor must be positive")
    if softening is not None and softening < 0:
        raise ValueError("softening must be non-negative")
    com, radius = _initial_sphere(xyz, mass, center0, r0)
    # Power et al. (2003): min_npart particles or 1% of the halo, whichever is smaller
    nmin = min_npart if npart_frac is None else min(min_npart, int(npart_frac * len(mass)))
    nmin = max(1, nmin)
    r2_min = 0.0 if softening is None else float(4 * softening)**2
    com, npart, radius, niter, stop = loop(xyz, mass, com, radius,
                                           -1.0 if delta is None else float(delta),
                                           nmin, r2_min)
    r2 = np.sum((xyz - com)**2, axis=1)
    if rvel_factor is not None:
        rcut_vel = rvel_factor * radius
    com_vel, n_vel = _com_velocity(vxyz, mass, r2, rcut_vel, nvel)
    if not return_info:
        return com, com_vel
    mass_in = np.sum(mass[r2 < radius**2], dtype=np.float64)
    info = {
        "radius": float(radius),
        "npart": int(npart),
        "nmin": int(nmin),
        "niter": niter,
        "stop": _STOP_NAMES[stop],
        "density": float(mass_in / (4 / 3 * np.pi * radius**3)),
        "nvel": n_vel,
    }
    return com, com_vel, info


_SSPHERE_DOC = """
    Shrinking Sphere method (Power et al. 2003).

    Parameters
    ----------
    delta : float or None
        Optional early stop: end when the center moves less than `delta`
        between steps. None (default) shrinks down to the minimum number of
        particles. Each step shrinks the radius by only 2.5%, so the center
        moves little per step even when it is far from converged; a delta
        stop can end close to the global mean when a satellite is present.

    All other parameters are keyword-only.

    rcut_vel : float
        Radius around the center used to compute the COM velocity.
    min_npart : int
        The sphere stops shrinking before it holds fewer than `min_npart`
        particles or a fraction `npart_frac` of the halo particles, whichever
        is smaller.
    npart_frac : float or None
        Default 0.01, as in Power et al. (2003). None removes this cap, so
        that `min_npart` is used as given: a larger final sphere gives a
        steadier center when the core of a disrupting satellite loses its
        density, but departs from Power et al. The value used is returned in
        `info['nmin']`.
    softening : float, optional
        Gravitational softening length. The sphere stops shrinking before its
        radius drops below 4 * softening, where the softened cusp turns into a
        core and the center gets noisier. No radius limit if None.
    center0 : array-like, shape (3,), optional
        Initial guess of the center, e.g. the center found in the previous
        snapshot. Defaults to the global center of mass.
    r0 : float, optional
        Initial radius around `center0`. Defaults to a sphere enclosing all
        particles.
    nvel : int, optional
        If given, the COM velocity is computed from the `nvel` particles
        closest to the center instead of those within `rcut_vel`, which
        adapts the velocity region to the size of the halo.
    rvel_factor : float, optional
        If given, the COM velocity is computed from the particles within
        `rvel_factor` times the final sphere radius instead of `rcut_vel`, so
        that it describes the same region as the center. Cannot be combined
        with `nvel`.
    return_info : bool
        Also return a dict with the final sphere radius (`radius`), its
        number of particles (`npart`), the minimum number of particles used
        (`nmin`), the number of steps (`niter`), what stopped it (`stop`:
        'min_npart', 'softening' or 'delta'), the mean density within
        `radius` of the center (`density`) and the number of particles used
        for the velocity (`nvel`). A jump in `radius` or a drop in `density`
        between snapshots flags a center that is no longer well defined.

    The old names `minNpart` (and `rcut` in the Numba version) are accepted
    with a DeprecationWarning.

    Returns
    -------
    com_pos : np.ndarray, shape (3,)
        Center of mass position.
    com_vel : np.ndarray, shape (3,)
        Center of mass velocity. A ValueError is raised if no particles are
        selected for it.
    info : dict
        Only if `return_info` is True.
"""


def ssphere_numba(xyz, vxyz, mass, delta=None, *, rcut_vel=20.0, min_npart=1000,
                  center0=None, r0=None, nvel=None, rvel_factor=None, npart_frac=0.01,
                  return_info=False, softening=None):
    """
    Numba-accelerated shrinking sphere on arrays `xyz`, `vxyz` (N, 3) and
    `mass` (N,). See `CenterHalo.shrinking_sphere` for the other parameters.
    """
    return _shrinking_sphere(_ssphere_kernel, xyz, vxyz, mass, delta, rcut_vel, min_npart,
                             softening, center0, r0, nvel, rvel_factor, npart_frac,
                             return_info)



class CenterHalo:
    def __init__(self, Halo):
        self.pos = Halo['pos']
        self.vel = Halo.get('vel', None)
        self.mass = Halo.get('mass', None)
        self.pot = Halo.get('pot', None)  # Optional

    def _require(self, *names):
        """Raise a ValueError if any of the given attributes is missing."""
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            raise ValueError(f"Halo is missing required quantities: {missing}")

    def recenter(self, com, vcom=None, copy=False):
        """
        Subtract the center-of-mass position (and velocity, if given).

        By default this is done in place, so the arrays of the dictionary
        passed to CenterHalo are modified too. With `copy=True` new arrays
        are created and the original ones are left untouched.
        """
        if copy:
            self.pos = self.pos - com
        else:
            self.pos -= com
        if vcom is not None and self.vel is not None:
            if copy:
                self.vel = self.vel - vcom
            else:
                self.vel -= vcom

    def min_potential(self, disk_pot=None, rcut: float = 2.0, npart=None, return_info=False):
        """
        Center-of-mass position and velocity near the potential minimum.

        Use it for host galaxies only. The snapshot potential is the total
        potential, so for a satellite the lowest-potential particles are its
        stripped particles sitting in the host's potential well, not its
        center; use the shrinking sphere for satellites.

        By default the particles within `rcut` of the lowest-potential
        particle are averaged. If `npart` is given, the `npart` particles
        with the lowest potential are averaged instead, which does not depend
        on the noise of a single particle or on a fixed radius. `disk_pot`
        overrides the halo potential. A warning is raised when fewer than
        10 particles are averaged.

        With `return_info`, a dict with the position of the lowest-potential
        particle (`anchor`), its potential (`pot_min`) and the number of
        particles averaged (`npart`) is also returned.
        """
        if disk_pot is None:
            disk_pot = self.pot
        if disk_pot is None:
            raise ValueError("min_potential needs 'pot' in the halo or a disk_pot argument")
        self._require("vel")

        imin = np.argmin(disk_pot)
        if npart is not None:
            if npart <= 0:
                raise ValueError("npart must be positive")
            npart = min(npart, len(disk_pot))
            idx = np.argpartition(disk_pot, npart - 1)[:npart]
        else:
            # Distance to potential minimum
            r = np.linalg.norm(self.pos - self.pos[imin], axis=1)
            idx = np.where(r < rcut)[0]
        if len(idx) < 10:
            warnings.warn(f"min_potential averages only {len(idx)} particles", stacklevel=2)

        weights = None if self.mass is None else self.mass[idx]
        com, vcom = _weighted_mean(self.pos[idx], weights), _weighted_mean(self.vel[idx], weights)
        if not return_info:
            return com, vcom
        info = {"anchor": np.array(self.pos[imin], dtype=np.float64),
                "pot_min": float(disk_pot[imin]), "npart": len(idx)}
        return com, vcom, info

    def velocities_com(self, cm_pos: np.ndarray, r_cut: float = 20.0) -> np.ndarray:
        """Compute the (mass-weighted, if masses are given) COM velocity within `r_cut` of `cm_pos`."""
        self._require("vel")
        dist = np.linalg.norm(self.pos - cm_pos, axis=1)
        mask = dist < r_cut
        if not np.any(mask):
            raise ValueError(f"No particles found within r_cut={r_cut} of cm_pos")
        weights = None if self.mass is None else self.mass[mask]
        return _weighted_mean(self.vel[mask], weights)

    def mean_pos(self, rmin: float = 0, rmax=None, center=None):
        """
        Mean position and velocity, mass-weighted if masses are given. Only
        particles with rmin <= r < rmax are used, where r is measured from
        `center` (the coordinate origin if not given). `rmax` None (or 0) means
        no upper limit, so the default uses all particles.
        """
        self._require("vel")
        if rmax == 0:
            rmax = None
        if rmin < 0 or (rmax is not None and rmax < 0):
            raise ValueError("rmin and rmax must be non-negative")
        if rmax is not None and rmin > rmax:
            raise ValueError("rmin must be less than or equal to rmax")

        if rmin == 0 and rmax is None:
            return self.mean_pos_from_arrays(self.pos, self.vel, self.mass)

        origin = np.zeros(3) if center is None else np.asarray(center)
        r = np.linalg.norm(self.pos - origin, axis=1)
        mask = r >= rmin
        if rmax is not None:
            mask &= r < rmax
        if not np.any(mask):
            raise ValueError("No particles found in the rmin–rmax range")
        weights = None if self.mass is None else self.mass[mask]
        return self.mean_pos_from_arrays(self.pos[mask], self.vel[mask], weights)

    @staticmethod
    def mean_pos_from_arrays(pos: np.ndarray, vel: np.ndarray, mass=None):
        """Helper to compute COM from given arrays (unweighted if `mass` is None)."""
        return _weighted_mean(pos, mass), _weighted_mean(vel, mass)

    def shrinking_sphere(self, delta=None, *, rcut_vel=20.0, min_npart=1000, center0=None,
                         r0=None, nvel=None, rvel_factor=None, npart_frac=0.01,
                         return_info=False, softening=None, **kwargs):
        min_npart = _renamed(kwargs, "minNpart", "min_npart", min_npart)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self._require("vel", "mass")
        return _shrinking_sphere(_ssphere_power, self.pos, self.vel, self.mass, delta,
                                 rcut_vel, min_npart, softening, center0, r0, nvel,
                                 rvel_factor, npart_frac, return_info)

    def shrinking_sphere_numba(self, delta=None, *, rcut_vel=20.0, min_npart=1000, center0=None,
                               r0=None, nvel=None, rvel_factor=None, npart_frac=0.01,
                               return_info=False, softening=None, **kwargs):
        min_npart = _renamed(kwargs, "minNpart", "min_npart", min_npart)
        rcut_vel = _renamed(kwargs, "rcut", "rcut_vel", rcut_vel)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self._require("vel", "mass")
        return _shrinking_sphere(_ssphere_kernel, self.pos, self.vel, self.mass, delta,
                                 rcut_vel, min_npart, softening, center0, r0, nvel,
                                 rvel_factor, npart_frac, return_info)

    shrinking_sphere.__doc__ = ("NumPy version, kept as a reference: `shrinking_sphere_numba` gives\n"
                                "    the same result about 13x faster on large halos.\n" + _SSPHERE_DOC)
    shrinking_sphere_numba.__doc__ = "Numba-accelerated version of `shrinking_sphere`.\n" + _SSPHERE_DOC
