.. module:: nba.com

*******************************
Halo centering (`nba.com`)
*******************************

Introduction
============

`nba.com` finds the center (position and velocity) of a halo from its
particles. `~nba.com.CenterHalo` takes a dictionary of particle arrays, as
returned by the :doc:`readers <../ios/index>` (``pos``, and depending on the
method ``vel``, ``mass`` and ``pot``), and offers three methods:

========================================================  ===========================================================
Method                                                    Use it for
========================================================  ===========================================================
`~nba.com.CenterHalo.shrinking_sphere_numba`              Any halo, and the only reliable choice for a satellite
`~nba.com.CenterHalo.min_potential`                       Host galaxies only (see :ref:`com-min-potential`)
`~nba.com.CenterHalo.mean_pos`                            A first guess, or a satellite before it is stripped
========================================================  ===========================================================

`~nba.com.CenterHalo.shrinking_sphere` is the NumPy version of the shrinking
sphere: it gives identical results, more slowly, and is kept as a reference.
`~nba.com.ssphere_numba` runs the shrinking sphere on arrays without a
`~nba.com.CenterHalo`.

The examples below use a halo made of random particles: a cuspy halo centered
at (3, -2, 1) kpc moving at (100, 0, 0) km/s, and a smaller clump 80 kpc
away, like a satellite near its host or debris::

    >>> import numpy as np
    >>> from nba.com import CenterHalo
    >>> rng = np.random.default_rng(1)
    >>> n = 200_000
    >>> directions = rng.normal(size=(n, 3))
    >>> directions /= np.linalg.norm(directions, axis=1)[:, None]
    >>> halo_pos = directions * (rng.pareto(1.5, n) * 5)[:, None] + [3.0, -2.0, 1.0]
    >>> clump_pos = rng.normal(size=(n // 10, 3)) * 3 + [80.0, 0.0, 0.0]
    >>> pos = np.vstack([halo_pos, clump_pos])
    >>> vel = np.vstack([rng.normal(size=(n, 3)) * 30 + [100.0, 0.0, 0.0],
    ...                  rng.normal(size=(n // 10, 3)) * 10])
    >>> halo = {"pos": pos, "vel": vel, "mass": np.ones(len(pos))}
    >>> center = CenterHalo(halo)

The shrinking sphere
====================

The shrinking sphere (`Power et al. 2003
<https://ui.adsabs.harvard.edu/abs/2003MNRAS.338...14P>`_) starts from a
sphere that holds all the particles, centered on their center of mass. At each
step it moves the sphere to the center of mass of the particles inside it and
shrinks its radius by 2.5%, until the next sphere would hold fewer than a
minimum number of particles::

    >>> com, vcom = center.shrinking_sphere_numba()
    >>> np.round(com, 1)
    array([ 3., -2.,  1.])

The mean position of all particles, in contrast, is pulled 7 kpc toward the
clump::

    >>> mean, _ = center.mean_pos()
    >>> round(float(np.linalg.norm(mean - com)), 1)
    7.0

All the options after ``delta`` are keyword-only.

When to stop
------------

``min_npart`` and ``npart_frac``
    The sphere stops before it holds fewer than ``min_npart`` particles or a
    fraction ``npart_frac`` (1%) of all the particles, whichever is smaller,
    as in Power et al. ``npart_frac=None`` removes the 1% cap.

``softening``
    The sphere also stops before its radius is smaller than 4 times the
    softening length. Inside that radius the density is flattened by the
    softening and the center is noisier. Give the softening of the particles
    whenever it is known.

``delta``
    Stops when the center moves less than ``delta`` between steps. Since each
    step shrinks the radius by only 2.5%, the center moves little per step
    even far from convergence, so this can stop early; it is off by default.

Starting from a previous center
-------------------------------

By default the first sphere holds all the particles. ``center0`` and ``r0``
start instead from a sphere of radius ``r0`` around ``center0``, typically the
center found in the previous snapshot. This keeps the sphere on the halo when
debris or another halo is nearby, and is faster::

    >>> com_tracked, _ = center.shrinking_sphere_numba(center0=[3.5, -2.0, 1.0], r0=15)
    >>> np.round(com_tracked, 1)
    array([ 3., -2.,  1.])
    >>> com_clump, _ = center.shrinking_sphere_numba(center0=[78.0, 0.0, 0.0], r0=15, softening=0.75)
    >>> np.round(com_clump) + 0.0  # + 0.0 turns -0. into 0.
    array([80.,  0.,  0.])

If the halo moves more than ``r0`` between snapshots the sphere can lose it,
so ``r0`` should be larger than the distance the halo travels in one snapshot
(10-15 kpc for the LMC in GC21).

The velocity
------------

The velocity is the mean velocity of the particles in a region around the
center:

- by default, the particles within ``rcut_vel`` (20) of the center;
- with ``rvel_factor``, the particles within ``rvel_factor`` times the radius
  of the final sphere;
- with ``nvel``, the ``nvel`` particles closest to the center.

For a satellite, use ``rvel_factor`` (e.g. 5) or ``nvel``: a fixed 20 kpc
sphere around a satellite contains debris moving differently from it, which
biases the velocity by up to ~100 km/s near pericenter::

    >>> _, vcom = center.shrinking_sphere_numba(rvel_factor=5)
    >>> np.round(vcom, -1) + 0.0
    array([100.,   0.,   0.])

Diagnostics
-----------

``return_info=True`` also returns a dictionary describing the final sphere::

    >>> com, vcom, info = center.shrinking_sphere_numba(softening=0.05, return_info=True)
    >>> sorted(info)
    ['density', 'niter', 'nmin', 'npart', 'nvel', 'radius', 'stop']
    >>> info["stop"], info["nmin"]  # the cusp is dense enough to reach 4 * softening first
    ('softening', 1000)

``radius``, ``npart``, ``density``
    Radius of the final sphere, and the number of particles and mean density
    within it, around the returned center. When a satellite is disrupted its
    core loses density: a final radius that grows, or a density that drops to
    ~1% of its initial value, means that the center is no longer well defined.
``nmin``
    The minimum number of particles used (``min_npart``, capped by
    ``npart_frac``).
``niter``, ``stop``
    The number of steps, and what stopped the shrinking: ``'min_npart'``,
    ``'softening'`` or ``'delta'``.
``nvel``
    The number of particles used for the velocity.

`nba.orbits.orbit` and `nba.orbits.iter_orbit` follow a halo through a series
of snapshots with these options, track it with ``r0``, and warn when the
center jumps, drifts against its velocity or loses its density (see the
`README <https://github.com/jngaravitoc/nba#centering-halos-and-computing-orbits>`_).

.. _com-min-potential:

The potential minimum
=====================

`~nba.com.CenterHalo.min_potential` averages the particles near the minimum of
the gravitational potential: those within ``rcut`` of the lowest-potential
particle, or with ``npart`` the ``npart`` particles with the lowest potential.
It needs ``pot`` in the halo dictionary, or a ``disk_pot`` argument, e.g. to
center the MW on its disk particles::

    >>> pot = np.linalg.norm(pos - [3.0, -2.0, 1.0], axis=1)  # a toy potential
    >>> com_pot, _ = CenterHalo(dict(halo, pot=pot)).min_potential(npart=1000)
    >>> np.round(com_pot)
    array([ 3., -2.,  1.])

.. warning::

   Use it for host galaxies only. Snapshots store the *total* potential of
   all the halos, so for a satellite the particles with the lowest potential
   are its stripped particles sitting in the host's potential well, not its
   center: in GC21 the LMC's "potential minimum" follows the MW from the
   first snapshot. Use the shrinking sphere for satellites.

The mean position
=================

`~nba.com.CenterHalo.mean_pos` returns the mean position and velocity,
mass-weighted if masses are given. ``rmin``, ``rmax`` and ``center`` restrict
it to a shell around a point::

    >>> mean_inner, _ = center.mean_pos(rmax=10, center=com)
    >>> round(float(np.linalg.norm(mean_inner - com)), 1)
    0.0

Recentering
===========

`~nba.com.CenterHalo.recenter` subtracts a center position (and velocity) from
the particles. It modifies the arrays in place, including those of the
dictionary given to `~nba.com.CenterHalo`, unless ``copy=True``::

    >>> center.recenter(com, vcom, copy=True)

API
===

.. automodapi:: nba.com
   :no-heading:
   :no-main-docstr:
   :headings: =-
