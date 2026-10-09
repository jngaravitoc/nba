***************
Getting started
***************

A typical analysis reads the particles of a halo from a snapshot, finds the
halo's center, and moves the particles to a frame centered on it.

Reading a halo
==============

`nba.ios` has readers for Gadget-4 HDF5 snapshots. `~nba.ios.ReadGC21` also
selects the particles of the MW or the LMC in the GC21 simulations::

    >>> from nba.ios import ReadGC21
    >>> reader = ReadGC21("/path/to/snapshots", "MWLMC5_100M_b0_vir_OM3_G4_110.hdf5")  # doctest: +SKIP
    >>> lmc = reader.read_halo(["pos", "vel", "mass"], halo="LMC", ptype="dm",
    ...                        randomsample=1_000_000, seed=1)  # doctest: +SKIP

``lmc`` is a dictionary of arrays: ``lmc["pos"]`` and ``lmc["vel"]`` have shape
(N, 3), ``lmc["mass"]`` shape (N,). See :doc:`ios/index`.

Finding the center
==================

`~nba.com.CenterHalo` finds the center of the particles in such a dictionary.
Here a halo made of random particles stands in for ``lmc``: a cuspy halo
centered at (3, -2, 1) kpc, plus a smaller clump 80 kpc away, as for a
satellite with debris or a neighbor::

    >>> import numpy as np
    >>> from nba.com import CenterHalo
    >>> rng = np.random.default_rng(1)
    >>> n = 200_000
    >>> directions = rng.normal(size=(n, 3))
    >>> directions /= np.linalg.norm(directions, axis=1)[:, None]
    >>> halo_pos = directions * (rng.pareto(1.5, n) * 5)[:, None] + [3.0, -2.0, 1.0]
    >>> clump_pos = rng.normal(size=(n // 10, 3)) * 3 + [80.0, 0.0, 0.0]
    >>> pos = np.vstack([halo_pos, clump_pos])
    >>> halo = {"pos": pos, "vel": np.zeros_like(pos), "mass": np.ones(len(pos))}

The shrinking sphere finds the center of the halo, while the mean position is
pulled toward the clump::

    >>> center = CenterHalo(halo)
    >>> com, vcom = center.shrinking_sphere_numba()
    >>> np.round(com, 1)
    array([ 3., -2.,  1.])
    >>> mean, _ = center.mean_pos()
    >>> np.round(mean)
    array([10., -2.,  1.])

Recentering
===========

`~nba.com.CenterHalo.recenter` subtracts the center from the positions (and the
velocity from the velocities, if given). By default it modifies the arrays in
place, including those of the dictionary; ``copy=True`` leaves them untouched::

    >>> center.recenter(com, vcom, copy=True)
    >>> np.round(center.pos[0] - halo["pos"][0], 1)  # shifted by -com; halo["pos"] unchanged
    array([-3.,  2., -1.])

Next steps
==========

- :doc:`com/index` explains the centering methods and their options, and
  which one to use for a host galaxy or a satellite.
- The :doc:`tutorials` apply them to the GC21 simulations.
