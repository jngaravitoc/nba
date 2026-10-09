.. module:: nba.ios

******************************
Snapshot readers (`nba.ios`)
******************************

Introduction
============

`nba.ios` reads particle data from simulation snapshots into dictionaries of
numpy arrays, the format used by the rest of ``nba``:

=====================================  =======================================================
Reader                                 Simulations
=====================================  =======================================================
`~nba.ios.ReadGadgetSim`               Any Gadget-4 HDF5 snapshot
`~nba.ios.ReadGC21`                    The MW-LMC simulations of Garavito-Camargo et al. (2021)
`~nba.ios.ReadSheng24`                 The MW-LMC simulations of Sheng et al. (2024)
=====================================  =======================================================

All readers take the folder and the file name of a snapshot, including its
``.hdf5`` extension.

Quantities and particle types
-----------------------------

Quantities are named as in the rest of ``nba``:

============  ==================  ==========
nba name      HDF5 dataset        Shape
============  ==================  ==========
``pos``       ``Coordinates``     (N, 3)
``vel``       ``Velocities``      (N, 3)
``mass``      ``Masses``          (N,)
``pot``       ``Potential``       (N,)
``pid``       ``ParticleIDs``     (N,)
``acc``       ``Acceleration``    (N, 3)
============  ==================  ==========

and the particle types are ``'dm'`` (``PartType1``), ``'disk'``
(``PartType2``) and ``'bulge'`` (``PartType3``). Arrays keep the units and the
data type of the file (e.g. float32 positions); ``nba`` accumulates sums in
float64 where it matters.

Getting started
===============

The examples below use a small file written with h5py in the layout of a
Gadget-4 snapshot: 3000 dark matter particles with IDs 0 to 2999, positions in
kpc and velocities in km/s::

    >>> import os, tempfile
    >>> import h5py
    >>> import numpy as np
    >>> folder = tempfile.mkdtemp()
    >>> rng = np.random.default_rng(0)
    >>> with h5py.File(os.path.join(folder, "snap_000.hdf5"), "w") as f:
    ...     f.create_group("Header").attrs["Time"] = 0.5
    ...     dm = f.create_group("PartType1")
    ...     dm["Coordinates"] = rng.normal(size=(3000, 3)).astype(np.float32)
    ...     dm["Velocities"] = rng.normal(size=(3000, 3)).astype(np.float32)
    ...     dm["Masses"] = np.full(3000, 1e-6, dtype=np.float32)
    ...     dm["ParticleIDs"] = np.arange(3000)

Gadget-4 snapshots
------------------

`~nba.ios.ReadGadgetSim` reads the header and any quantities of one particle
type::

    >>> from nba.ios import ReadGadgetSim
    >>> snap = ReadGadgetSim(folder, "snap_000.hdf5")
    >>> snap.read_header()
    {'Time': np.float64(0.5)}
    >>> snap.has_parttype("PartType2")  # no disk in this file
    False
    >>> dm = snap.read_snapshot(["pos", "vel", "mass"], ptype="dm")
    >>> sorted(dm), dm["pos"].shape, dm["pos"].dtype.name
    (['mass', 'pos', 'vel'], (3000, 3), 'float32')

`~nba.ios.ReadGadgetSim.open_snap` reads datasets by their names in the file
instead, including ones without an ``nba`` name.

GC21 simulations
----------------

In the GC21 simulations the dark matter of both the MW and the LMC is in
``PartType1``. `~nba.ios.ReadGC21.read_halo` selects one of them by particle ID:
the `~nba.ios.ReadGC21.npart_mw` (100 million) particles with the lowest IDs
are the MW, the others the LMC. The disk and bulge belong to the MW::

    >>> from nba.ios import ReadGC21
    >>> reader = ReadGC21("/path/to/snapshots", "MWLMC5_100M_b0_vir_OM3_G4_110.hdf5")  # doctest: +SKIP
    >>> lmc = reader.read_halo(["pos", "vel", "mass"], halo="LMC", ptype="dm")  # doctest: +SKIP
    >>> disk = reader.read_halo(["pos", "vel", "mass", "pot"], halo="MW", ptype="disk")  # doctest: +SKIP

``pid`` is always returned. To try it on the small file, use a reader with a
smaller MW (here the 2000 lowest IDs)::

    >>> class SmallGC21(ReadGC21):
    ...     npart_mw = 2000
    >>> lmc = SmallGC21(folder, "snap_000.hdf5").read_halo(["pos", "mass"], halo="LMC", ptype="dm")
    >>> len(lmc["pos"]), int(lmc["pid"].min())
    (1000, 2000)

``randomsample`` draws that many particles at random, without replacement, from
the selected halo, and ``seed`` makes the draw reproducible::

    >>> sample = SmallGC21(folder, "snap_000.hdf5").read_halo(
    ...     ["pos"], halo="LMC", ptype="dm", randomsample=500, seed=1)
    >>> len(np.unique(sample["pid"]))
    500

.. note::

   `~nba.ios.ReadGC21.read_halo` reads all the dark matter particles before
   selecting a halo, so it needs the memory of the whole snapshot: about
   6-7 GB for the 115 million particles of GC21 with positions, velocities and
   masses, whatever ``randomsample`` is.

Sheng et al. (2024) simulations
-------------------------------

`~nba.ios.ReadSheng24` selects the MW or the LMC dark matter by particle mass:
the snapshots must have exactly two dark matter particle masses, and the more
numerous particles are the MW. ``randomsample`` is not supported.

Adding a reader
===============

Readers for other simulations go in ``nba/ios/snap_reader.py``. Like
`~nba.ios.ReadGC21`, a reader can use `~nba.ios.ReadGadgetSim.read_snapshot`
to read the file and then select particles, and should return a dictionary
with the names above, so that its output works with `nba.com` and the other
modules.

API
===

.. automodapi:: nba.ios
   :no-heading:
   :no-main-docstr:
   :headings: =-
