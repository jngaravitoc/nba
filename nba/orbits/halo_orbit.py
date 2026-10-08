"""
Center-of-mass orbits of GC21-style halos computed from a sequence of snapshots.
"""

import numpy as np

from nba.com import CenterHalo
from nba.ios import ReadGadgetSim, ReadGC21

# Names of the CenterHalo methods. 'diskpot' applies min_potential to the disk
# particles (MW only); 'min_potential' applies it to the dark matter particles.
COM_METHODS = ("mean_pos", "shrinking_sphere", "shrinking_sphere_numba",
               "min_potential", "diskpot")

# Shorter names kept for backwards compatibility
ALIASES = {"mean": "mean_pos", "shrinking": "shrinking_sphere_numba"}


def _method_name(method):
    name = ALIASES.get(method, method)
    if name not in COM_METHODS:
        raise ValueError(f"com_method must be one of {COM_METHODS + tuple(ALIASES)}, got '{method}'")
    return name


def _center(data, method, rcut_pot, rcut_vel, center0=None, r0=None, softening=None):
    """
    Apply one of the CenterHalo methods to the particles in `data`.
    `center0`, `r0` and `softening` are passed to the shrinking sphere methods.
    """
    center = CenterHalo(data)
    if method == "mean_pos":
        return center.mean_pos()
    if method == "shrinking_sphere":
        return center.shrinking_sphere(rcut_vel=rcut_vel, center0=center0, r0=r0,
                                       softening=softening)
    if method == "shrinking_sphere_numba":
        return center.shrinking_sphere_numba(rcut_vel=rcut_vel, center0=center0, r0=r0,
                                             softening=softening)
    return center.min_potential(rcut=rcut_pot)  # 'min_potential' and 'diskpot'


def iter_orbit(path, snapname, snapshots, halo="MW", com_method="shrinking",
               rcut_pot=2.0, rcut_vel=20.0, randomsample=None, r0=None, softening=None):
    """
    Iterate over snapshots computing the center of mass of a halo with one or
    several methods. Each snapshot is read only once, whatever the number of
    methods, so results can be written as they are produced.

    Parameters
    ----------
    path : str
        Directory containing the snapshots.
    snapname : str
        Snapshot file name with a format field for the snapshot number,
        e.g. ``"MWLMC5_100M_b0_vir_OM3_G4_{:03d}.hdf5"``.
    snapshots : iterable of int
        Snapshot numbers to process.
    halo : {'MW', 'LMC'}
        Halo to follow (see ``nba.ios.ReadGC21``).
    com_method : str or sequence of str
        Any of ``COM_METHODS``:

        - 'mean_pos': mass-weighted mean of the halo dark matter particles.
        - 'shrinking_sphere': shrinking sphere (NumPy) on the dark matter particles.
        - 'shrinking_sphere_numba': shrinking sphere (Numba) on the dark matter particles.
        - 'min_potential': potential minimum of the dark matter particles.
        - 'diskpot': potential minimum of the disk particles (``halo='MW'`` only).

        'shrinking' and 'mean' are accepted as aliases of 'shrinking_sphere_numba'
        and 'mean_pos'.
    rcut_pot : float
        Radius (in the units of the snapshot) averaged around the potential
        minimum by 'min_potential' and 'diskpot'.
    rcut_vel : float
        Radius used to compute the COM velocity by the shrinking sphere methods.
    randomsample : int or None
        Number of particles randomly drawn from each snapshot.
    r0 : float or None
        If given, the shrinking sphere methods start each snapshot from a
        sphere of radius `r0` around the center they found in the previous
        snapshot (the first snapshot uses all particles). This keeps the
        sphere on the halo when another halo or its debris overlaps it.
        Snapshots should then be consecutive enough that the halo moves
        less than `r0` between them.
    softening : float or None
        Softening length of the halo particles. The shrinking sphere methods
        do not shrink below 4 * softening.

    Yields
    ------
    snapshot : int
    time : float
        Simulation time from the snapshot header.
    centers : dict
        Maps each method (as given in ``com_method``) to ``(pos_com, vel_com)``.
    """
    methods = [com_method] if isinstance(com_method, str) else list(com_method)
    names = {m: _method_name(m) for m in methods}
    if halo not in ("MW", "LMC"):
        raise ValueError("halo must be one of ('MW', 'LMC')")
    if "diskpot" in names.values() and halo != "MW":
        raise ValueError("com_method='diskpot' is only available for halo='MW'")

    use_disk = "diskpot" in names.values()
    use_dm = any(name != "diskpot" for name in names.values())
    quantities = ['pos', 'vel', 'mass'] + (['pot'] if "min_potential" in names.values() else [])

    previous = {}  # center found in the previous snapshot, per method
    for k in snapshots:
        file = snapname.format(k)
        time = ReadGadgetSim(path, file).read_header()["Time"]
        reader = ReadGC21(path, file)

        disk = dm = None
        if use_disk:
            disk = reader.read_halo(['pos', 'vel', 'mass', 'pot'], halo=halo,
                                    ptype='disk', randomsample=randomsample)
        if use_dm:
            dm = reader.read_halo(quantities, halo=halo, ptype='dm',
                                  randomsample=randomsample)

        centers = {}
        for method, name in names.items():
            data = disk if name == "diskpot" else dm
            prev = previous.get(method)
            centers[method] = _center(data, name, rcut_pot, rcut_vel,
                                      center0=prev, r0=r0 if prev is not None else None,
                                      softening=softening)
        if r0 is not None:
            previous = {m: c[0] for m, c in centers.items()}
        yield k, float(time), centers


def orbit(path, snapname, snapshots, halo="MW", com_method="shrinking",
          rcut_pot=2.0, rcut_vel=20.0, randomsample=None, r0=None, softening=None):
    """
    Compute the center-of-mass position and velocity of a halo for a sequence
    of snapshots. See :func:`iter_orbit` for the parameters.

    Returns
    -------
    If ``com_method`` is a string: ``pos_com, vel_com``, each of shape
    ``(len(snapshots), 3)``.

    If ``com_method`` is a sequence: a dict mapping each method to
    ``(pos_com, vel_com)``.
    """
    single = isinstance(com_method, str)
    methods = [com_method] if single else list(com_method)

    snapshots = list(snapshots)
    pos = {m: np.zeros((len(snapshots), 3)) for m in methods}
    vel = {m: np.zeros((len(snapshots), 3)) for m in methods}

    steps = iter_orbit(path, snapname, snapshots, halo=halo, com_method=methods,
                       rcut_pot=rcut_pot, rcut_vel=rcut_vel, randomsample=randomsample,
                       r0=r0, softening=softening)
    for i, (_, _, centers) in enumerate(steps):
        for m in methods:
            pos[m][i], vel[m][i] = centers[m]

    if single:
        return pos[com_method], vel[com_method]
    return {m: (pos[m], vel[m]) for m in methods}
