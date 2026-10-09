"""
Center-of-mass orbits of GC21-style halos computed from a sequence of snapshots.
"""

import warnings

import numpy as np

from nba.com import CenterHalo
from nba.ios import ReadGadgetSim, ReadGC21

# Names of the CenterHalo methods. 'diskpot' applies min_potential to the disk
# particles and 'min_potential' to the dark matter particles; both are for the
# host (MW) only, since the snapshot potential is the total potential.
COM_METHODS = ("mean_pos", "shrinking_sphere", "shrinking_sphere_numba",
               "min_potential", "diskpot")
POTENTIAL_METHODS = ("min_potential", "diskpot")
SSPHERE_METHODS = ("shrinking_sphere", "shrinking_sphere_numba")

# Shorter names kept for backwards compatibility
ALIASES = {"mean": "mean_pos", "shrinking": "shrinking_sphere_numba"}


def _method_name(method):
    name = ALIASES.get(method, method)
    if name not in COM_METHODS:
        raise ValueError(f"com_method must be one of {COM_METHODS + tuple(ALIASES)}, got '{method}'")
    return name


def _center(data, method, rcut_pot, rcut_vel, center0=None, r0=None, ssphere_kwargs=None):
    """
    Apply one of the CenterHalo methods to the particles in `data`, returning
    ``(pos, vel, info)``. `center0`, `r0` and `ssphere_kwargs` are passed to
    the shrinking sphere methods.
    """
    center = CenterHalo(data)
    if method == "mean_pos":
        return (*center.mean_pos(), {"npart": len(data["pos"])})
    if method in SSPHERE_METHODS:
        return getattr(center, method)(rcut_vel=rcut_vel, center0=center0, r0=r0,
                                       return_info=True, **(ssphere_kwargs or {}))
    return center.min_potential(rcut=rcut_pot, return_info=True)  # 'min_potential' and 'diskpot'


def _check_step(k, method, prev, pos, vel, time, jump_factor):
    """Warn if the center moved more than `jump_factor` * |v| * dt since `prev`."""
    prev_pos, prev_vel, prev_time = prev
    dt = time - prev_time
    if jump_factor is None or dt <= 0:
        return
    step = np.linalg.norm(pos - prev_pos)
    expected = max(np.linalg.norm(prev_vel), np.linalg.norm(vel)) * dt
    if step > jump_factor * expected:
        warnings.warn(f"snapshot {k}, {method}: the center moved {step:.3g} in a time over which "
                      f"|v| * dt = {expected:.3g}; the center may have jumped to another object",
                      stacklevel=3)


def iter_orbit(path, snapname, snapshots, halo="MW", com_method="shrinking",
               rcut_pot=2.0, rcut_vel=20.0, randomsample=None, r0=None, softening=None, *,
               min_npart=1000, npart_frac=0.01, nvel=None, rvel_factor=None,
               time_offset=None, jump_factor=5.0, return_info=False):
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
        - 'min_potential': potential minimum of the dark matter particles
          (``halo='MW'`` only).
        - 'diskpot': potential minimum of the disk particles (``halo='MW'`` only).

        The potential methods are refused for the LMC: the snapshot potential
        is the total potential, so the lowest-potential LMC particles are
        stripped particles in the MW's potential well.

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
        do not shrink below 4 * softening. A warning is raised if a shrinking
        sphere method is used without it.
    min_npart, npart_frac, nvel, rvel_factor
        Passed to the shrinking sphere methods (see
        ``CenterHalo.shrinking_sphere``). ``nvel`` or ``rvel_factor`` compute
        the velocity from a region set by the final sphere instead of the
        fixed ``rcut_vel``.
    time_offset : float, callable or None
        Added to the header time: a constant, or a function of the snapshot
        number, e.g. to correct snapshots written after a restart that reset
        the time.
    jump_factor : float or None
        Warn when a center moves more than ``jump_factor * |v| * dt`` between
        consecutive snapshots, with |v| the larger of the two speeds. Assumes
        that velocity times time gives a position, as in Gadget code units.
        None disables the check. A warning is also raised when the time
        decreases.
    return_info : bool
        If True, each center is ``(pos_com, vel_com, info)``, with ``info``
        the dict returned by the method (see ``CenterHalo``; for 'mean_pos'
        only the number of particles).

    Yields
    ------
    snapshot : int
    time : float
        Simulation time from the snapshot header, plus ``time_offset``.
    centers : dict
        Maps each method (as given in ``com_method``) to ``(pos_com, vel_com)``.
    """
    methods = [com_method] if isinstance(com_method, str) else list(com_method)
    names = {m: _method_name(m) for m in methods}
    if halo not in ("MW", "LMC"):
        raise ValueError("halo must be one of ('MW', 'LMC')")
    potential = sorted(set(names.values()) & set(POTENTIAL_METHODS))
    if potential and halo != "MW":
        raise ValueError(f"com_method {potential} is only available for halo='MW'; "
                         "use the shrinking sphere for satellites")
    if softening is None and set(names.values()) & set(SSPHERE_METHODS):
        warnings.warn("No softening given: the shrinking sphere can stop inside the softened "
                      "core, where the center is noisier", stacklevel=2)
    ssphere_kwargs = dict(min_npart=min_npart, npart_frac=npart_frac, nvel=nvel,
                          rvel_factor=rvel_factor, softening=softening)

    use_disk = "diskpot" in names.values()
    use_dm = any(name != "diskpot" for name in names.values())
    quantities = ['pos', 'vel', 'mass'] + (['pot'] if "min_potential" in names.values() else [])

    previous = {}  # (pos, vel, time) found in the previous snapshot, per method
    prev_time = None
    for k in snapshots:
        file = snapname.format(k)
        time = float(ReadGadgetSim(path, file).read_header()["Time"])
        if time_offset is not None:
            time += time_offset(k) if callable(time_offset) else time_offset
        if prev_time is not None and time < prev_time:
            warnings.warn(f"snapshot {k}: the time decreases ({prev_time} -> {time}); "
                          "use time_offset if the simulation was restarted", stacklevel=2)
        prev_time = time
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
            track = r0 is not None and prev is not None
            pos, vel, info = _center(data, name, rcut_pot, rcut_vel,
                                     center0=prev[0] if track else None,
                                     r0=r0 if track else None, ssphere_kwargs=ssphere_kwargs)
            if prev is not None:
                _check_step(k, method, prev, pos, vel, time, jump_factor)
            previous[method] = (pos, vel, time)
            centers[method] = (pos, vel, info) if return_info else (pos, vel)
        yield k, time, centers


def orbit(path, snapname, snapshots, halo="MW", com_method="shrinking",
          rcut_pot=2.0, rcut_vel=20.0, randomsample=None, r0=None, softening=None, **kwargs):
    """
    Compute the center-of-mass position and velocity of a halo for a sequence
    of snapshots. See :func:`iter_orbit` for the parameters; its keyword-only
    parameters other than ``return_info`` are accepted too.

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
                       r0=r0, softening=softening, **kwargs)
    for i, (_, _, centers) in enumerate(steps):
        for m in methods:
            pos[m][i], vel[m][i] = centers[m]

    if single:
        return pos[com_method], vel[com_method]
    return {m: (pos[m], vel[m]) for m in methods}
