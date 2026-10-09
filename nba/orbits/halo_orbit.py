"""
Center-of-mass orbits of GC21-style halos computed from a sequence of snapshots.
"""

import inspect
import os
import warnings

import astropy.units as u
import h5py
import numpy as np

from nba.com import CenterHalo
from nba.ios import ReadGadgetSim, ReadGC21
from nba.orbits.orbit_io import DEFAULT_UNITS, write_orbit

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


def _warn(log, snap, method, message):
    """Raise a warning and, if `log` is a list, record it as (snap, method, message)."""
    warnings.warn(message, stacklevel=3)
    if log is not None:
        log.append((snap, method, message))


class _Track:
    """What iter_orbit remembers about one method between snapshots."""

    def __init__(self):
        self.pos = self.vel = self.time = None
        self.density0 = None  # density in the final sphere in the first snapshot
        self.inconsistent = 0  # consecutive snapshots failing the velocity check
        self.density_warned = False


def _velocity_error(track, pos, vel, time):
    """
    |dx - v dt| / (|v| dt) since the previous snapshot, with v the mean of
    the two velocities (NaN for the first snapshot or if dt <= 0).
    """
    dt = time - track.time
    vmean = 0.5 * (vel + track.vel)
    expected = np.linalg.norm(vmean) * dt
    if track.pos is None or dt <= 0 or expected == 0:
        return np.nan
    return float(np.linalg.norm(pos - track.pos - vmean * dt) / expected)


def iter_orbit(path, snapname, snapshots, halo="MW", com_method="shrinking",
               rcut_pot=2.0, rcut_vel=20.0, randomsample=None, r0=None, softening=None, *,
               min_npart=1000, npart_frac=0.01, nvel=None, rvel_factor=None,
               time_offset=None, jump_factor=5.0, velocity_tol=0.5, velocity_window=3,
               min_density_ratio=None, return_info=False, warning_log=None):
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
        consecutive snapshots, with |v| the larger of the two speeds. None
        disables the check. A warning is also raised when the time decreases.
    velocity_tol, velocity_window : float or None, int
        Warn when the step of a shrinking sphere center disagrees with its
        velocity, ``|dx - v dt| / (|v| dt) > velocity_tol`` with v the mean
        velocity of the two snapshots, for `velocity_window` consecutive
        snapshots. This catches a center that drifts in small steps once the
        satellite has dissolved, which the jump check misses. It is only
        applied when the velocity comes from a region tied to the final
        sphere (``nvel`` or ``rvel_factor``): the fixed ``rcut_vel`` sphere
        includes debris and fails it at pericenters. None disables the check.
    min_density_ratio : float or None
        Warn once, per shrinking sphere method, when the mean density in the
        final sphere (``info['density']``) falls below this fraction of its
        value in the first snapshot, e.g. 0.01: by then the satellite has
        dissolved and its center is no longer physical. Tracking continues.
    return_info : bool
        If True, each center is ``(pos_com, vel_com, info)``, with ``info``
        the dict returned by the method (see ``CenterHalo``; for 'mean_pos'
        only the number of particles), plus ``velocity_error``
        (``|dx - v dt| / (|v| dt)`` since the previous snapshot, NaN for the
        first) and, for the shrinking sphere, ``density_ratio`` (density
        relative to the first snapshot).
    warning_log : list or None
        If given, every warning is also appended to it as
        ``(snapshot, method, message)``, with None for warnings that do not
        concern one snapshot or method.

    Velocity times time is assumed to give a position, as in Gadget code
    units, by the jump and velocity checks.

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
        _warn(warning_log, None, None, "No softening given: the shrinking sphere can stop inside "
              "the softened core, where the center is noisier")
    ssphere_kwargs = dict(min_npart=min_npart, npart_frac=npart_frac, nvel=nvel,
                          rvel_factor=rvel_factor, softening=softening)
    check_velocity = velocity_tol is not None and (nvel is not None or rvel_factor is not None)

    use_disk = "diskpot" in names.values()
    use_dm = any(name != "diskpot" for name in names.values())
    quantities = ['pos', 'vel', 'mass'] + (['pot'] if "min_potential" in names.values() else [])

    tracks = {m: _Track() for m in methods}
    prev_time = None
    for k in snapshots:
        file = snapname.format(k)
        time = float(ReadGadgetSim(path, file).read_header()["Time"])
        if time_offset is not None:
            time += time_offset(k) if callable(time_offset) else time_offset
        if prev_time is not None and time < prev_time:
            _warn(warning_log, k, None, f"snapshot {k}: the time decreases ({prev_time} -> {time}); "
                  "use time_offset if the simulation was restarted")
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
            track = tracks[method]
            tracking = r0 is not None and track.pos is not None
            pos, vel, info = _center(data, name, rcut_pot, rcut_vel,
                                     center0=track.pos if tracking else None,
                                     r0=r0 if tracking else None, ssphere_kwargs=ssphere_kwargs)
            info["velocity_error"] = np.nan
            if track.pos is not None:
                _check_step(warning_log, k, method, track, pos, vel, time, jump_factor)
                info["velocity_error"] = _velocity_error(track, pos, vel, time)
            if name in SSPHERE_METHODS:
                if check_velocity:
                    _check_velocity(warning_log, k, method, track, info["velocity_error"],
                                    velocity_tol, velocity_window)
                _check_density(warning_log, k, method, track, info, min_density_ratio)
            track.pos, track.vel, track.time = pos, vel, time
            centers[method] = (pos, vel, info) if return_info else (pos, vel)
        yield k, time, centers


def _check_step(log, k, method, track, pos, vel, time, jump_factor):
    """Warn if the center moved more than `jump_factor` * |v| * dt since the last snapshot."""
    dt = time - track.time
    if jump_factor is None or dt <= 0:
        return
    step = np.linalg.norm(pos - track.pos)
    expected = max(np.linalg.norm(track.vel), np.linalg.norm(vel)) * dt
    if step > jump_factor * expected:
        _warn(log, k, method, f"snapshot {k}, {method}: the center moved {step:.3g} in a time over "
              f"which |v| * dt = {expected:.3g}; the center may have jumped to another object")


def _check_velocity(log, k, method, track, error, tol, window):
    """Warn when the velocity error stays above `tol` for `window` snapshots."""
    track.inconsistent = track.inconsistent + 1 if error > tol else 0
    if track.inconsistent == window:
        _warn(log, k, method, f"snapshot {k}, {method}: for {window} snapshots the center moved "
              f"inconsistently with its velocity (|dx - v dt| / (|v| dt) = {error:.3g} > {tol}); "
              "it may no longer follow a bound object")


def _check_density(log, k, method, track, info, min_ratio):
    """Add info['density_ratio'] and warn once when it falls below `min_ratio`."""
    if track.density0 is None:
        track.density0 = info["density"]
    ratio = info["density"] / track.density0 if track.density0 > 0 else np.nan
    info["density_ratio"] = ratio
    if min_ratio is not None and ratio < min_ratio and not track.density_warned:
        track.density_warned = True
        _warn(log, k, method, f"snapshot {k}, {method}: the density in the final sphere is "
              f"{ratio:.3g} of its value in the first snapshot (< {min_ratio}); the satellite may "
              "have dissolved and its center may no longer be physical")


def _snapshot_units(filename):
    """
    Unit labels for the orbit file and the code time unit in Gyr, from the
    Parameters of a Gadget snapshot. A code unit is labelled with the name in
    DEFAULT_UNITS when it matches it to 1e-3 (e.g. UnitLength_in_cm =
    3.085678e21 is kpc, UnitMass_in_g = 1.989e43 is 1e10 Msun), else with its value in cgs units. Positions and
    velocities are written as they are, so they are not rescaled.
    """
    with h5py.File(filename, "r") as f:
        if "Parameters" not in f:
            return None
        par = dict(f["Parameters"].attrs)
    cgs = {"length": ("UnitLength_in_cm", u.cm), "velocity": ("UnitVelocity_in_cm_per_s", u.cm / u.s),
           "mass": ("UnitMass_in_g", u.g)}
    units, values = {}, {}
    for name, (key, unit) in cgs.items():
        if key not in par:
            units[name] = None
            continue
        values[key] = float(par[key])
        quantity = float(par[key]) * unit
        ratio = quantity.to(DEFAULT_UNITS[name]).value
        units[name] = DEFAULT_UNITS[name] if abs(ratio - 1) < 1e-3 else quantity.to_string()
    time_to_gyr = None
    if "UnitLength_in_cm" in values and "UnitVelocity_in_cm_per_s" in values:
        time_to_gyr = (values["UnitLength_in_cm"] / values["UnitVelocity_in_cm_per_s"] * u.s).to(u.Gyr).value
    return units, values, time_to_gyr


def _info_columns(infos):
    """Per-snapshot info dicts -> {column: values}, splitting 3-vectors into _x, _y, _z."""
    columns = {}
    for key in infos[0]:
        values = [info[key] for info in infos]
        if np.ndim(values[0]) == 1:
            values = np.asarray(values, dtype=float)
            for i, a in enumerate("xyz"):
                columns[f"{key}_{a}"] = values[:, i]
        else:
            columns[key] = np.asarray(values)
    return columns


def _write_orbits(outfile, path, snapname, snapshots, halo, names, times, offsets, results,
                  log, parameters, simulation, notes, full_provenance):
    """Write one orbit file per method (see `orbit`)."""
    found = _snapshot_units(os.path.join(path, snapname.format(snapshots[0])))
    units, values, time_to_gyr = found if found else ({}, {}, None)
    units = {**{key: None for key in DEFAULT_UNITS}, **units,
             "time": "Gyr" if time_to_gyr else None}
    t_code = np.asarray(times)
    sim = {"snapshot_dir": path, "snapshot_pattern": snapname,
           "snapshots": f"{snapshots[0]}-{snapshots[-1]} ({len(snapshots)})",
           "units": {**values, "time_code_to_Gyr": time_to_gyr}, **(simulation or {})}
    for method, (pos, vel, infos) in results.items():
        name = names[method]
        selection = {"reader": "nba.ios.ReadGC21.read_halo", "halo": halo,
                     "ptype": "disk" if name == "diskpot" else "dm",
                     "randomsample": parameters["randomsample"]}
        if name != "diskpot":
            selection["rule"] = (f"the ReadGC21.npart_mw = {ReadGC21.npart_mw} dark matter particles "
                                 "with the lowest IDs are the MW, the others the LMC")
        params = dict(parameters, method=name)
        if name in SSPHERE_METHODS:
            params["velocity_region"] = (
                f"the {params['nvel']} particles closest to the center" if params["nvel"] is not None
                else f"particles within {params['rvel_factor']:g} times the final radius"
                if params["rvel_factor"] is not None
                else f"particles within rcut_vel = {params['rcut_vel']:g} of the center")
            params["center0"] = ("the center found in the previous snapshot"
                                 if params["r0"] is not None else "the global center of mass")
        method_log = [(k, msg) for k, m, msg in log if m in (None, method)]
        write_orbit(outfile.format(method=method), snapshots,
                    t_code * time_to_gyr if time_to_gyr else t_code, pos, vel,
                    t_code=t_code, time_offset=offsets, info=_info_columns(infos), units=units,
                    simulation=sim, selection=selection, method=name, parameters=params,
                    warnings=method_log, notes=notes, full_provenance=full_provenance)


def orbit(path, snapname, snapshots, halo="MW", com_method="shrinking",
          rcut_pot=2.0, rcut_vel=20.0, randomsample=None, r0=None, softening=None, *,
          outfile=None, simulation=None, notes=None, full_provenance=False, **kwargs):
    """
    Compute the center-of-mass position and velocity of a halo for a sequence
    of snapshots. See :func:`iter_orbit` for the parameters; its keyword-only
    parameters other than ``return_info`` and ``warning_log`` are accepted too.

    Parameters
    ----------
    outfile : str or None
        If given, also write the orbit of each method to an ECSV file (see
        :func:`nba.orbits.write_orbit`), with the diagnostics of each snapshot,
        the parameters, the warnings and the provenance. With several
        methods it must contain ``{method}``, e.g. ``"lmc_orbit_{method}.ecsv"``.
    simulation : dict or None
        Extra simulation metadata for the file, e.g. ``{"name": "MWLMC5_b0"}``.
    notes : str or None
        Free text stored in the file.
    full_provenance : bool
        Also record the nba path, user, host, SLURM job and command line in
        the file (see :func:`nba.orbits.provenance`).

    Returns
    -------
    If ``com_method`` is a string: ``pos_com, vel_com``, each of shape
    ``(len(snapshots), 3)``.

    If ``com_method`` is a sequence: a dict mapping each method to
    ``(pos_com, vel_com)``.
    """
    single = isinstance(com_method, str)
    methods = [com_method] if single else list(com_method)
    if outfile is not None and len(methods) > 1 and "{method}" not in outfile:
        raise ValueError("With several methods, outfile must contain '{method}'")

    snapshots = list(snapshots)
    pos = {m: np.zeros((len(snapshots), 3)) for m in methods}
    vel = {m: np.zeros((len(snapshots), 3)) for m in methods}
    infos = {m: [] for m in methods}
    times = []
    log = []

    steps = iter_orbit(path, snapname, snapshots, halo=halo, com_method=methods,
                       rcut_pot=rcut_pot, rcut_vel=rcut_vel, randomsample=randomsample,
                       r0=r0, softening=softening, return_info=True, warning_log=log, **kwargs)
    for i, (_, time, centers) in enumerate(steps):
        times.append(time)
        for m in methods:
            pos[m][i], vel[m][i], info = centers[m]
            infos[m].append(info)

    if outfile is not None:
        time_offset = kwargs.get("time_offset")
        offsets = [float(time_offset(k)) if callable(time_offset) else float(time_offset or 0.0)
                   for k in snapshots]
        # Every parameter as used, defaults included
        options = ("min_npart", "npart_frac", "nvel", "rvel_factor", "jump_factor",
                   "velocity_tol", "velocity_window", "min_density_ratio")
        defaults = inspect.signature(iter_orbit).parameters
        parameters = {"rcut_pot": rcut_pot, "rcut_vel": rcut_vel, "randomsample": randomsample,
                      "r0": r0, "softening": softening,
                      **{key: kwargs.get(key, defaults[key].default) for key in options},
                      "time_offset": (f"function of the snapshot number ({time_offset!r}), see the "
                                      "time_offset column" if callable(time_offset) else time_offset),
                      "shrink_floor": None if softening is None else 4 * softening}
        names = {m: _method_name(m) for m in methods}
        results = {m: (pos[m], vel[m], infos[m]) for m in methods}
        _write_orbits(outfile, path, snapname, snapshots, halo, names, times, offsets, results,
                      log, parameters, simulation, notes, full_provenance)

    if single:
        return pos[com_method], vel[com_method]
    return {m: (pos[m], vel[m]) for m in methods}
