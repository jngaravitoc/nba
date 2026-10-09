"""
Self-describing orbit files.

An orbit is written as an astropy ECSV table: plain text, one row per
snapshot, with units on every column and a YAML metadata block in the
header holding the provenance (nba version and commit, environment,
command), the simulation, the particle selection, every centering parameter
as used and the warnings raised during the run.

The columns are ``snap t x y z vx vy vz``, then the times in code units and
the per-snapshot diagnostics of the method. ``read_orbit`` returns an
astropy Table with the units and metadata restored. With plain numpy, skip
the header: the column-name line of ECSV is not commented, e.g.
``np.loadtxt(path, skiprows=n_comment_lines + 1)``, or use
``read_orbit(path, as_array=True)``.
"""

import datetime
import getpass
import os
import platform
import socket
import subprocess
import sys

import astropy.units as u
import numpy as np
from astropy.table import Table

FORMAT_VERSION = "0.1"

# Default units of the position, velocity, time and mass columns
DEFAULT_UNITS = {"length": "kpc", "velocity": "km / s", "time": "Gyr", "mass": "1e10 solMass"}

DESCRIPTIONS = {
    "snap": "snapshot number",
    "t": "time",
    "x": "center position", "y": "center position", "z": "center position",
    "vx": "center velocity", "vy": "center velocity", "vz": "center velocity",
    "t_code": "time in code units (header time + time_offset)",
    "time_offset": "offset added to the header time (code units)",
    "radius": "final shrinking-sphere radius",
    "npart": "particles within radius of the center",
    "nmin": "minimum number of particles used",
    "niter": "shrinking steps",
    "stop": "what stopped the shrinking",
    "density": "mean density within radius of the center",
    "density_ratio": "density relative to the first snapshot",
    "nvel": "particles used for the velocity",
    "velocity_error": "|dx - v dt| / (|v| dt) since the previous snapshot",
}


def _git(path, *args):
    try:
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def provenance():
    """
    Software and environment that produced a file: nba version, path and git
    commit (None for installs that are not git checkouts), whether the
    checkout has uncommitted changes, Python and numpy versions, date, user,
    host, SLURM job and command.
    """
    import nba
    nba_path = os.path.dirname(os.path.abspath(nba.__file__))
    repo = os.path.dirname(nba_path)
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain", "--untracked-files=no") if commit else None
    return {
        "nba_version": getattr(nba, "__version__", None),
        "nba_path": nba_path,
        "nba_commit": commit,
        "nba_branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD") if commit else None,
        "nba_dirty": (status != "") if status is not None else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "command": " ".join(sys.argv),
    }


def _quantity(values, unit):
    values = np.asarray(values)
    if unit is None or values.dtype.kind not in "fi":
        return values
    return values * u.Unit(unit)


def write_orbit(path, snap, t, pos, vel, *, t_code=None, time_offset=None, info=None,
                units=None, simulation=None, selection=None, method=None, parameters=None,
                warnings=None, notes=None, provenance_info=None):
    """
    Write an orbit as an ECSV table.

    Parameters
    ----------
    path : str
        Output file (.ecsv).
    snap : array of int, shape (N,)
    t : array, shape (N,)
        Time.
    pos, vel : arrays, shape (N, 3)
        Center position and velocity.
    t_code : array, optional
        Time in code units (header time plus offset).
    time_offset : array, optional
        Offset added to each snapshot's header time (code units), stored per
        snapshot because a function of the snapshot number cannot be saved.
    info : dict of arrays, optional
        Per-snapshot diagnostics, e.g. the ``info`` dicts of the shrinking
        sphere (``radius`` and ``density`` get length and density units).
    units : dict, optional
        Units of ``length``, ``velocity``, ``time`` and ``mass`` (astropy
        unit strings, or None for values without units). Defaults to
        ``DEFAULT_UNITS``.
    simulation, selection, parameters : dict, optional
        Simulation (path, snapshot pattern, units...), particle selection and
        the centering parameters as used, defaults included.
    method : str, optional
    warnings : list of (snap, message), optional
    notes : str, optional
    provenance_info : dict, optional
        Provenance recorded when the orbit was computed. Defaults to
        `provenance()`, which is only right when the file is written by the
        run itself.

    Returns
    -------
    astropy.table.Table
    """
    units = {**DEFAULT_UNITS, **(units or {})}
    length, velocity, mass = units["length"], units["velocity"], units["mass"]
    density = (u.Unit(mass) / u.Unit(length)**3 if mass is not None and length is not None
               else None)
    info_units = {"radius": length, "density": density}

    pos, vel = np.asarray(pos, dtype=float), np.asarray(vel, dtype=float)
    cols = {"snap": np.asarray(snap, dtype=int),
            "t": _quantity(np.asarray(t, dtype=float), units["time"])}
    for i, a in enumerate("xyz"):
        cols[a] = _quantity(pos[:, i], length)
    for i, a in enumerate("xyz"):
        cols[f"v{a}"] = _quantity(vel[:, i], velocity)
    if t_code is not None:
        cols["t_code"] = np.asarray(t_code, dtype=float)
    if time_offset is not None:
        cols["time_offset"] = np.asarray(time_offset, dtype=float)
    for key, values in (info or {}).items():
        cols[key] = _quantity(values, info_units.get(key))

    table = Table(cols)
    for name in table.colnames:
        if name in DESCRIPTIONS:
            table[name].description = DESCRIPTIONS[name]
    table.meta = {
        "format": f"nba orbit file {FORMAT_VERSION}",
        "provenance": provenance_info or provenance(),
        "simulation": simulation or {},
        "selection": selection or {},
        "method": method,
        "parameters": parameters or {},
        "warnings": [{"snap": None if k is None else int(k), "message": str(m)}
                     for k, m in (warnings or [])],
    }
    if notes:
        table.meta["notes"] = notes
    table.write(path, format="ascii.ecsv", overwrite=True)
    return table


def read_orbit(path, as_array=False):
    """
    Read an orbit file written by `write_orbit`: an astropy Table with units
    and the metadata in ``.meta``, or a numpy structured array (without
    units or metadata) if `as_array` is True.
    """
    table = Table.read(path, format="ascii.ecsv")
    return table.as_array() if as_array else table
