"""
Center-of-mass orbits of GC21-style halos computed from a sequence of snapshots.
"""

import numpy as np

from nba.com import CenterHalo
from nba.ios import ReadGC21

COM_METHODS = ("shrinking", "diskpot", "mean")


def orbit(path, snapname, snapshots, halo="MW", com_method="shrinking",
          rcut=None, randomsample=None):
    """
    Compute the center-of-mass position and velocity of a halo for a sequence
    of snapshots.

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
    com_method : {'shrinking', 'diskpot', 'mean'}
        - 'shrinking': shrinking sphere on the halo dark matter particles.
        - 'diskpot': potential minimum of the disk particles (``halo='MW'`` only).
        - 'mean': mass-weighted mean of the halo dark matter particles.
    rcut : float or None
        Radius (in the units of the snapshot) used to average around the
        potential minimum when ``com_method='diskpot'`` (default 2.0), and to
        compute the COM velocity when ``com_method='shrinking'`` (default 20.0).
    randomsample : int or None
        Number of particles randomly drawn from each snapshot.

    Returns
    -------
    pos_com, vel_com : ndarray, shape (len(snapshots), 3)
    """
    if com_method not in COM_METHODS:
        raise ValueError(f"com_method must be one of {COM_METHODS}")
    if com_method == "diskpot" and halo != "MW":
        raise ValueError("com_method='diskpot' is only available for halo='MW'")

    if rcut is None:
        rcut = 2.0 if com_method == "diskpot" else 20.0

    snapshots = list(snapshots)
    pos_com = np.zeros((len(snapshots), 3))
    vel_com = np.zeros((len(snapshots), 3))

    for i, k in enumerate(snapshots):
        reader = ReadGC21(path, snapname.format(k))
        if com_method == "diskpot":
            data = reader.read_halo(['pos', 'vel', 'mass', 'pot'], halo=halo,
                                    ptype='disk', randomsample=randomsample)
            pos_com[i], vel_com[i] = CenterHalo(data).min_potential(rcut=rcut)
        else:
            data = reader.read_halo(['pos', 'vel', 'mass'], halo=halo,
                                    ptype='dm', randomsample=randomsample)
            center = CenterHalo(data)
            if com_method == "shrinking":
                pos_com[i], vel_com[i] = center.shrinking_sphere_numba(rcut=rcut)
            else:
                pos_com[i], vel_com[i] = center.mean_pos()
    return pos_com, vel_com
