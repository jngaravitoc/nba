"""
Compute the orbits of the MW and the LMC center of mass from a sequence of
GC21 snapshots.

Usage:
    python compute_orbit.py /path/to/snapshots --init 0 --final 10
"""
from argparse import ArgumentParser

import numpy as np

from nba.orbits import orbit

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Directory with the snapshots")
    parser.add_argument("--snapname", default="MWLMC5_100M_b0_vir_OM3_G4_{:03d}.hdf5",
                        help="Snapshot name with a format field for the snapshot number")
    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--final", type=int, default=10)
    parser.add_argument("--out", default="orbit_mwlmc5.txt")
    args = parser.parse_args()

    snapshots = range(args.init, args.final + 1)

    # The MW is centered on the disk potential minimum, the LMC with a shrinking sphere.
    pos_host, vel_host = orbit(args.path, args.snapname, snapshots, halo="MW", com_method="diskpot")
    pos_sat, vel_sat = orbit(args.path, args.snapname, snapshots, halo="LMC", com_method="shrinking")

    # Columns: MW (x, y, z, vx, vy, vz), LMC (x, y, z, vx, vy, vz)
    np.savetxt(args.out, np.hstack([pos_host, vel_host, pos_sat, vel_sat]))
