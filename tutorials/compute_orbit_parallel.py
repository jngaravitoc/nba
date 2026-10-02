"""
Parallel version of compute_orbit.py: each snapshot is processed by a worker
using Schwimmbad (multiprocessing or MPI).

Usage:
    python compute_orbit_parallel.py /path/to/snapshots --init 0 --final 400 --ncores 8
    mpiexec -n 8 python compute_orbit_parallel.py /path/to/snapshots --mpi
"""
from argparse import ArgumentParser
from functools import partial

import numpy as np
import schwimmbad

from nba.orbits import orbit


def worker(k, path, snapname):
    host = orbit(path, snapname, [k], halo="MW", com_method="diskpot")
    sat = orbit(path, snapname, [k], halo="LMC", com_method="shrinking")
    return np.concatenate([host[0][0], host[1][0], sat[0][0], sat[1][0]])


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Directory with the snapshots")
    parser.add_argument("--snapname", default="MWLMC5_100M_b0_vir_OM3_G4_{:03d}.hdf5",
                        help="Snapshot name with a format field for the snapshot number")
    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--final", type=int, default=10)
    parser.add_argument("--out", default="orbit_mwlmc5.txt")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1, type=int,
                       help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                       help="Run with MPI.")
    args = parser.parse_args()

    with schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores) as pool:
        # list() is needed: the serial pool returns a lazy map
        results = list(pool.map(partial(worker, path=args.path, snapname=args.snapname),
                                range(args.init, args.final + 1)))

    # Columns: MW (x, y, z, vx, vy, vz), LMC (x, y, z, vx, vy, vz)
    np.savetxt(args.out, np.array(results))
