"""
Compute the orbit of the LMC center of mass in a GC21 MW-LMC simulation with
each of the centering methods in nba.com.CenterHalo, and save one orbit file
per method.

Each snapshot is read once (LMC dark matter particles only); all methods are
applied to the same particles (see nba.orbits.iter_orbit). Output files are
named lmc_orbit_<method>.txt with columns: snap, time, x, y, z, vx, vy, vz.
One row is written (and flushed) per snapshot, so partial results survive if
the job is interrupted.

Note: ReadGC21.read_halo reads all 115M dark matter particles before selecting
the LMC, so each snapshot needs ~6-7 GB of memory.

Usage:
    # test on the first 3 snapshots
    python compute_lmc_orbits.py /path/to/snapshots --init 0 --final 2
    # full run, from --init to the last snapshot found in the directory
    python compute_lmc_orbits.py /path/to/snapshots
"""
import logging
import os
import re
import time
from argparse import ArgumentParser

import numpy as np

from nba.orbits import iter_orbit

# min_potential is not used: it finds the MW's potential well, not the LMC center
METHODS = ("mean_pos", "shrinking_sphere", "shrinking_sphere_numba")

logger = logging.getLogger(__name__)


def find_snapshots(path, snapname):
    """Snapshot numbers of the files in `path` that match `snapname` (e.g. 'snap_{:03d}.hdf5')."""
    prefix, suffix = re.match(r"(.*)\{[^}]*\}(.*)", snapname).groups()
    pattern = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + "$")
    matches = (pattern.match(f) for f in os.listdir(path))
    return sorted(int(m.group(1)) for m in matches if m)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="/mnt/home/nico/ceph/gadget_runs/MWLMC/MWLMC5/out/",
                        help="Directory with the snapshots")
    parser.add_argument("--snapname", default="MWLMC5_100M_b0_vir_OM3_G4_{:03d}.hdf5",
                        help="Snapshot name with a format field for the snapshot number")
    parser.add_argument("--init", type=int, default=0, help="First snapshot")
    parser.add_argument("--final", type=int, default=None,
                        help="Last snapshot (default: last one found in path)")
    parser.add_argument("--outdir", default=".", help="Directory for the orbit files")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--rcut-vel", type=float, default=20.0,
                        help="Radius used for the COM velocity by the shrinking sphere methods")
    parser.add_argument("--rvel-factor", type=float, default=None,
                        help="Use the particles within this factor times the final sphere radius "
                             "for the COM velocity instead of --rcut-vel")
    parser.add_argument("--softening", type=float, default=0.08,
                        help="Softening of the LMC dark matter particles (GC21: 0.08 kpc)")
    parser.add_argument("--r0", type=float, default=15.0,
                        help="Start each shrinking sphere within r0 of the previous center")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    available = find_snapshots(args.path, args.snapname)
    if not available:
        raise FileNotFoundError(f"No snapshots matching '{args.snapname}' in {args.path}")
    final = available[-1] if args.final is None else args.final
    snapshots = [k for k in available if args.init <= k <= final]
    missing = sorted(set(range(args.init, final + 1)) - set(snapshots))
    if missing:
        logger.warning(f"Missing snapshots (skipped): {missing}")
    logger.info(f"Processing {len(snapshots)} snapshots ({args.init}-{final}) with {args.methods}")

    os.makedirs(args.outdir, exist_ok=True)
    outfiles = {}
    for method in args.methods:
        outfiles[method] = open(os.path.join(args.outdir, f"lmc_orbit_{method}.txt"), "w")
        outfiles[method].write("# snap time x y z vx vy vz\n")

    try:
        t0 = time.perf_counter()
        orbit_steps = iter_orbit(args.path, args.snapname, snapshots, halo="LMC",
                                 com_method=args.methods, rcut_vel=args.rcut_vel,
                                 rvel_factor=args.rvel_factor, softening=args.softening,
                                 r0=args.r0)
        for k, sim_time, centers in orbit_steps:
            for method, (pos_com, vel_com) in centers.items():
                row = np.concatenate([[k, sim_time], pos_com, vel_com])
                outfiles[method].write(" ".join(f"{v:.6e}" for v in row) + "\n")
                outfiles[method].flush()
                logger.info(f"snap {k:03d} {method}: pos = {np.round(pos_com, 3)}")
            logger.info(f"snap {k:03d} done in {time.perf_counter() - t0:.1f} s")
            t0 = time.perf_counter()
    finally:
        for f in outfiles.values():
            f.close()
