"""
Compute the orbital pole distribution of the MW dark matter halo (50-500 kpc)
for a sequence of GC21 snapshots and save a Mollweide map for each one.

Requires healpy (``pip install astro-nba[extra]``).

Usage:
    python orbital_poles_computation.py /path/to/snapshots --init 325 --final 399
"""
from argparse import ArgumentParser

import numpy as np

from nba.com import CenterHalo
from nba.ios import ReadGC21
from nba.kinematics import Kinematics
from nba.visuals.mollweide import healpix_density_map, plot_mollweide_galactic

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Directory with the snapshots")
    parser.add_argument("--snapname", default="MWLMC5_100M_b0_vir_OM3_G4_{:03d}.hdf5",
                        help="Snapshot name with a format field for the snapshot number")
    parser.add_argument("--init", type=int, default=325)
    parser.add_argument("--final", type=int, default=399)
    parser.add_argument("--rmin", type=float, default=50.0)
    parser.add_argument("--rmax", type=float, default=500.0)
    parser.add_argument("--nside", type=int, default=48)
    parser.add_argument("--randomsample", type=int, default=None,
                        help="Randomly sub-sample this many halo particles")
    args = parser.parse_args()

    for k in range(args.init, args.final + 1):
        reader = ReadGC21(args.path, args.snapname.format(k))

        # Center on the disk potential minimum
        disk = reader.read_halo(['pos', 'vel', 'mass', 'pot'], halo='MW', ptype='disk')
        com, vcom = CenterHalo(disk).min_potential()

        halo = reader.read_halo(['pos', 'vel'], halo='MW', ptype='dm',
                                randomsample=args.randomsample)
        CenterHalo(halo).recenter(com, vcom)

        # Select particles in the requested radial range
        d_host = np.linalg.norm(halo['pos'], axis=1)
        rcut = (d_host > args.rmin) & (d_host < args.rmax)

        op_l, op_b = Kinematics(halo['pos'][rcut], halo['vel'][rcut]).orbpole()

        hpx_map = healpix_density_map(op_l, op_b, nside=args.nside, smooth=5)
        plot_mollweide_galactic(hpx_map, figname="OP_MWLMC5_{:03d}.png".format(k))
