"""
Compute the center of mass, angular momentum, anisotropy, density and enclosed
mass profiles of a DM halo from a sequence of Gadget-4 snapshots.

Usage:
    python halo_kinematics.py /path/to/snapshots LMC5_15M_vir_eps_100pc_ics2_{:03d}.hdf5 out_name \
        --init 0 --final 500 --step 100
"""
from argparse import ArgumentParser

import numpy as np

from nba.com import CenterHalo
from nba.ios import ReadGadgetSim
from nba.kinematics import Kinematics
from nba.structure import Profiles

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Directory with the snapshots")
    parser.add_argument("snapname", help="Snapshot name with a format field for the snapshot number")
    parser.add_argument("out_name", help="Prefix of the output files")
    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--final", type=int, default=0)
    parser.add_argument("--step", type=int, default=1, help="Snapshot stride")
    parser.add_argument("--nbins", type=int, default=100, help="Number of radial bins")
    parser.add_argument("--rmin", type=float, default=0.0)
    parser.add_argument("--rmax", type=float, default=120.0)
    parser.add_argument("--nsample", type=int, default=None,
                        help="Randomly sub-sample this many particles (faster)")
    args = parser.parse_args()

    snapshots = list(range(args.init, args.final + 1, args.step))
    edges = np.linspace(args.rmin, args.rmax, args.nbins + 1)
    rng = np.random.default_rng(0)

    com = np.zeros((len(snapshots), 3))
    vcom = np.zeros((len(snapshots), 3))
    ang_mom = np.zeros((len(snapshots), 3))
    beta = np.zeros((len(snapshots), args.nbins))
    density = np.zeros((len(snapshots), args.nbins))
    enclosed = np.zeros((len(snapshots), args.nbins))

    for i, k in enumerate(snapshots):
        print(f"Loading snapshot {k}")
        reader = ReadGadgetSim(args.path, args.snapname.format(k))
        halo = reader.read_snapshot(['pos', 'vel', 'mass'], ptype='dm')
        if args.nsample is not None:
            idx = rng.choice(len(halo['mass']), size=args.nsample, replace=False)
            halo = {key: value[idx] for key, value in halo.items()}

        # Center the halo on its shrinking-sphere center of mass
        center = CenterHalo(halo)
        com[i], vcom[i] = center.shrinking_sphere_numba()
        center.recenter(com[i], vcom[i])

        # Kinematics (nbins + 1 edges, i.e. the same radial grid as Profiles below)
        ang_mom[i] = Kinematics(halo['pos'], halo['vel']).total_angular_momentum()
        beta[i] = Kinematics(halo['pos'], halo['vel']).profiles(
            nbins=args.nbins + 1, quantity="beta", rmin=args.rmin, rmax=args.rmax)

        # Structure
        profiles = Profiles(halo['pos'], edges)
        r_centers, density[i] = profiles.density(mass=halo['mass'])
        _, enclosed[i] = profiles.enclosed_mass(halo['mass'])

    np.savetxt(args.out_name + "_com.txt", np.hstack([com, vcom]))
    np.savetxt(args.out_name + "_angular_momentum.txt", ang_mom)
    np.savetxt(args.out_name + "_beta.txt", beta)
    np.savetxt(args.out_name + "_dens_profile.txt", density)
    np.savetxt(args.out_name + "_encl_mass.txt", enclosed)
    np.savetxt(args.out_name + "_radii.txt", r_centers)
