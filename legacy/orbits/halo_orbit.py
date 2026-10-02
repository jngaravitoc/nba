import numpy as np
import sys
from nba.ios.io_snaps import halo_ids, load_snapshot
from nba.ios import get_com

def orbit(snapname, ninit, nfinal, com_frame, galaxy, N_halo_part, snapformat, com_method, rmin=0, rmax=0):
    """
    Computes the COM for a sequence of snapshots. 

    """
    pos_com = np.zeros((nfinal-ninit+1, 3))
    vel_com = np.zeros((nfinal-ninit+1, 3))

    for k in range(ninit, nfinal+1):
        all_ids = load_snapshot(snapname+'_{:03d}'.format(k), snapformat, 'pid', 'dm')
        print(len(all_ids))
        ids = halo_ids(all_ids, N_halo_part, galaxy)
        all_pos = load_snapshot(snapname+'_{:03d}'.format(k), snapformat, 'pos', 'dm')
        all_vel = load_snapshot(snapname+'_{:03d}'.format(k), snapformat, 'vel', 'dm')
        all_mass = load_snapshot(snapname+'_{:03d}'.format(k), snapformat, 'mass', 'dm')
        pos = all_pos[ids]
        vel = all_vel[ids]
        mass = all_mass[ids]
        pos_com[k-ninit], vel_com[k-ninit] = get_com(pos, vel, mass, com_method, snapname+'_{:03d}'.format(k), snapformat, rmin=rmin, rmax=rmax)
    return pos_com, vel_com

if __name__ == '__main__':

    # Define variables 
    # including the path of the snapshot
    snapshot = "/mnt/home/nico/ceph/gadget_runs/MWLMC/MWLMC5/out/" # sys.argv[1]
    out_name = 'MWLMC5_100M_b0_vir_OM3_G4' #sys.argv[2]
    init_snap = 0 #int(sys.argv[3])
    final_snap = 10 # int(sys.argv[4])
    snap_format = 3 # gadget4 - hdf5
    com_method1 = 'shrinking'
    com_method2 = 'diskpot'
    nhost=100000000
    nsat=15000000
    pos_com_host, vel_com_host = orbit(snapshot+out_name, init_snap, final_snap, 0, 0, [nhost, nsat], snap_format, com_method2)
    pos_com_sat, vel_com_sat = orbit(snapshot+out_name, init_snap, final_snap, 1, 1, [nhost, nsat], snap_format, com_method1)
    
    # Save data
    np.savetxt(out_name, np.array([pos_com_host[:,0], pos_com_host[:,1], pos_com_host[:,2],
				   vel_com_host[:,0], vel_com_host[:,1], vel_com_host[:,2],
                            	   pos_com_sat[:,0], pos_com_sat[:,1], pos_com_sat[:,2],
                                   vel_com_sat[:,0], vel_com_sat[:,1], vel_com_sat[:,2]]).T)

