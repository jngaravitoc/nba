import numpy as np
import sys
from nba.ios.read_snap import load_snapshot


def orbit(snapname, ninit, nfinal, com_frame, galaxy, snapformat, com_method):
    """
    Computes the COM for a sequence of snapshots. 

    """
    pos_com = np.zeros((nfinal-ninit+1, 3))
    vel_com = np.zeros((nfinal-ninit+1, 3))

    for k in range(ninit, nfinal+1):
        all_ids = load_snapshot(snapname, snapformat, 'pid', 'dm')
        ids = halo_ids(all_ids, N_halo_part, galaxy)
        all_pos = load_snapshot(snapname, snapformat, 'pos', 'dm')
        all_vel = load_snapshot(snapname, snapformat, 'vel', 'dm')
        all_mass = load_snapshot(snapname, snapformat, 'mass', 'dm')
        pos = all_pos[ids]
        vel = all_vel[ids]
        mass = all_mass[ids]
        pos_com[k], vel_com[k] = com.get_com(pos, vel, mass, com_method)

    return pos_com, vel_com

if __name__ == '__main__':

    # Define variables 
    # including the path of the snapshot
    snapshot = sys.argv[1]
    out_name = sys.argv[2]
    init_snap = int(sys.argv[3])
    final_snap = int(sys.argv[4])
    snap_format = 3 # gadget4 - hdf5
    com_method1 = 'shrinking'
    com_method2 = 'diskpot'
    pos_com_host, vel_com_host = orbit(snapshot, init_snap, final_snap, 0, 0, snap_format, com_method2)
    pos_com_sat, vel_com_sat = orbit(snapshot, init_snap, final_snap, 1, 1, snap_format, com_method1)
    
    # Save data
    np.savetxt(out_name, np.array([pos_com_host[:,0], pos_com_host[:,1], pos_com_host[:,2],
				   vel_com_host[:,0], vel_com_host[:,1], vel_com_host[:,2],
                            	   pos_com_sat[:,0], pos_com_sat[:,1], pos_com_sat[:,2],
                                   vel_com_sat[:,0], vel_com_sat[:,1], vel_com_sat[:,2]]).T)

