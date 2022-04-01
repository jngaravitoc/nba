import numpy as np
from nba.orbits import orbit

if __name__ == "__main__":
    snapshot = "/mnt/home/nico/ceph/gadget_runs/MWLMC/MWLMC5/out/"
    out_name = 'MWLMC5_100M_b0_vir_OM3_G4'
    init_snap = 0 
    final_snap = 10 
    snap_format = 3 # gadget4 - hdf5
    com_method1 = 'shrinking'
    com_method2 = 'diskpot'
    nhost=100000000
    nsat=15000000
    orbit_name = "orbit_mwlmc5.txt"
    pos_com_host, vel_com_host = orbit(snapshot+out_name, init_snap, final_snap, 0, 0, [nhost, nsat], snap_format, com_method2)
    pos_com_sat, vel_com_sat = orbit(snapshot+out_name, init_snap, final_snap, 1, 1, [nhost, nsat], snap_format, com_method1)
    
    # Save data
    np.savetxt(orbit_name, np.array([pos_com_host[:,0], pos_com_host[:,1], pos_com_host[:,2],
                				   vel_com_host[:,0], vel_com_host[:,1], vel_com_host[:,2],
                            	   pos_com_sat[:,0], pos_com_sat[:,1], pos_com_sat[:,2],
                                   vel_com_sat[:,0], vel_com_sat[:,1], vel_com_sat[:,2]]).T)

