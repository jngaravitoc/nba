import numpy as np
from nba.ios import load_halo, load_snapshot
from nba.visuals import Visuals
from nba.kinematics import Kinematics

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

    
    pos_com_host, vel_com_host = load_halo(snapshot+out_name+"_000", N_halo_part=[nhost, nsat], q=['mass'],  com_frame=0, galaxy=0, snapformat=3, com_method='diskpot')

    ## TODO: select particles from 50-500 kpc

    halo_kinematics = Kinematics(pos_com_host, vel_com_host)
    op_l, op_b = halo_kinematics.orbpole()

    halo_vis = Visuals()
    twd_map = halo_vis.mollweide_projection(op_l, op_b)
    fig, ax = plt.subplots(twd_map.T, origin='lower')
    
    
    #rcut  = np.where(np.sqrt(np.sum(pos_com_host**2, axis=1)) < 500)
    rcut  = np.where(np.sqrt(np.sum(pos_all**2, axis=1)) < 500)
    f = halo_vis.particle_slice(pos_all[rcut], 1000,  norm=None, grid_size=[-300, 300, -300, 300, -300, 300], cmap='magma')
    f.savefig('test_figure.png', bbox_inches='tight')
    
