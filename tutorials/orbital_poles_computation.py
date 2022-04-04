import numpy as np
from nba.ios import load_halo, load_snapshot
from nba.visuals import Visuals
from nba.kinematics import Kinematics

if __name__ == "__main__":
    snapshot = "/mnt/home/nico/ceph/gadget_runs/MWLMC/MWLMC5/out/"
    snapname = 'MWLMC5_100M_b0_vir_OM3_G4'
    nhost=100000000
    nsat=15000000
    
    for k in range(325, 400):
        pos_com_host, vel_com_host = load_halo(snapshot+snapname+"_{:03d}".format(k), N_halo_part=[nhost, nsat], q=['mass'],  com_frame=0, galaxy=0, snapformat=3, com_method='diskpot')

        ## TODO: select particles from 50-500 kpc
        d_host = np.sqrt(np.sum(pos_com_host**2, axis=1))
        rcut = np.where( (d_host < 500) & (d_host > 50))

        halo_kinematics = Kinematics(pos_com_host[rcut], vel_com_host[rcut])
        op_l, op_b = halo_kinematics.orbpole()

        halo_vis = Visuals()
        twd_map, hpx_map = halo_vis.compute_mollweide(op_l, op_b, nside=48)
        #twd_map = halo_vis.compute_mollweide(op_l, op_b, nside=68)
        figname = "OP_MWLMC5_100M_b0_OM3_{:03d}.png".format(k)
        halo_vis.plot_mollweide_galactic(hpx_map, rotation=(180, 0, 0), bmin=1150, bmax=1450, figname=figname, fig_title="")

    
  
