#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('pylab', 'inline')
import sys
sys.path.append("../../../codes/nbody_analysis/src/")
from gadget_reader import read_snap
import gala.potential as pd
from astropy import units as u




path = "/mnt/home/nico/ceph/HQ_iso_halo/iso_softening_100pc_grav_MO3/LMC5_15M_vir_eps_100pc_ics2_grav_MO3_000.hdf5"




pos = read_snap(path, "PartType1", "Coordinates")
vel = read_snap(path, "PartType1", "Velocities")
pid = read_snap(path, "PartType1", "ParticleIDs")


# Select particles with near circular velocities


halo_potential = pd.HernquistPotential(1.8e11, 25.2, units=[u.Msun, u.kpc, u.Gyr, u.radian])




vc = np.zeros(20)
r = np.arange(20, 401, 20)
#j=0
for j in range(len(r)):
    vc[j] = halo_potential.circular_velocity([0, 0, r[j]]*u.kpc).to(u.km/u.s).value



halo_r = np.sqrt(np.sum(np.ascontiguousarray(pos)**2, axis=1))
vel_r = np.sqrt(np.sum(np.ascontiguousarray(vel)**2, axis=1))




dr=(r[1]-r[0])/2.
Delta_v = 0.01
all_pids = np.zeros_like(r)
np_rand = 20
for k in range(len(r)):
    part_r = np.where(((halo_r<r[k]+dr) & (halo_r > r[k]-dr)))
    vc_cut = np.where(np.abs((vel_r[part_r] - vc[k])) < Delta_v)[0]
    n_part = len(vc_cut)
    rand_part = np.random.randint(0, n_part, np_rand)
    id_rand_part = np.ascontiguousarray(pid)[part_r][vc_cut][rand_part]
    all_pids[k] = id_rand_part[0]
    #print(np.ascontiguousarray(pos)[part_r][vc_cut][rand_part])
    #print(np.ascontiguousarray(vel)[part_r][vc_cut][rand_part])



np.savetxt('pids_circv_LMC5_15M_vir_eps_100pc_ics2_grav_M03_000.txt', all_pids)





