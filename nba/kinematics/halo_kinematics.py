"""
Script to compute halo kinematics fnd structure from Gadget4 outputs.

Usage : python3 halo_kinematics.py snap_path out_name 

"""

import numpy as np
import sys
sys.path.append("../src/")
from read_snap import load_snapshot
import shrinking_sphere as ssphere
from kinematics import Kinematics
from structure import Structure



# Define variables 

if __name__ == "__main__":
	snapshot = "/mnt/ceph/users/nico/HQ_iso_halo/iso_softening_100pc/LMC5_15M_vir_eps_100pc_ics2"
	out_name = "LMC5_15M_vir_eps_100pc"
	init_snap = 0
	final_snap = 500
	snap_format = 3 # gadget4 - hdf5
	nsnaps = final_snap - init_snap + 1
	dsnaps = 100 # skips snapshots
	
	# Profile properties
	nbins = 101
	rmin = 0
	rmax = 120

	# Halo properties:
	angular_momentum = True
	com = True
	shrinking_sphere = False
	beta_profile = True
	mass_profile = True
	density_profile = True
	potential_profile = False
	enclosed_mass = True
	
	ntotal = int(nsnaps/dsnaps) + 1
	# initialize arrays
	halo_com = np.zeros((ntotal, 3))
	halo_vcom = np.zeros((ntotal, 3))
	halo_in_com = np.zeros((ntotal, 3))
	halo_in_vcom = np.zeros((ntotal, 3))
	Lx = np.zeros(ntotal)
	Ly = np.zeros(ntotal)
	Lz = np.zeros(ntotal)
	beta_halo = np.zeros((ntotal, nbins-1))
	dens_profile = np.zeros((ntotal, nbins-1))
	pot_profile = np.zeros((ntotal, nbins-1))
	encl_mass  = np.zeros((ntotal, nbins-1))

	r = np.linspace(rmin, rmax, nbins-1)
	# Load snaphots 

	ns = 0
	for k in range(init_snap, final_snap+1, dsnaps):
		print('Loading snapshot \n')
		pos, vel, mass = load_snapshot(snapshot+"_{:03d}.hdf5".format(k), snap_format)
	
		# Compute COM
		if com == True:
			print('Computing COM \n')
			halo_com[ns], halo_vcom[ns] = ssphere.com(pos, vel, mass)
		if shrinking_sphere == True:
			print('Computing COM with shrinking sphere method \n')
			halo_in_com[ns], halo_in_vcom[ns] = ssphere.shrinking_sphere(pos, vel, mass, r_init=300)
			print('Done computing COM \n')
		
		# Halo kinematics
		halo_kin = Kinematics(pos, vel)

		if angular_momentum == True:
			print('Computing angular momentum \n')
			Lx[ns], Ly[ns], Lz[ns] = halo_kin.total_angular_momentum()
		
		if beta_profile == True:
			print('Computing anisotropy profile \n')
			beta_halo[ns] = halo_kin.profiles(nbins=nbins, quantity="beta", rmin=rmin, rmax=rmax)

		# Halo structure
		halo_structure = Structure(pos, mass)

		if enclosed_mass == True:
			encl_mass[ns] = halo_structure.enclosed_mass(mass, nbins, rmin, rmax)

		if density_profile == True:
			dens_profile[ns] = halo_structure.density_profile(nbins, rmin, rmax)
			
		if potential_profile == True:
			pot_profile[ns] = halo_structure.potential_profile(nbins, rmin, rmax)

		ns+=1
	# Save data
	if ((com == True) | (shrinking_sphere == True)):
		np.savetxt(out_name+"_com.txt", np.array([halo_com[:,0], halo_com[:,1], halo_com[:,2],
												halo_vcom[:,0], halo_vcom[:,1], halo_vcom[:,2],
												halo_in_com[:,0], halo_in_com[:,1], halo_in_com[:,2],
												halo_in_vcom[:,0],halo_in_vcom[:,1], halo_in_com[:,2]]).T)

	if angular_momentum == True:
		np.savetxt(out_name+"_kinematics.txt", np.array([Lx, Ly, Lz]).T)
	if beta_profile == True:
		np.savetxt(out_name+"_beta.txt", beta_halo)

	if enclosed_mass == True:
		np.savetxt(out_name+"_encl_mass.txt", encl_mass)

	if density_profile == True:
		np.savetxt(out_name+"_dens_profile.txt", dens_profile)

	if potential_profile == True:	
		np.savetxt(out_name+"_pot_profile.txt", pot_profile)
