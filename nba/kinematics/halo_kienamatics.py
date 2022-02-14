"""
Script to compute halo kinematics from Gadget4 outputs.

Usage : python3 halo_kinematics.py snap_path out_name init_snap final_snap

"""

import numpy as np
import sys
sys.path.append("../src/")
from read_snap import load_snapshot
import shrinking_sphere as ssphere
from kinematics import Kinematics
from structure import Structure

# Load gagdet files 


# Compute kinematics

# Compute angular momentu,
# Compute COM
# Compute \Beta(r)
# Compute M(r)
# Compute \rho(r)




# Define variables 

if __name__ == "__main__":
	snapshot = sys.argv[1]
	out_name = sys.argv[2]
	init_snap = int(sys.argv[3])
	final_snap = int(sys.argv[4])
	snap_format = 3 # gadget4 - hdf5
	nsnaps = final_snap - init_snap + 1
	
	# Profile properties
	nbins = 100
	rmin = 0
	rmax = 120
	dsnaps = 5 # skips snapshots

	# Halo properties:
	angular_momentum = False
	com = False
	shrinking_sphere = False
	beta_profile = False
	mass_profile = True
	density_profile = True
	potential_profile = False
	enclosed_mass = True
	
	# initialize arrays
	halo_com = np.zeros((nsnaps, 3))
	halo_vcom = np.zeros((nsnaps, 3))
	halo_in_com = np.zeros((nsnaps, 3))
	halo_in_vcom = np.zeros((nsnaps, 3))
	Lx = np.zeros(nsnaps)
	Ly = np.zeros(nsnaps)
	Lz = np.zeros(nsnaps)
	beta_halo = np.zeros((nsnaps, nbins-1))
	dens_profile = np.zeros((nsnaps, nbins-1))
	pot_profile = np.zeros((nsnaps, nbins-1))
	encl_mass  = np.zeros((nsnaps, nbins-1))


	# Load snaphots 

	for k in range(init_snap, final_snap+1, dsnaps):
		print('Loading snapshot \n')
		pos, vel, mass = load_snapshot(snapshot+"_{:03d}.hdf5".format(k), snap_format)
	
		# Compute COM
		if com == True:
			print('Computing COM \n')
			halo_com[k-init_snap], halo_vcom[k-init_snap] = ssphere.com(pos, vel, mass)
		if shrinking_sphere == True:
			print('Computing COM with shrinking sphere method \n')
			halo_in_com[k-init_snap], halo_in_vcom[k-init_snap] = ssphere.shrinking_sphere(pos, vel, mass, r_init=300)
			print('Done computing COM \n')
		
		# Halo kinematics
		halo_kin = Kinematics(pos, vel)

		if angular_momentum == True:
			print('Computing angular momentum \n')
			Lx[k-init_snap], Ly[k-init_snap], Lz[k-init_snap] = halo_kin.total_angular_momentum()
		
		if beta_profile == True:
			print('Computing anisotropy profile \n')
			beta_halo[k-init_snap] = halo_kin.profiles(nbins=nbins, quantity="beta", rmin=rmin, rmax=rmax)

		# Halo structure
		halo_structure = Structure(pos, mass)

		if enclosed_mass == True:
			encl_mass[k-init_snap] = halo_structure.enclosed_mass(mass, nbins, rmin, rmax)

		if density_profile == True:
			dens_profile[k-init_snap] = halo_structure.density_profile(nbins, rmin, rmax)
			
		if potential_profile == True:
			pot_profile[k-init_snap] = halo_structure.potential_profile(nbins, rmin, rmax)

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
