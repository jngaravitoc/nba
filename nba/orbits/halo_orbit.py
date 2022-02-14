import numpy as np
import sys

from read_snap import load_snapshot
import shrinking_sphere as ssphere

# Define variables 
# including the path of the snapshot
snapshot = sys.argv[1]
out_name = sys.argv[2]
init_snap = int(sys.argv[3])
final_snap = int(sys.argv[4])

snap_format = 3 # gadget4 - hdf5
halo_com = np.zeros((final_snap-init_snap+1, 3))
halo_vcom = np.zeros((final_snap-init_snap+1, 3))
halo_in_com = np.zeros((final_snap-init_snap+1, 3))
halo_in_vcom = np.zeros((final_snap-init_snap+1, 3))

# Load snaphot
for k in range(init_snap, final_snap+1):
	print('loading snapshot \n')
	pos, vel, mass = load_snapshot(snapshot+"_{:03d}.hdf5".format(k), snap_format)
	# Compute COM
	print('computing COM \n')
	halo_com[k], halo_vcom[k] = ssphere.com(pos, vel, mass)
	print('computing COM with shrinkins sphere method \n')
	halo_in_com[k], halo_in_vcom[k] = ssphere.shrinking_sphere(pos, vel, mass, r_init=300)
	print('Done computing COM \n')
# Save data
np.savetxt(out_name, np.array([halo_com[:,0], halo_com[:,1], halo_com[:,2],
							   halo_vcom[:,0], halo_vcom[:,1], halo_vcom[:,2],
							   halo_in_com[:,0], halo_in_com[:,1], halo_in_com[:,2],
                               halo_in_vcom[:,0],halo_in_vcom[:,1], halo_in_com[:,2]]).T)

