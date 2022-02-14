"""
Script to extract a particle orbit from a N-body halo simulation

"""

import numpy as np
#import pygadgetreader 
import sys
from gadget_reader import read_snap 


# This function is not used in this script, but it is helpful to build the particle ids file. 

def get_particle_id(pos, vel, ids, r_lim, dr, v_lim, dv):
	"""

	Selects particle ids based on positions and velocities. 
	This is useful to select particles on specific orbits. 

	"""
	r = np.sqrt(np.sum(pos**2, axis=1))
	v = np.sqrt(np.sum(vel**2, axis=1))
	index = np.where((r<r_lim+dr) & (r>r_lim-dr) & (v<v_lim+dv) & (v>v_lim-dv))
	return ids[index]


def get_snap_ids(snap, ids_p, i):
	"""

	Select ids of particles in a snapshot

	"""

    #os = read_snap(snap+'_{:0>3d}'.format(i), 'pos', 'dm')
    #ds = read_snap(snap+'_{:0>3d}'.format(i), 'pid', 'dm')
	# Read snapshot
	pos = read_snap(snap+'_{:0>3d}.hdf5'.format(i), 'PartType1', 'Coordinates')
	vel = read_snap(snap+'_{:0>3d}.hdf5'.format(i), 'PartType1', 'Velocities')
	ids = read_snap(snap+'_{:0>3d}.hdf5'.format(i), 'PartType1', 'ParticleIDs')
	
	# select ids 
	sort_ids = np.argsort(np.ascontiguousarray(ids))
	particles = np.in1d(np.ascontiguousarray(ids)[sort_ids], ids_p)
	print(ids_p, ids[particles], np.linspace(0, int(len(ids)-1), int(len(ids)))[particles])	
	#print(len(particles))
	pos_orbit = np.ascontiguousarray(pos)[sort_ids][particles]
	vel_orbit = np.ascontiguousarray(vel)[sort_ids][particles]
	assert len(pos_orbit) == len(ids_p), "something wrong with selecting particles in this snapshot" 
	return pos_orbit, vel_orbit

def extract_orbits(snap, snap_i, snap_f, ids_p):
	"""
	Extract particles orbits by selecting the particle ids in different snapshots

	"""
	N_snaps = snap_f - snap_i +1 
	N_part = len(ids_p)
	pos_orbits = np.zeros((N_snaps, N_part, 3))
	vel_orbits = np.zeros((N_snaps, N_part, 3))
	j=0
	for i in range(snap_i, snap_f+1):
		pos_orbits[j], vel_orbits[j] = get_snap_ids(snap, ids_p, i)
		j+=1

	return pos_orbits, vel_orbits
    
def get_orbits(out_name, snapname, init_snap, final_snap, ids_particles):
	"""
	Write all the orbtis. One file per particle orbit
	
	return
	0
	"""
	N_part = len(ids_particles)
	all_pos, all_vel = extract_orbits(snapname, init_snap, final_snap, ids_particles)
	assert N_part < 1000, "Currently only supporting up to 1000 orbits, if more are needed edit this function"

	for i in range(N_part):
		np.savetxt(out_name+"_particle_{:0>3d}.txt".format(i), np.array([all_pos[:,i,0], all_pos[:,i,1], all_pos[:,i,2],
																		all_vel[:,i,0], all_vel[:,i,1], all_vel[:,i,2]]).T)
	return 0  

 
if __name__ == "__main__":
	ids_file = sys.argv[1]
	snaps_file = sys.argv[2]
	out_path = sys.argv[3]
	out_name = sys.argv[4]
	snap_i = int(sys.argv[5])
	snap_f = int(sys.argv[6])
	ids_all = np.loadtxt(ids_file)	
	print("Done loading particle IDs")
	get_orbits(out_path+out_name, snaps_file, snap_i, snap_f, ids_all)
