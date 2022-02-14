"""
A simple gadget4 hdf5 reader

Nico Garavito-Camargo 2021

"""

import numpy 
import h5py

def read_snap(snap_name, partType, properties):
	"""
	partType: str 
		 partType1, partType2, following Gadget's convention of particles types
	properties : str 
		Acceleration, Coordinates, Masses, ParticleIDs, Potential, Velocities

	returns:

	numpy array : property 

	"""

	f = h5py.File(snap_name, 'r')
	print("Loading '{}' of particles types '{}' from snapshot: '{}'".format(properties, partType, snap_name))
	particles = f[partType]
	part_prop = particles.get(properties)
	print(particles.keys())
	return part_prop


def read_header(snap_name):
	## time 
	## 
	f = h5py.File(snap_name, 'r')
	print("Loading '{}' of particles types '{}' from snapshot: '{}'".format(properties, partType, snap_name))

def is_parttype_in_file(snap_name, partype):
	## Check if partype is present in file
	## 
	f = h5py.File(snap_name, 'r')
	if partype in f.keys():
		return True
	else:
		return False
