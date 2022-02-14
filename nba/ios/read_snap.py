import numpy as np
from pygadgetreader import readsnap
import sys
from nba.ios.gadget_reader import read_snap

def load_snapshot(snapname, snapformat, quantity, ptype):

	if snapformat == 1:
		q = readsnap(snapname, quantity, ptype)

	elif snapformat == 3:
		if quantity == 'pos':
			q = 'Coordinates'
				
		elif quantity == 'vel':
			q = 'Velocities'
	
		elif quantity == 'mass':
			q = 'Masses'
				
		elif quantity == 'pot':
			q = 'Potential'
				
		elif quantity == 'pid':
			q = 'ParticleIDs'

		elif quantity == 'acc':
			q = 'Acceleration'
				
		if ptype == "dm":
			ptype =	'PartType1'			
		if ptype == "disk":
			ptype =	'PartType2'			
		if ptype == "bulge":
			ptype =	'PartType3'
			
		q = read_snap(snapname+".hdf5", ptype, q)
		#a = read_snap(snapname, 'PartType1', 'Acceleration')
		#potential = read_snap(snapname, 'PartType1', 'Potential')
		#print(mass[0], a[0], potential[0])
	else : 
		print('Wrong snapshot format: (1) Gadget2/3, (2) ASCII, (3) Gadget4 (HDF5)')
		sys.exit()
	return np.ascontiguousarray(q)
