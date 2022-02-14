import numpy as np
#from pygadgetreader import readsnap
import sys
from gadget_reader import read_snap

def load_snapshot(snapname, snapformat, masscol=3):
	if snapformat == 1:
		pos = readsnap(snapname, 'pos', 'dm')
		vel = readsnap(snapname, 'vel', 'dm')
		mass = readsnap(snapname, 'mass', 'dm')
	elif snapformat == 2:
		snap = np.loadtxt(snapname)
		pos = snap[:,0:3]
		mass = snap[:,masscol]
	elif snapformat == 3:
		pos = read_snap(snapname, 'PartType1', 'Coordinates')
		vel = read_snap(snapname, 'PartType1', 'Velocities')
		mass = read_snap(snapname, 'PartType1', 'Masses')
		#a = read_snap(snapname, 'PartType1', 'Acceleration')
		#potential = read_snap(snapname, 'PartType1', 'Potential')
		ids = read_snap(snapname, 'PartType1', 'ParticleIDs')
		#print(mass[0], a[0], potential[0])
	else : 
		print('Wrong snapshot format: (1) Gadget, (2) ASCII')
		sys.exit()
	return np.ascontiguousarray(pos), np.ascontiguousarray(vel), np.ascontiguousarray(mass)
