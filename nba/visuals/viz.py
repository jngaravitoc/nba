import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import sys
from read_snap import load_snapshot

def scatter(pos, npart_sample, box_size=[-100, 100, -100, 100], figure_name = 0):
	# Variables 
	Nparticles = len(pos)

	# Colors
	scatter_color = (1, 1, 1)
	axis_colors = (1, 0, 0)
	bgrd_color = 'k'
	marker_size = 3
	transparency = 1
	
	# Define figure	
	fig, ax = plt.subplots(1, 1, figsize=(8,8), facecolor=bgrd_color)
	
	# Axis cnd background olors 
	ax.patch.set_facecolor(bgrd_color)
	ax.tick_params(axis='x', colors=axis_colors)
	ax.tick_params(axis='y', colors=axis_colors)
	ax.yaxis.label.set_color(axis_colors)
	ax.xaxis.label.set_color(axis_colors)
	ax.spines['bottom'].set_color((1, 1, 1))
	ax.spines['top'].set_color((1, 1, 1)) 
	ax.spines['right'].set_color((1, 1, 1))
	ax.spines['left'].set_color((1, 1, 1))
	
	# Figure axis limits	
	ax.set_xlim(box_size[0], box_size[1])
	ax.set_ylim(box_size[1], box_size[2])


	# figure plot 
	rand_p = np.random.randint(0, Nparticles, npart_sample)
	ax.scatter(pos[rand_p,0], pos[rand_p, 1], c=scatter_color, s=marker_size, alpha=transparency)

	if type(figure_name) == str:
		plt.savefig(figure_name, bbox_inches='tight', dpi=80)
		print("here") 
	else:
		plt.show()
	
	# Loop over snapshots

def animate(snapshot, snap_format, init, final, fig_name=0, npart=10000):
	for k in range(final-init+1):
		pos, vel, mass = load_snapshot(snapshot+"_{:03d}.hdf5".format(k), snap_format)
		scatter(pos, npart, figure_name=fig_name+"{:03d}.png".format(k))
	return 0


if __name__ == "__main__":	
	# Define variables 
	# including the path of the snapshot
	snapshot = sys.argv[1]
	out_name = sys.argv[2]
	init_snap = 0 #t(sys.argv[3])
	final_snap = 20 #int(sys.argv[4])
	snap_format = 3 # gadget4 - hdf5
	npart = 100000
	n_snaps = final_snap - init_snap 
	animate(snapshot, snap_format, init_snap, final_snap, fig_name=out_name, npart=npart)
	#s, vel, mass = load_snapshot(snapshot, snap_format, init_snap, final_snap, out_name, npart)
	#catter(pos, npart, figure_name=fig_name+"{:03d}.png".format(k))
	#animate(pos, npart)



