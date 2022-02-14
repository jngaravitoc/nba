import numpy as np
import matplotlib.pyplot as plt
import sys

def kinematics_plot(r, t, com, vcom, com_in, vcom_in, beta, L, m_encl, rho, name):
	fig, ax = plt.subplots(2, 3, figsize=(18, 10))

	# COM 
	ax[0][0].plot(t, com[:,0], c='C0')
	ax[0][0].plot(t, com[:,1], c='C1')
	ax[0][0].plot(t, com[:,2], c='C2')

	# VCOM
	ax[0][1].plot(t, vcom[:,0], c='C0', label=r'$vcom_x$')
	ax[0][1].plot(t, vcom[:,1], c='C1', label=r'$vcom_y$')
	ax[0][1].plot(t, vcom[:,2], c='C2', label=r'$vcom_z$')
	
	ax[0][1].legend()
	# Angular momentum
	ax[0][2].plot(t, L[:,0], c='C0')
	ax[0][2].plot(t, L[:,1], c='C1')
	ax[0][2].plot(t, L[:,2], c='C2')

	#sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.normalize(min=0, max=1))
	#plt.colorbar(sm)
	ax[1][1].plot(r, m_encl[3])
	for k in range(6):
		# Anisotopy  profile 
		#ax[1][0].plot(r, beta[k], c='C4', lw=0.5, alpha=0.3)
		
		# Anisotopy  profile 
		ax[1][1].plot(r, m_encl[k], c='C0')

		# Anisotopy  profile 
		ax[1][2].plot(r, rho[k], c='C0')
		ax[1][2].set_yscale('log')
		
	plt.savefig(name+"kinematics_and_structure.png", bbox_inches='tight')
	
	return 0

if __name__ == "__main__":
	file_name = sys.argv[1]
	figs_name = sys.argv[2]
	# Kinematics
	angular_momentum = np.loadtxt(file_name + "_kinematics.txt")
	# com
	c_o_m = np.loadtxt(file_name + "_com.txt")
	com = c_o_m[:,0:3]
	vcom = c_o_m[:,3:6]
	com_in = c_o_m[:,6:9]
	vcom_in = c_o_m[:,9:12]
	# beta 
	beta = np.loadtxt(file_name + "_com.txt")
	# Enclosed mass
	density = np.loadtxt(file_name + "_dens_profile.txt")
	encl_mass = np.loadtxt(file_name + "_encl_mass.txt")
	dt = 0.02

	nsnaps = len(angular_momentum)
	t = np.arange(0, dt*nsnaps, dt)
	r = np.linspace(0, 120, 100)
	print(np.shape(beta), np.shape(density), np.shape(encl_mass))
	kinematics_plot(r, t, com, vcom, com_in, vcom_in, beta, angular_momentum,encl_mass, density, figs_name)	


	# Dens profile 
