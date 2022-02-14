# -*- coding: utf-8 -*-


import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

"""
TODO:
----

"""

class Structure:
	def __init__(self, pos, mass):
		self.pos = pos
		self.mass = mass

	def density_profile(self, nbins, rmin, rmax):
		"""
		Computes the number density radial profile. Assuming all the partiles have the same mass.
		
		"""
		r_p = np.sqrt(np.sum(self.pos**2, axis=1))
		dens_profile = np.zeros(nbins-1)
		dr = np.linspace(rmin, rmax, nbins)
		#delta_r = dr[2]-dr[1]

		for j in range(0, len(dr)-1):
			index = np.where((r_p < dr[j+1]) & (r_p > dr[j]))[0]
			V = 4/3 * np.pi * (dr[j+1]**3-dr[j]**3)
			dens_profile[j] = np.sum(self.mass[index])/V
		return  dens_profile

	def potential_profile(self, pot, nbins, rmin, rmax):
		"""
		Computes the halo potential profile.
		
		"""
		r_p = np.sqrt(np.sum(self.pos**2, axis=1))
		pot_profile = np.zeros(nbins-1)
		dr = np.linspace(rmin, rmax, nbins)
		#delta_r = dr[2]-dr[1]

		for j in range(0, len(dr)-1):
			index = np.where((r_p < dr[j+1]) & (r_p > dr[j]))[0]
			pot_profile[j] = np.sum(pot[index])
		return  pot_profile

	def enclosed_mass(self, mass, nbins, rmin, rmax):
		"""
		Computes the halo potential profile.
		
		"""
		r_p = np.sqrt(np.sum(self.pos**2, axis=1))
		mass_profile = np.zeros(nbins-1)
		dr = np.linspace(rmin, rmax, nbins)
		#delta_r = dr[2]-dr[1]

		for j in range(0, len(dr)-1):
			index = np.where((r_p < dr[j+1]))[0]
			mass_profile[j] = np.sum(mass[index])
		return  mass_profile

if __name__ == '__main__':
	print("Hello")
