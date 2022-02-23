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
        delta_r = (dr[2]-dr[1])*0.5

        for j in range(0, len(dr)-1):
            index = np.where((r_p < dr[j+1]) & (r_p > dr[j]))[0]
            V = 4/3 * np.pi * (dr[j+1]**3-dr[j]**3)
            dens_profile[j] = np.sum(self.mass[index])/V
        return  dr[:-1] + delta_r, dens_profile

    def potential_profile(self, pot, nbins, rmin, rmax):
        """
        Computes the halo potential profile.
    	"""
        r_p = np.sqrt(np.sum(self.pos**2, axis=1))
        pot_profile = np.zeros(nbins-1)
        dr = np.linspace(rmin, rmax, nbins)
        delta_r = (dr[2]-dr[1])*0.5

        for j in range(0, len(dr)-1):
            index = np.where((r_p < dr[j+1]) & (r_p > dr[j]))[0]
            pot_profile[j] = np.sum(pot[index])
        return  dr[:-1] + delta_r, pot_profile

    def enclosed_mass(self, nbins, rmin, rmax):
        """
        Computes the halo potential profile.
        """
        r_p = np.sqrt(np.sum(self.pos**2, axis=1))
        mass_profile = np.zeros(nbins-1)
        dr = np.linspace(rmin, rmax, nbins)
        delta_r = (dr[2]-dr[1])*0.5
        for j in range(0, len(dr)-1):
            index = np.where((r_p < dr[j+1]))[0]
            mass_profile[j] = np.sum(self.mass[index])
        return  dr[:-1] + delta_r, mass_profile

    def density_octants(pos, vel, nbins, rmax):

        """
        Computes the velocity dispersion in eight octants in the sky,
        defined in galactic coordinates as follows:

        octant 1 :
        octant 2 :
        octant 3 :
        octant 4 :
        octant 5 :
        octant 6 :
        octant 7 :
        octant 8 :

        Parameters:
        -----------

        Output:
        -------



        """

        ## Making the octants cuts:

        # Galactic latitude
        d_b_rads = np.linspace(-np.pi/2., np.pi/2.,3)

        # Galactic Longitude
        d_l_rads = np.linspace(-np.pi, np.pi, 5)

        #r_bins = np.linspace(0, 300, 31)

        ## Arrays to store the velocity dispersion profiles
        rho_octants = np.zeros((nbins, 8))

        ## Octants counter, k=0 is for the radial bins!
        k = 0

        l, b = pos_cartesian_to_galactic(pos, vel)

        for i in range(len(d_b_rads)-1):
            for j in range(len(d_l_rads)-1):
                index = np.where((l<d_l_rads[j+1]) & (l>d_l_rads[j]) &\
                                 (b>d_b_rads[i]) & (b<d_b_rads[i+1]))

                rho_octants[:,k] =  dens_r(pos[index], nbins, rmax)
                k+=1

        return rho_octants

if __name__ == '__main__':
    print("Hello")
