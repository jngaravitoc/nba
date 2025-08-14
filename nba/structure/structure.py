import numpy as np
from scipy.interpolate import BSpline, splrep


class Profiles:
    def __init__(self, pos, edges):
        self.pos = pos
        self.edges = edges

        assert np.shape(self.edges)[0] > 1, "Edges array needs to have more than one element!"
        assert np.shape(self.pos)[0] > 0, "Positions array is empty!"
        assert np.shape(self.pos)[1] == 3, "Positions array dimension it's not 3. positions arrays needs to be of shape (npart, 3) !"

    
    def density(self, smooth=0, mass=1.0):
        """
        Computes the number/mass density radial profile for equal-mass particles.

        Parameters
        ----------
        smooth : bool, optional
            If True, applies a smoothing spline to the density profile.
            Default is False.
            Common values are between 0.0 (not smoothed) - 2.0.
        mass : float or array-like, optional
            Particle mass (scalar if all equal).

        Returns
        -------
        r_centers : ndarray
            Bin center radii.
        dens_profile : ndarray
            Density in each radial bin.
        """
        # Radial distances
        r_p = np.sqrt(np.sum(self.pos**2, axis=1))

        # Histogram for counts (or mass if mass array provided)
        if np.isscalar(mass):
            mass_per_particle = mass 
            hist, _ = np.histogram(r_p, bins=self.edges)
            mass_hist = hist * mass_per_particle
        else:
            mass_hist, _ = np.histogram(r_p, bins=self.edges, weights=mass)

        # Volumes for each spherical shell
        r_outer = self.edges[1:]
        r_inner = self.edges[:-1]
        
        volumes = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)

        # Density = mass / volume
        dens_profile = mass_hist / volumes

        # Bin centers
        r_centers = 0.5 * (r_outer + r_inner)

        if smooth > 0:
            tck_s = splrep(r_centers, np.log10(dens_profile), s=smooth)
            dens_profile = 10**(BSpline(*tck_s)(r_centers))

        return r_centers, dens_profile

    def potential(self, pot, nbins, rmin, rmax):
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

    def enclosed_mass(self, nbins, rmin, rmax, mass):
        """
        Computes the halo potential profile.
        """
        r_p = np.sqrt(np.sum(self.pos**2, axis=1))
        mass_profile = np.zeros(nbins-1)
        dr = np.linspace(rmin, rmax, nbins)
        delta_r = (dr[2]-dr[1])*0.5
        for j in range(0, len(dr)-1):
            index = np.where((r_p < dr[j+1]))[0]
            mass_profile[j] = np.sum(mass[index])
        return  dr[:-1] + delta_r, mass_profile


    ''''
    ## TODO: this shouls be part of another function to make sky cuts
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
    '''

