"""
This code compute quantities related to the anisotropy parameter
beta defined as:

beta = 1 - (sigma_t^2 / 2*sigma_r^2)

where sigma_t is the tangential velocity dispersion and sigma_r the
radial velocity dispersion.

the code compute beta as a function of radius beta(r), beta as a
function of r and z. and the velocity dispersions as a function of
radius.

Requirements:
-------------
Numpy :
Astropy :
Scikit learn :

"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
#from sklearn.neighbors import NearestNeighbors
#from mpl_toolkits.basemap import Basemap
from numpy import linalg as la

class Kinematics:
    """
    Class that compute several kinematics properties of a DM halo.

    """
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
        self.v = (vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)**0.5


    def pos_cartesian_to_galactic(self):
        """
        Transforms carteisian coordinates to galactic.

        Parameters:
        -----------
        pos : 3d-numpy array.

        returns:
        --------
        l : numpy.array([]) in radians

        b : numpy.array([]) in radians

        """
        ## transforming to galactocentric coordinates.

        c_gal = SkyCoord(self.pos, representation_type='cartesian', frame='galactic')

        ## to degrees and wrapping l

        lon_radians = c_gal.spherical.lon.wrap_at(180 * u.deg).radian
        lat_radians = c_gal.spherical.lat.radian


        return lon_radians, lat_radians

    def galactic_coordinates_center(self):
        """
        Tranform system to the solar system barycenter.
        Assumes the input data is centered in the center of the galaxy.
        r_sun = (-8.1, 0, 0)
        v_sun = (11.1, 232.24, 7.25)
        velocities not included for now.

        """

        pos = np.copy(self.pos)
        #vel = np.copy(self.pos)

        pos[:,0] = self.pos[:,0] - 8.1 # kpc
        pos[:,2] = self.pos[:,2] + 0.027 # kpc
        #vel[:,0] = self.vel[:,0] - 11.1 # km/s
        #vel[:,1] = self.vel[:,1] - 232.24 # km/s
        #vel[:,2] = self.vel[:,2] - 7.25 # km/s
        return pos

    def vel_cartesian_to_spherical(self):
        """
        Transfromates velocites from the cartessian to spherical coordinates.

        theta : [0, np.pi] measured from the north pole.
        phi : [0, 2*np.pi] measured from x=0 towards positive y.

        v_r = sin(theta)cos(phi) \hat{v_x} + sin(theta)sin(phi) \hat{v_y} + cos(theta) \hat{v_z}
        v_theta = cos(theta)cos(phi) \hat{v_x} + cos(theta)sin(phi) \hat{v_y} + sin(\theta) \hat{v_z}
        v_phi = -sin \hat{v_x} + cos(phi) \hat{v_y}

        returns:
        --------
        v_r : np.array
        v_theta : np.array
        v_phi : np.array

        """

        r = np.sqrt(self.pos[:,0]**2 + self.pos[:,1]**2 + self.pos[:,2]**2)

        theta = np.arccos(self.pos[:,2]/r)
        phi = np.arctan2(self.pos[:,1], self.pos[:,0])

        v_r = np.sin(theta)*np.cos(phi)*self.vel[:,0] + \
             np.sin(theta)*np.sin(phi)*self.vel[:,1] + \
             np.cos(theta)*self.vel[:,2]

        v_theta = np.cos(theta)*np.cos(phi)*self.vel[:,0] +\
                  np.cos(theta)*np.sin(phi)*self.vel[:,1] - \
                  np.sin(theta)*self.vel[:,2]

        v_phi = -np.sin(phi)*self.vel[:,0] + np.cos(phi)*self.vel[:,1]

        return v_r, v_theta, v_phi



    def orbpole(self):
        # Hacked from Ekta's code
        # r x v in cartesian coordinates!
        uu = np.cross(self.pos, self.vel)
        # |r.v|
        uumag = la.norm(uu, axis=1)
        u = uu.T/uumag
        b = np.arcsin(u[2])
        sinl = u[1]/np.cos(b)
        cosl = u[0]/np.cos(b)
        ll = np.arctan2(sinl,cosl)

        gl = np.degrees(ll)
        gb = np.degrees(b)
        return gl, gb

    def part_angular_momentum(self):
        # r x v in cartessian coordinates!
        L = np.cross(self.pos, self.vel)
        # |r.v|
        Lmag = la.norm(L, axis=1)
        #norm = L.T/Lmag
        return L[:,0], L[:,1], L[:,2], Lmag

    def total_angular_momentum(self):
        """
        Angular momentum of the halo in cartessian coordinates
        """
        L = np.cross(self.pos, self.vel)
        npart = len(L[:,0])
        return np.sum(L[:,0])/npart, np.sum(L[:,1])/npart, np.sum(L[:,2])/npart

    def vel_cartesian_to_galactic(self):
        """
        Transfromates velocites from the cartessian to spherical coordinates.
        Very usefull when computing velocities in the galactic coordinates!

        theta : [-np.pi/2, np.pi/2] measured from z=0
        phi : [0, 2*np.pi] measured from x=0 towards positive y.

        v_r = cos(theta)cos(phi) \hat{v_x} + cos(theta)sin(phi) \hat{v_y} + sin(theta) \hat{v_z}
        v_b = -sin(theta)cos(phi) \hat{v_x}  - cos(theta)sin(phi) \hat{v_y} + sin(\theta) \hat{v_z}
        v_l = -sin \hat{v_x} + cos(phi) \hat{v_y}

        returns:
        --------
        v_r : np.array
        v_b : np.array
        v_l : np.array

        """
        r = np.sqrt(self.pos[:,0]**2 + self.pos[:,1]**2 + self.pos[:,2]**2)

        theta = np.arcsin(self.pos[:,2]/r)
        phi = np.arctan2(self.pos[:,1], self.pos[:,0])

        vr = np.cos(theta)*np.cos(phi)*self.vel[:,0] + \
             np.cos(theta)*np.sin(phi)*self.vel[:,1] + \
             np.sin(theta)*self.vel[:,2]

        v_b = -np.sin(theta)*np.cos(phi)*self.vel[:,0] -\
                  np.sin(theta)*np.sin(phi)*self.vel[:,1] + \
                  np.cos(theta)*self.vel[:,2]

        v_l = -np.sin(phi)*self.vel[:,0] + np.cos(phi)*self.vel[:,1]

        return vr, v_b, v_l


    def velocity_dispersion(self, **kwargs):
        """
        Computes the velocity dispersions in spherical coordinates.

        Parameters:
        ----------
        pos : 3d numpy array
            Array with the cartesian coordinates of the particles
        vel : 3d numpy array
            Array with the cartesian velocities of the particles

        **kwargs:
            LSR : computes the velocity dispersion using the velocities in the galactic frame.

        Returns:
        --------
        sigma_r : float
            The value of sigma_r.
        sigma_theta : float
            The value of sigma_theta
        sigma_phi : float
            The value of sigma_phi

        """


        if 'xyz' and 'vxyz' in kwargs:
            xyz = kwargs['xyz']
            vxyz = kwargs['vxyz']

        else:
            xyz = self.pos
            vxyz = self.vel

        if "LSR" in kwargs:
            vr, v_theta, v_phi = self.vel_cartesian_to_galactic(xyz=xyz, vxyz=vxyz)
        else:
            vr, v_theta, v_phi = self.vel_cartesian_to_spherical(xyz=xyz, vxyz=vxyz)


        sigma_r = np.std(vr)
        sigma_theta = np.std(v_theta)
        sigma_phi = np.std(v_phi)

        return sigma_r, sigma_theta, sigma_phi

    def velocities_means(self, **kwargs):
        """
        Computes the mean velocities in spherical coordinates.

        Parameters:
        -----------
        kwargs : LSR (to compute coordinates in LSR system)

        return:
        -------
        v_r : np.array of radial velocities
        v_theta : np.array of polar velocities.
                  for LSR uses :
                  for spherical :
        v_phi : np.array with azimuthal velocities.
        """
        if 'xyz' and 'vxyz' in kwargs:
            xyz = kwargs['xyz']
            vxyz = kwargs['vxyz']

        else:
            xyz = self.pos
            vxyz = self.vel

        if "LSR" in kwargs:
            v_r, v_theta, v_phi = self.vel_cartesian_to_galactic(xyz=xyz, vxyz=vxyz)
        else:
            v_r, v_theta, v_phi = self.vel_cartesian_to_spherical(xyz=xyz, vxyz=vxyz)

        return np.mean(v_r), np.mean(v_theta), np.mean(v_phi)



    def beta(self, **kwargs):
        """
        Computes the anisotropy $\beta$ defined as:

        beta = 1 - (sigma_t^2 / 2*sigma_r^2)


        Returns:
        --------

        Beta : double
            The value of the anisotropy parameter.
        """

        # This doesn't seem to be the best way of doing this. But it works!
        if 'xyz' and 'vxyz' in kwargs:
            xyz = kwargs['xyz']
            vxyz = kwargs['vxyz']

        else:
            xyz = self.pos
            vxyz = self.vel

        if "LSR" in kwargs:
            sigma_r, sigma_theta, sigma_phi = self.velocity_dispersion(xyz=xyz, vxyz=vxyz, LSR='yes')
        else:
            sigma_r, sigma_theta, sigma_phi = self.velocity_dispersion(xyz=xyz, vxyz=vxyz)

        sigma_t = ((sigma_theta**2 + sigma_phi**2))**0.5
        Beta = 1 - sigma_t**2.0/(2.0*sigma_r**2.0)
        return Beta


    def profiles(self, nbins, quantity, rmin=0, rmax=300, **kwargs):
        """
        Compute the mean velocities in radial bins.

        Parameters:
        ----------
        n_bins : int
            Number of radial bins to compute the velocity dispersions.
        quanity: string
            Compute a kinematic quantity (dispersions, mean, beta)
        rmin : int
            Minimum radius to compute the profile.
        rmax : int
            Maximum radius to compute the profile.

        Returns:
        --------
        vmean_r : numpy array
        vmean_theta : numpy array
        vmean_phi : numpy array

        """

        if 'xyz' and 'vxyz' in kwargs:
            xyz = kwargs['xyz']
            vxyz = kwargs['vxyz']
            r = np.sqrt(xyz[:,0]**2+xyz[:,1]**2+xyz[:,2]**2)
        else:
            xyz = self.pos
            vxyz = self.vel
            r = self.r

        dr = np.linspace(rmin, rmax, nbins)
        dr += (dr[1] - dr[0])/2.

        if ((quantity == 'dispersions') | (quantity == 'mean')):
            vr_q_r = np.zeros(len(dr)-1)
            vtheta_q_r = np.zeros(len(dr)-1)
            vphi_q_r = np.zeros(len(dr)-1)

        elif quantity == 'beta':
            beta_dr = np.zeros(len(dr)-1)

        for i in range(len(dr)-1):
            index = np.where((r<dr[i+1]) & (r>dr[i]))[0]

            if quantity == 'dispersions':
                vr_q_r[i], vtheta_q_r[i], vphi_q_r[i] = self.velocity_dispersion(xyz=xyz[index], vxyz=vxyz[index])
            elif quantity == 'mean':
                vr_q_r[i], vtheta_q_r[i], vphi_q_r[i] = self.velocities_means(xyz=xyz[index], vxyz=vxyz[index])
            elif quantity == 'beta':
                beta_dr[i] = self.beta(xyz=xyz[index], vxyz=vxyz[index])

        if quantity == 'beta':
            return beta_dr
        elif ((quantity == 'dispersions') | (quantity == 'mean')):
            return vr_q_r, vtheta_q_r, vphi_q_r


    def cells_profiles_galactic(self,  bbins, lbins, nbins, quantity, rmin=0, rmax=300, **kwargs):

        """
        Computes the velocity dispersion in region in the sky,
        defined in galactic coordinates.

        Parameters:
        -----------

        Output:
        -------
        **TBD** SPECIFY THE SHAPE OF THE OUTPUT

        """
        print('Computing velocity dispersions in {} regions'.format((lbins)*(bbins)))

        ## Making the octants cuts:

        d_b_rads = np.linspace(-np.pi/2., np.pi/2., bbins+1)
        d_l_rads = np.linspace(-np.pi, np.pi, lbins+1)
        r_bins = np.linspace(rmin, rmax, nbins)

        ## Arrays to store the velocity dispersion profiles
        if ((quantity == 'dispersions') | (quantity == 'mean')):
            vr_octants = np.zeros((nbins-1, (bbins)*(lbins)))
            v_theta_octants = np.zeros((nbins-1, (bbins)*(lbins)))
            v_phi_octants = np.zeros((nbins-1, (bbins)*(lbins)))
        elif quantity == 'beta':
            beta = np.zeros((nbins-1, (bbins)*(lbins)))
        ## Octants counter, k=0 is for the radial bins!
        k = 0

        l, b = self.pos_cartesian_to_galactic()

        for i in range(len(d_b_rads)-1):
            for j in range(len(d_l_rads)-1):
                index = np.where((l<d_l_rads[j+1]) & (l>d_l_rads[j]) &\
                                 (b>d_b_rads[i]) & (b<d_b_rads[i+1]))

                if quantity == 'dispersions':
                    vr_octants[:,k], v_theta_octants[:,k], v_phi_octants[:,k]  = self.profiles(xyz=self.pos[index], \
                                                                                               vxyz=self.vel[index],\
                                                                                               nbins=nbins,\
                                                                                               quantity=quantity,\
                                                                                               rmin=rmin, rmax=rmax)
                elif quantity == 'mean':
                    vr_octants[:,k], v_theta_octants[:,k], v_phi_octants[:,k]  = self.profiles(xyz=self.pos[index],\
                                                                                               vxyz=self.vel[index],\
                                                                                               nbins=nbins, \
                                                                                               quantity=quantity,\
                                                                                               rmin=rmin,\
                                                                                               rmax=rmax)
                elif quantity == 'beta':
                    beta[:,k]  = self.profiles(xyz=self.pos[index], vxyz=self.vel[index],\
                                               nbins=nbins, quantity=quantity, rmin=rmin, \
                                               rmax=rmax)

                k+=1
        if ((quantity == 'dispersions') | (quantity == 'mean')):
            return vr_octants, v_theta_octants, v_phi_octants
        elif quantity == 'beta':
            return beta

    def slice_NN(self, xbins, ybins, n_n, d_slice, quantity,\
                          relative=False, LSR=False, **kwargs):

        """
        Returns a 2d histogram of the anisotropy parameter in galactic coordinates.

        Parameters:
        ----------
        lbins : int
            Numer of bins to do the grid in latitude.
        bbins : int
            Number of bins to do the grid in logitude.
        n_n : int
            Number of neighbors.
        d_slice : float
            galactocentric distance to make the slice cut.
        shell_width:

        quanity: string
            Compute a kinematic quantity (dispersions, mean, beta)

        relative :  If True, the velocity dispersion is computed relative to the
                    mean. (default = False)

        ** kwargs

        Returns:
        --------

        sigma_r_grid : numpy ndarray
            2d array with the radial velocity dispersions.
        sigma_t_grid : numoy ndarray
            2d array with the tangential velocity dispersions.
        """

        ## Defining the
        d_b_rads = np.linspace(-np.pi/2., np.pi/2., bbins+1)
        d_l_rads = np.linspace(-np.pi, np.pi, lbins+1)
        ## Defining the 2d arrays for the velocity dispersions.
        if ((quantity == 'dispersions') | (quantity == 'mean')):
            sigma_r_grid = np.zeros((lbins, bbins))
            sigma_theta_grid = np.zeros((lbins, bbins))
            sigma_phi_grid = np.zeros((lbins, bbins))

        elif quantity == 'beta':
            beta_grid = np.zeros((lbins, bbins))

        xyz = self.pos
        vxyz = self.vel

        r = (xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)**0.5

        # Finding the NN.
        k = 0
        neigh = NearestNeighbors(n_neighbors=n_n, radius=1, algorithm='ball_tree')
        ngbrs = neigh.fit(xyz)

        # Computing mean velocity dispersions inside a spherical shell

        # This goes from l = -180 to l = 180
        for i in range(len(d_l_rads)-1):
            # This goes from b = -90 to b = 90
            for j in range(len(d_b_rads)-1):
                # Goes from l and b to cartessian. Check this
                # Finding the nearest neighbors.
                distances, indices = neigh.kneighbors([pos_grid])

                if quantity == 'dispersions':
                    if LSR == True:
                        sigma_r, sigma_theta, sigma_phi = self.velocity_dispersion(xyz=xyz[indices[0,:]],\
                                                                                   vxyz=vxyz[indices[0,:]],\
                                                                                   LSR='yes')
                    else:
                        sigma_r, sigma_theta, sigma_phi = self.velocity_dispersion(xyz=xyz[indices[0,:]],\
                                                                                   vxyz=vxyz[indices[0,:]])
                elif quantity == 'mean':
                    if LSR == True:
                        sigma_r, sigma_theta, sigma_phi = self.velocities_means(xyz=xyz[indices[0,:]],\
                                                                            vxyz=vxyz[indices[0,:]], LSR='yes')
                    else:
                        sigma_r, sigma_theta, sigma_phi = self.velocities_means(xyz=xyz[indices[0,:]],\
                                                                            vxyz=vxyz[indices[0,:]])
                elif quantity == 'beta':
                    if LSR == True:
                        beta = self.beta(xyz=xyz[indices[0,:]], vxyz=vxyz[indices[0,:]], LSR='yes')
                    else:
                        beta = self.beta(xyz=xyz[indices[0,:]], vxyz=vxyz[indices[0,:]])


                if ((relative==True) & ((quantity=='dispersions') | (quantity=='mean'))):
                    sigma_theta_grid[i][j] = sigma_theta - sigma_theta_mean
                    sigma_phi_grid[i][j] = sigma_phi - sigma_phi_mean
                    sigma_r_grid[i][j] = sigma_r - sigma_r_mean

                elif ((relative==True) & (quantity=='beta')):
                    beta_grid[i][j] = beta - beta_grid_mean

                elif ((quantity=='dispersions') | (quantity=='mean')):
                    sigma_theta_grid[i][j] = sigma_theta
                    sigma_phi_grid[i][j] = sigma_phi
                    sigma_r_grid[i][j] = sigma_r

                elif (quantity=='beta'):
                    beta_grid[i][j] = beta
                k+=1

        if ((quantity == 'dispersions') | (quantity == 'mean')):
            return sigma_r_grid, sigma_theta_grid, sigma_phi_grid

        elif quantity == 'beta':
            return beta_grid

    def shell_NN_galactic(self, lbins, bbins, n_n, d_slice, shell_width, quantity,\
                          relative=False, LSR=False, lmin=-np.pi, lmax=np.pi, **kwargs):

        """
        Returns a 2d histogram of the anisotropy parameter in galactic coordinates.

        Parameters:
        ----------
        lbins : int
            Numer of bins to do the grid in latitude.
        bbins : int
            Number of bins to do the grid in logitude.
        n_n : int
            Number of neighbors.
        d_slice : float
            galactocentric distance to make the slice cut.
        shell_width:

        quanity: string
            Compute a kinematic quantity (dispersions, mean, beta)

        relative :  If True, the velocity dispersion is computed relative to the
                    mean. (default = False)

        ** kwargs

        Returns:
        --------

        sigma_r_grid : numpy ndarray
            2d array with the radial velocity dispersions.
        sigma_t_grid : numoy ndarray
            2d array with the tangential velocity dispersions.
        """

        ## Defining the
        d_b_rads = np.linspace(-np.pi/2., np.pi/2., bbins+1)
        d_l_rads = np.linspace(lmin, lmax, lbins+1)
        ## Defining the 2d arrays for the velocity dispersions.
        if ((quantity == 'dispersions') | (quantity == 'mean')):
            sigma_r_grid = np.zeros((lbins, bbins))
            sigma_theta_grid = np.zeros((lbins, bbins))
            sigma_phi_grid = np.zeros((lbins, bbins))

        elif quantity == 'beta':
            beta_grid = np.zeros((lbins, bbins))

        if LSR == True:
            xyz = self.galactic_coordinates_center()
        else:
            xyz = self.pos

        vxyz = self.vel
        r = (xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)**0.5

        # Finding the NN.
        k = 0
        neigh = NearestNeighbors(n_neighbors=n_n, radius=1, algorithm='ball_tree')
        ngbrs = neigh.fit(xyz)

        # Computing mean velocity dispersions inside a spherical shell
        if relative==True:
            index_cut =  np.where((r<(d_slice+shell_width/2.)) & (r>(d_slice-shell_width/2.)))

            if quantity == 'dispersions':
                sigma_r_mean, sigma_theta_mean, sigma_phi_mean = self.velocity_dispersion(xyz=xyz[index_cut],\
                                                                                          vxyz=vxyz[index_cut])

            elif quantity == 'mean':
                sigma_r_mean, sigma_theta_mean, sigma_phi_mean = self.velocities_means(xyz=xyz[index_cut],\
                                                                                       vxyz=vxyz[index_cut])

            elif quantity == 'beta':
                beta_grid_mean = self.beta(xyz=xyz[index_cut], vxyz=vxyz[index_cut])

        # This goes from l = -180 to l = 180
        for i in range(len(d_l_rads)-1):
            # This goes from b = -90 to b = 90
            for j in range(len(d_b_rads)-1):
                # Goes from l and b to cartessian. Check this
                gc = SkyCoord(l=d_l_rads[i]*u.radian, b=d_b_rads[j]*u.radian,\
                              frame='galactic', distance=d_slice*u.kpc)
                # produce grid of values in Cartesian coordinates.
                pos_grid = gc.cartesian.xyz.value
                # Finding the nearest neighbors.
                distances, indices = neigh.kneighbors([pos_grid])

                if quantity == 'dispersions':
                    if LSR == True:
                        sigma_r, sigma_theta, sigma_phi = self.velocity_dispersion(xyz=xyz[indices[0,:]],\
                                                                                   vxyz=vxyz[indices[0,:]],\
                                                                                   LSR='yes')
                    else:
                        sigma_r, sigma_theta, sigma_phi = self.velocity_dispersion(xyz=xyz[indices[0,:]],\
                                                                                   vxyz=vxyz[indices[0,:]])
                elif quantity == 'mean':
                    if LSR == True:
                        sigma_r, sigma_theta, sigma_phi = self.velocities_means(xyz=xyz[indices[0,:]],\
                                                                            vxyz=vxyz[indices[0,:]], LSR='yes')
                    else:
                        sigma_r, sigma_theta, sigma_phi = self.velocities_means(xyz=xyz[indices[0,:]],\
                                                                            vxyz=vxyz[indices[0,:]])
                elif quantity == 'beta':
                    if LSR == True:
                        beta = self.beta(xyz=xyz[indices[0,:]], vxyz=vxyz[indices[0,:]], LSR='yes')
                    else:
                        beta = self.beta(xyz=xyz[indices[0,:]], vxyz=vxyz[indices[0,:]])


                if ((relative==True) & ((quantity=='dispersions') | (quantity=='mean'))):
                    sigma_theta_grid[i][j] = sigma_theta - sigma_theta_mean
                    sigma_phi_grid[i][j] = sigma_phi - sigma_phi_mean
                    sigma_r_grid[i][j] = sigma_r - sigma_r_mean

                elif ((relative==True) & (quantity=='beta')):
                    beta_grid[i][j] = beta - beta_grid_mean

                elif ((quantity=='dispersions') | (quantity=='mean')):
                    sigma_theta_grid[i][j] = sigma_theta
                    sigma_phi_grid[i][j] = sigma_phi
                    sigma_r_grid[i][j] = sigma_r

                elif (quantity=='beta'):
                    beta_grid[i][j] = beta
                k+=1

        if ((quantity == 'dispersions') | (quantity == 'mean')):
            return sigma_r_grid, sigma_theta_grid, sigma_phi_grid

        elif quantity == 'beta':
            return beta_grid
