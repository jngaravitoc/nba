# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.basemap import Basemap
from astropy.coordinates import SkyCoord
from astropy import units as u
from sklearn.neighbors import NearestNeighbors
import sys

sys.path.insert(0, '../kinematics/')
sys.path.insert(0, '../')

#import kinematics
#from reading_snapshots import *
#import weights as WW
#from tracers_wake import *

"""
TODO:
----

1. Function to identify the location of the over-dense regions and
   return it's coordinates.
2. Implement under densities finder.2. Implement under densities
finder.
"""

def od_coordinates(dens, sigma_t, xmin=0, xmax=1, ymin=0, ymax=1):
    assert xmin < xmax, "xmax should be greater than xmin"
    assert ymin < ymax, "ymax should be greater than ymin"
    assert sigma_max >0, "sigma_t should be larger than 0"
    assert type(sigma_max) == int, "sigma_t should be a integer"

    # Defining grid
    x = np.linspace(xmin, xmax, np.shape(dens)[0])
    y = np.linspace(ymin, ymax, np.shape(dens)[1])
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    # Defining sigma as the standard deviation of the data
    sigma = np.std(dens.flatten())
    print('sigma === ', sigma)
    # Finding the median of the all the data in the field
    dens_median = np.median(dens.flatten())
    # Defining the contours range.·
    overdensities = []
    color_bar_labels = []

    index = np.where(dens>(dens_median+sigma*sigma_t))

    x_coord = index[0]*dx
    y_coord = index[1]*dy

    x_mean = np.sum(x_coord)/len(x_coord)
    y_mean = np.sum(y_coord)/len(y_coord)

    return x_mean, y_mean

def density_grid(pos, nbinsx, nbinsy, xmin, xmax):
    """
    Returns a 2d histogram with the number densities of the halo.
    In Cartessian coordinates, xmin and xmax determine the range in
    x-coordinates in which the overdensities are going to be computed.

    returns:
    --------
    rho_grid : 2d.numpy.array.
            number of particles per cell bin.


    """

    y_bins = np.linspace(-200, 200, nbinsx)
    z_bins = np.linspace(-200, 200, nbinsy)

    rho_grid = np.zeros((nbinsx-1, nbinsy-1))

    for i in range(len(y_bins)-1):
        for j in range(len(z_bins)-1):
                # selecting particles inside the cell
                index = np.where((pos[:,1]<y_bins[i+1]) &\
                                  (pos[:,1]>y_bins[i]) &\
                                  (pos[:,2]>z_bins[j]) &\
                                  (pos[:,2]<z_bins[j+1]) &\
                                  (pos[:,0]<xmax) &\
                                  (pos[:,0]>xmin))[0]
                rho_grid[i][j] = len(index)
    return rho_grid


def density_nn(pos, nbinsx, nbinsy, z_plane, n_n, xmin=-200, xmax=200,\
               ymin=-200, ymax=200, **kwargs):
    """
    Returns a 2d histogram with the density of the halo.
    In Cartessian coordinates, xmin and xmax determine the range in
    x-coordinates in which the overdensities are going to be computed.
    """

    y_bins = np.linspace(xmin, xmax, nbinsx)
    z_bins = np.linspace(ymin, ymax, nbinsy)

    density_grid = np.zeros((nbinsx-1, nbinsy-1))


    r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    max_r = np.max([(xmax-xmin), (ymax-ymin)])
    r_cut = np.where(r<max_r)[0]
    pos_cut = pos[r_cut]
    # Finding the NN.

    if 'mass' in kwargs.keys():
        mass = kwargs['mass']
        m_cut = mass[r_cut]

    k = 0
    neigh = NearestNeighbors(n_neighbors=n_n, radius=1,
                             algorithm='ball_tree')
    ngbrs = neigh.fit(pos_cut)

    for i in range(len(y_bins)-1):
        for j in range(len(z_bins)-1):
            pos_grid = [z_plane, y_bins[i], z_bins[j]]
            distances, indices = neigh.kneighbors([pos_grid])
            if 'mass' in kwargs.keys():
                density_grid[i][j] = 3*np.sum(m_cut[indices])/(4*np.pi*(np.max(distances)**3.0))
            else:
                density_grid[i][j] = 3*n_n/(4*np.pi*(np.max(distances)**3.0))

    return density_grid

def density_aitoff(l, b, lbins, bbins, rmin, rmax, pos):
    """
    Returns a 2d histogram of the over densities on the MW halo.
    In Cartessian coordinates, xmin and xmax determine the range in
    x-coordinates in which the overdensities are going to be computed.
    """

    d_b_rads = np.linspace(-np.pi/2., np.pi/2., bbins)
    d_l_rads = np.linspace(-np.pi, np.pi, lbins)
    r_lmc = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    density_grid = np.zeros((lbins-1, bbins-1))

    for i in range(len(d_l_rads)-1):
        for j in range(len(d_b_rads)-1):
                #print(y_bins[i+1], y_bins[i], z_bins[j+1], z_bins[j])
                index = np.where((l<d_l_rads[i+1]) &\
                                  (l>d_l_rads[i]) &\
                                  (b>d_b_rads[j]) &\
                                  (b<d_b_rads[j+1]) &\
                                  (r_lmc<rmax) &\
                                  (r_lmc>rmin))[0]

                density_grid[i][j] = len(index)
    return density_grid


def density_aitoff_nn(pos, lbins, bbins, r_shell, n_n, mass, shell_width=5,\
        err_p=0, n_part=False, galactic=False, lmin=-np.pi, lmax=np.pi):
    """
    Returns a 2d histogram of the over densities on the MW halo
    in sphercal Galactocentric coordinates coordinates.

    Input:
    ------
    pos : cartessian positions np.array(3, n)
    lbins : number of longitude bins int
    bbins : number of laitude bins int
    r_shell  : Galactocentric distance where the density is compute numpy.float
    n_n : number of neighboors used to compute the density
    mass : particles masses  np.array(n)
    shell_width : width in kpc of the spherical shell where the density is computed
    	default 5 kpc

    Returns:
	density_grid : 2d grid with the densities
	distances_mean : mean distances used to computed the densities
    """

    print('computing density using error in distances of {} kpc'.format(err_p))

    d_b_rads = np.linspace(-np.pi/2., np.pi/2., bbins)
    d_l_rads = np.linspace(lmin, lmax, lbins)

    density_grid = np.zeros((lbins-1, bbins-1))



    if galactic == 'True':
        pos_x = pos[:,0] - 8.1
        pos_z = pos[:,2] + 0.027
    else:
        pos_x = pos[:,0]
        pos_z = pos[:,2]

    pos_y = pos[:,1]

    pos_x = np.random.normal(pos_x, err_p/(3**0.5))
    pos_y = np.random.normal(pos_y, err_p/(3**0.5))
    pos_z = np.random.normal(pos_z, err_p/(3**0.5))

    pos = np.array([pos_x, pos_y, pos_z]).T

    r = (pos_x**2 + pos_y**2 + pos_z**2)**0.5

    r_cut = np.where((r<r_shell+shell_width/2.0) & (r>r_shell-shell_width/2.0))[0]
    pos_cut = pos[r_cut]
    mass_cut = mass[r_cut]
    # Finding the NN.
    k = 0
    neigh = NearestNeighbors(n_neighbors=n_n, radius=1, algorithm='ball_tree')
    ngbrs = neigh.fit(pos_cut)
    distances_mean = []
    for i in range(len(d_l_rads)-1):
        for j in range(len(d_b_rads)-1):
                c = SkyCoord(l=d_l_rads[i]*u.radian, b=d_b_rads[j]*u.radian, distance=r_shell*u.kpc, frame='galactic')
                pos_grid = c.cartesian.xyz # should I need to substract to galacocentric now?
                distances, indices = neigh.kneighbors([pos_grid])
                distances_mean.append(np.mean(distances))
                density_grid[i][j] =  3*np.sum(mass_cut[indices])/(4*np.pi*(np.max(distances)**3.0))

    return density_grid, distances_mean



def potential_aitoff_nn(pos, pot, lbins, bbins, r_shell, n_n):
    """
    Returns a 2d histogram of the potential feel it by the MW halo.
    In Cartessian coordinates, xmin and xmax determine the range in
    x-coordinates in which the over-densities are going to be computed.
    """

    d_b_rads = np.linspace(-np.pi/2., np.pi/2., bbins)
    d_l_rads = np.linspace(-np.pi, np.pi, lbins)
    r_lmc = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    pot_grid = np.zeros((lbins-1, bbins-1))


    r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    r_cut = np.where((r<r_shell+5) & (r>r_shell-5))[0]
    pos_cut = pos[r_cut]
    pot_cut = pot[r_cut]
    print(len(pot_cut))

    # Finding the NN.
    k = 0
    neigh = NearestNeighbors(n_neighbors=n_n, radius=1, algorithm='ball_tree')
    ngbrs = neigh.fit(pos_cut)

    for i in range(len(d_l_rads)-1):
        for j in range(len(d_b_rads)-1):
            c = SkyCoord(l=d_l_rads[i]*u.radian, b=d_b_rads[j]*u.radian, distance=r_shell*u.kpc, frame='galactic')
            pos_grid = c.cartesian.xyz
            distances, indices = neigh.kneighbors([pos_grid])
            #print(len(indices[0]))
            pot_grid[i][j] =  np.mean(pot_cut[indices[0]])

    return pot_grid

def octants():

    plt.annotate('$1$', xy=(0.15, 0.25), xycoords='axes fraction', fontsize=25, color='k')
    plt.annotate('$2$', xy=(0.30, 0.25), xycoords='axes fraction', fontsize=25, color='k')
    plt.annotate('$3$', xy=(0.65, 0.25), xycoords='axes fraction', fontsize=25, color='k')
    plt.annotate('$4$', xy=(0.85, 0.25), xycoords='axes fraction', fontsize=25, color='k')
    plt.annotate('$5$', xy=(0.15, 0.70), xycoords='axes fraction', fontsize=25, color='k')
    plt.annotate('$6$', xy=(0.30, 0.70), xycoords='axes fraction', fontsize=25, color='k')
    plt.annotate('$7$', xy=(0.65, 0.70), xycoords='axes fraction', fontsize=25, color='k')
    plt.annotate('$8$', xy=(0.85, 0.70), xycoords='axes fraction', fontsize=25, color='k')

    return 0

def degrees():

    plt.annotate('$-180^{\circ}$', xy=(0, 0.5), xycoords='axes fraction', fontsize=25, color='k')
    plt.annotate('$-90^{\circ}$', xy=(0.25, 0.5), xycoords='axes fraction', fontsize=25, color='k')
    plt.annotate('$0^{\circ}$', xy=(0.52, 0.5), xycoords='axes fraction', fontsize=25, color='k')
    plt.annotate('$90^{\circ}$', xy=(0.75, 0.5), xycoords='axes fraction', fontsize=25, color='k')
    plt.annotate('$180^{\circ}$', xy=(0.91, 0.5), xycoords='axes fraction', fontsize=25, color='k')

    plt.annotate('$-90^{\circ}$', xy=(0.7, 0.0), xycoords='axes fraction', fontsize=20)
    plt.annotate('$-45^{\circ}$', xy=(0.92, 0.2), xycoords='axes fraction', fontsize=20)
    plt.annotate('$0^{\circ}$', xy=(1.01, 0.47), xycoords='axes fraction', fontsize=20)
    plt.annotate('$45^{\circ}$', xy=(0.92, 0.77), xycoords='axes fraction', fontsize=20)
    plt.annotate('$90^{\circ}$', xy=(0.7, 1), xycoords='axes fraction', fontsize=20)

    return 0

def lmc_orbit(orbit):
    """
    Integrate orbit.

    """
    lmc_orbit = np.loadtxt(orbit)

    x = lmc_orbit[:,6]-lmc_orbit[:,0]
    y = lmc_orbit[:,7]-lmc_orbit[:,1]
    z = lmc_orbit[:,8]-lmc_orbit[:,2]
    vx = lmc_orbit[:,9]-lmc_orbit[:,3]
    vy = lmc_orbit[:,10]-lmc_orbit[:,4]
    vz = lmc_orbit[:,11]-lmc_orbit[:,5]

    r_lmc = np.sqrt(x**2 + y**2 + z**2)



    pos = np.array([x, y, z]).T
    vel = np.array([vx, vy, vz]).T

    orbit_lmc = kinematics.Kinematics(pos, vel)
    l_lmc, b_lmc = orbit_lmc.pos_cartesian_to_galactic()
    return r_lmc, l_lmc*180/np.pi, b_lmc*180/np.pi




def kernel(r, d):
	#  From
	#  https://www.cs.cornell.edu/courses/cs5643/2014sp/stuff/BridsonFluidsCourseNotes_SPH_pp83-86.pdf
	w = np.zeros(len(r))
	index_in = np.where(np.sqrt(r)<=d)[0]
	#print(len(index_in))
	w[index_in] = 315/(64*np.pi*d**9) * (d**2 -	r[index_in])**3
	return w


def density(r, part_m, part_pos, d):
    r_particles_2 = (part_pos[:,0]-r[0])**2 + (part_pos[:,1]-r[1])**2 + (part_pos[:,2]-r[2])**2
    #print('Computing density in {} dimensions'.format(len(r)))

    rho = np.sum(part_m*kernel(r_particles_2, d))
    return rho

def density_grid_kernel(part_m, part_pos, d, grid_res, **kwargs):
    """
    kwargs:
        box :
    """

    if 'box' in kwargs:
        print('here')
        xmin, xmax, ymin, ymax, zmin, zmax = kwargs['box']
        print('Computing density inside a box of xmin = {}, xmax {}'.format(xmin, xmax))

    else:
        xmin = min(part_pos[:,0])
        xmax = max(part_pos[:,0])
        ymin = min(part_pos[:,1])
        ymax = max(part_pos[:,1])
        zmin = min(part_pos[:,2])
        zmax = max(part_pos[:,2])

    x = np.arange(xmin, xmax, grid_res)
    y = np.arange(ymin, ymax, grid_res)
    z = np.arange(zmin, zmax, grid_res)
    rho_grid = np.zeros((len(x), len(y), len(z)))

    index_cut = np.where((part_pos[:,0]<xmax) & (part_pos[:,0]>xmin) &\
                         (part_pos[:,1]<ymax) & (part_pos[:,1]<ymax) &\
                         (part_pos[:,2]<zmax) & (part_pos[:,2]<zmax))

    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                rho_grid[i][j][k] = density([x[i], y[j], z[k]], part_m[index_cut], \
                                            part_pos[index_cut], d)
    return rho_grid


def density_grid_2d_kernel(part_m, part_pos, d, grid_res):
    xmin = min(part_pos[:,0])
    xmax = max(part_pos[:,0])
    ymin = min(part_pos[:,1])
    ymax = max(part_pos[:,1])

    x = np.arange(xmin, xmax, grid_res)
    y = np.arange(ymin, ymax, grid_res)

    rho_grid = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            rho_grid[i][j] = density([x[i], y[j]], part_m, part_pos, d)
    return rho_grid



def lmc_orbit_cartesian(orbit):
    """
    Integrate orbit.

    """
    lmc_orbit = np.loadtxt(orbit)

    x = lmc_orbit[:,6]-lmc_orbit[:,0]
    y = lmc_orbit[:,7]-lmc_orbit[:,1]
    z = lmc_orbit[:,8]-lmc_orbit[:,2]


    return x[:114], y[:114], z[:114]


def sgr_particles(snap, r_shell, shell_width):
     sgr = io.readsav('snap')
     sag = sgr['ppsgr']
     r = (sag[:,0]**2 + sag[:,1]**2 + sag[:,2]**2)**0.5

     c_gal = SkyCoord(sag, representation='cartesian',frame='galactic')
     c_gal.representation = 'spherical'

     #l_degrees = c_gal.l.wrap_at(180 * u.deg).radian
     #b_degrees = c_gal.b.radian
     r_cut = np.where((r<=r_shell + shell_width) & (r>= r_shell - shell_width))
     l_degrees_cut = l_degrees[r_cut]
     b_degrees_cut = b_degrees[r_cut]
     #r_sgr = r[r_cut]
     return l_degrees_cut, b_degrees_cut

def GC(snap, r_shell, shell_width):
     GCs = np.genfromtxt('snap', delimiter=',')
     x_gc = GCs[:,5]
     x_gc = GCs[:,6]
     x_gc = GCs[:,7]

     sag = sgr['ppsgr']

     r_GC = (x_gc**2 + y_gc**2 + z_gc**2)**0.5

     c_gal = SkyCoord(sag, representation='cartesian',frame='galactic')
     c_gal.representation = 'spherical'

     l_degrees = c_gal.l.wrap_at(180 * u.deg).radian
     b_degrees = c_gal.b.radian
     r_cut = np.where((r_GC<=r_shell + shell_width) & (r_GC>= r_shell - shell_width))
     l_degrees_cut = l_degrees[r_cut]
     b_degrees_cut = b_degrees[r_cut]
     #r_sgr = r[r_cut]
     return l_degrees_cut, b_degrees_cut

def density_peaks(dens,  xmin=0, xmax=1, ymin=0, ymax=1, fsize=(10, 4.2), **kwargs):
    """
    Function to plot contours on densities:

    Parameters:
    -----------
    dens : numpy array
          A 2D array with the densities.
    xmin : int
          Minimum value in the x-direction (default = 0).
    xmax : int
          Maximum value in the x-direction (default = 1).
    ymin : int
          Minimum value in the y-direction (default = 0).
    ymax : int
          Maximum value in the y-direction (default = 1).

   **kwargs:

    levels : list
        A list with the levels for the color bar
    cbar_label : string
        A string with the colorbar label.
    octants : key
        Octants enumeration will appear in the plot.
    Distance :
        Distance at which the density field is computed, this is to put a text with the value.
    lmc_orbit : str
        Overplot the orbit of the LMC it requires the distance argument! you
        have to pass the path to the orbit of the LMC.

    sgr_particles :
        Overplot the sgr particles at that distance.

    returns :
    --------


    Figure with contours.


    """

    # Defining grid
    # sigma_R comes as latitude and then longitude
    print(np.shape(dens))
    print(np.shape(dens)[0])
    x = np.linspace(xmin, xmax, np.shape(dens)[0])
    y = np.linspace(ymin, ymax, np.shape(dens)[1])
    X, Y = np.meshgrid(x, y)


    fig = plt.figure(figsize=fsize)

    if 'projection' in kwargs.keys():
        m = Basemap(projection='moll', lon_0=0)

    if 'levels' in kwargs.keys():
        levels = kwargs['levels']

        if 'projection' in kwargs.keys():
            print('Projection and levels')
            cont = m.contourf(X, Y, dens.T, 12, vmin=-20, vmax=40, latlon=True, cmap=cm.Spectral_r)
            m.drawmeridians(np.arange(-180, 180, 90), linewidth=1.5, \
                            labels=[True, True, True])
            m.drawparallels(np.arange(-90, 90, 45), linewidth=1.5)
            degrees()
        else :

            vmn = kwargs['vmin']
            vmx = kwargs['vmax']
            cont = plt.contourf(X, Y, dens.T, len(levels), vmin=vmn, vmax=vmx,  cmap=cm.Spectral_r)


        cbar = plt.colorbar(cont)

    else:
        if 'projection' in kwargs.keys():
            #cl = np.arange(-20, 45, 5)
            cl = np.arange(-0.6, 2.05, 0.05)
            cont = m.contourf(X, Y, dens.T,  latlon=True, cmap=cm.Spectral_r)
            m.drawmeridians(np.arange(-180, 180, 90), linewidth=1.5, \
                            labels=[True, True, True])

            m.drawparallels(np.arange(-90, 90, 45), linewidth=1.5)
            degrees()
        else:
            cont = plt.contourf(X, Y, dens.T, cmap=cm.Spectral_r)
        cbar = plt.colorbar(cont)

    if 'cbar_label' in kwargs.keys():
        cbar_label = kwargs['cbar_label']
        cbar.set_label(cbar_label)

    #cbar.ax.tick_params(labelsize=22)

    if 'octants' in kwargs.keys():
        octants()

    if 'distance' in kwargs.keys():
        r = kwargs['distance']
        plt.annotate('${}$'.format(r)+'$\, \mathrm{kpc}$', xy=(0,1), xycoords='axes fraction', fontsize=25)

        if 'lmc_orbit' in kwargs.keys():
            orbit_name = kwargs['lmc_orbit']
            shell_width = kwargs['shell_width']
            r_lmc, l_lmc, b_lmc = lmc_orbit(orbit_name)
            r_cut_lmc = np.where((r_lmc <= r+shell_width/2.0) & (r_lmc >= r-shell_width/2.0))
            b_proj, l_proj = m(l_lmc[r_cut_lmc], b_lmc[r_cut_lmc])
            m.scatter(b_proj, l_proj, c='k', marker='*', s=80)

        if 'sgr particles' in kwargs.keys():
            orbit_name = kwargs['lmc_orbit']
            l_sgr, b_sgr = sgr_particles(orbit_name)
            b_sgr_proj, l_sgr_proj = m(b_sgr[r_cut_sgr], l_sgr[r_cut_sgr])
            m.scatter(b_sgr_proj, l_sgr_proj)

    if 'lmc_orbit_cartesian' in kwargs.keys():
        orbit_name = kwargs['lmc_orbit_cartesian']
        xlmc, ylmc, zlmc = lmc_orbit_cartesian(orbit_name)
        plt.plot(ylmc, zlmc, c='k', lw=0.5)

    return 0





if __name__ == '__main__':
    #ath = sys.argv[1]
    #plot_type = sys.argv[1]
    #plot_type = 'mollweide_pot'
    plot_type = 'wake'
    #nap_name1 = 'MW_'
    #snap_MWLMC = sys.argv[2]
    #snap_MW = sys.argv[3]
    #out_image = sys.argv[4]
    #path = './test_snaps/'
    #mc = int(sys.argv[5])
    #lmc_orbit_name = sys.argv[6]
    #snap_MWLMC = 'MWLMC6_100M_new_b0_2_113'
    #snap_MW = 'MW2_100M_beta0_vir_020'
    radial_cuts = [45]

    N_halo_part = 100000000

    if plot_type == 'mollweide_pot':
        snap_name1 = 'MWLMC6_100M_new_b0_2_113'
        #snap_name1 = 'MW2_100M_beta0_vir_020'
        path1 = '/media/ngaravito/4fb4fd3d-1665-4892-a18d-bdbb1185a07b1/xzk/work/sims/MWLMC6_100M_beta0/'
        #path1 = '/media/ngaravito/4fb4fd3d-1665-4892-a18d-bdbb1185a07b1/xzk/work/sims/MW2_100M_beta0_vir/'
        #path2 =
        LMC = 1
        MW_pos, MW_vel, MW_pot = read_MW_snap_com_coordinates(path1, snap_name1,\
                                                              LMC=True,\
                                                              N_halo_part=N_halo_part,\
                                                              pot=True)

        #MW_pos_lmc, MW_vel_lmc, MW_pot_lmc, MW_mas_lmc, MW_ids_lmc = WW.read_MW_snap_com_coordinates(path2, snap_MWLMC, \
        #                                                                                             LMC=LMC2,\
        #                                                  N_halo_part=N_halo_part)
        #f lmc == 1:
        #   r_lmc, l_lmc, b_lmc = lmc_orbit(lmc_orbit_name)

        #rint('Snapshots loaded')
        pot_M = potential_aitoff_nn(MW_pos, MW_pot, 200, 100, 45, 100)

        density_peaks(np.log10(np.abs(pot_M)), xmin=-180, xmax=180, ymin=-90, ymax=90,
                      projection='yes')

        plt.savefig('pot_mollweide.pdf', bbox_inches='tight', dpi=300)
        plt.close()
    """
        print('radius = {:0>3d} Kpc'.format(i))
        sigma_r_mw, sigma_t_mw  = sigma2d_NN(MW_pos, MW_vel, 200, 100 , 1000, i)

        #rho_mw  = wake_aitoff(MW_pos, 600, 300 , 1000, i)
        #rho_mwlmc  = wake_aitoff(MW_pos_lmc, 600, 300 , 1000, i)

        sigma_r_grid_lmc, sigma_t_grid_lmc = sigma2d_NN(MW_pos_lmc, MW_vel_lmc,\
                                                        200, 100, 1000, i)
        print('Velocity dispersions computed')

        #print('writing files')
        #np.savetxt('sigma_r_mw_l200_b100_nn1000_r{}.txt'.format(i), sigma_r_mw)


        density_peaks(sigma_t_grid_lmc-sigma_t_mw, xmin=-180, xmax=180, \
                      ymin=-90, ymax=90, distance=i,\
                      projection='yes', levels=np.arange(-20, 45, 5),\
                      cbar_label=r'$\Delta \sigma$')
        if lmc == 1:
            r_cut = np.where((r_lmc < i+2.5) & (r_lmc > i-2.5))[0]
            plt.scatter(l_lmc[r_cut], b_lmc[r_cut], c='k', marker='*', s=180)
        #density_peaks(rho_mwlmc/rho_mw - 1, -4, 6, i, i+5, xmin=-180, xmax=180, ymin=-90, ymax=90)

        plt.xlabel('$l[^{\circ}]$', fontsize=20)
        plt.ylabel('$b[^{\circ}]$', fontsize=20)
        plt.title(r'$\Delta \sigma_t$')
        plt.savefig(out_image+'t_{:0>3d}.png'.format(i), bbox_inches='tight',\
                    dpi=300)
        plt.close()


        density_peaks(sigma_r_grid_lmc-sigma_r_mw, xmin=-180, xmax=180,
                      ymin=-90, ymax=90, distance=i,\
                      projection='yes', levels=np.arange(-20, 45, 5),\
                      cbar_label=r'$\Delta \sigma$')
        #if lmc == 1:
        #    r_cut = np.where((r_lmc < i+2.5) & (r_lmc > i-2.5))[0]
        #    plt.scatter(l_lmc[r_cut], b_lmc[r_cut], c='k', marker='*', s=180)

        #density_peaks(sigma_r_grid_lmc-np.mean(sigma_r_grid), -4, 6, i, i+5,  xmin=-180, xmax=180, ymin=-90, ymax=90)
        plt.xlabel('$l[^{\circ}]$', fontsize=20)
        plt.ylabel('$b[^{\circ}]$', fontsize=20)
        plt.title(r'$\Delta \sigma_r$'.format(i))
        plt.savefig(out_image+'r_{:0>3d}.png'.format(i), bbox_inches='tight',\
                    dpi=300)
        plt.close()
    """
    if plot_type == 'wake':
        snap_name1 = 'MWLMC6_100M_new_b0_2_113'
        snap_name2 = 'MW2_100M_beta0_vir_020'
        path1 = '../test_snaps/'
        path2 = '../test_snaps/'
        #path2 =
        LMC = 1
        MWLMC_pos, MW_vel, MW_pot = read_MW_snap_com_coordinates(path1, snap_name1,\
                                                              LMC=True,\
                                                              N_halo_part=N_halo_part,\
                                                              )

        MW_pos, MW_vel, MW_pot = read_MW_snap_com_coordinates(path2, snap_name2,\
                                                              LMC=False,\
                                                              N_halo_part=N_halo_part,\
                                                              )
        #MW_pos_lmc, MW_vel_lmc, MW_pot_lmc, MW_mas_lmc, MW_ids_lmc = WW.read_MW_snap_com_coordinates(path2, snap_MWLMC, \
        #                                                                                             LMC=LMC2,\
        #                                                  N_halo_part=N_halo_part)
        #f lmc == 1:
        #   r_lmc, l_lmc, b_lmc = lmc_orbit(lmc_orbit_name)

        #rint('Snapshots loaded')
        #pot_M = potential_aitoff_nn(MW_pos, MW_pot, 200, 100, 45, 100)

        rho_mwlmc = density_nn(MWLMC_pos, 200, 200, 10, 1000)
        rho_mw = density_nn(MW_pos, 200, 200, 10, 1000)
        density_peaks(rho_mwlmc/rho_mw, xmin=-200, xmax=200, ymin=-200,
                ymax=200)

        plt.savefig('wake_test.pdf', bbox_inches='tight', dpi=300)
        plt.close()
