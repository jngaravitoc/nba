import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

class GlobalProperties:
     def __init__(self):


     def kinematics_plot(self, r, t, com, vcom, com_in, vcom_in, beta, L, m_encl, rho, name):
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



class Particles:
    """
    Class for plotting several halo quantities.

    """
    def __init__(self, halo):
        self.pos = halo['pos']
        self.vel = halo['vel']
        self.mass = halo['mass']

    def quick_scatter_plot(self, pos, npart_sample, box_size=[-100, 100, -100, 100], figure_name=None):
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

   
    def compute_mollweide(self, l, b, nside, smooth=5):
        """
        Makes mollweide plot using healpix
        Parameters:
        -----------
        l : numpy.array

        b : numpy.array
        """

        mwlmc_indices = hp.ang2pix(nside,  (90-b)*np.pi/180., l*np.pi/180.)
        npix = hp.nside2npix(nside)

        idx, counts = np.unique(mwlmc_indices, return_counts=True)
        degsq = hp.nside2pixarea(nside, degrees=True)
        # fill the fullsky map
        hpx_map = np.zeros(npix, dtype=int)
        hpx_map[idx] = counts/degsq
        map_smooth = hp.smoothing(hpx_map, fwhm=smooth*np.pi/180)
        twd_map = hp.mollview(map_smooth, rot=180, return_projected_map=True)
        return twd_map, hpx_map


    def plot_mollweide_galactic(self, twd_map, rotation, bmin, bmax, figname, fig_title):
        projview(twd_map, coord=["G"], graticule=True, graticule_labels=True,
                rot=rotation, unit=" ",  xlabel="Galactic Longitude (l) ", 
                ylabel="Galactic Latitude (b)", cb_orientation="horizontal", min=bmin, max=bmax, 
                latitude_grid_spacing=45, projection_type="mollweide", title=fig_title)
 
        #newprojplot(theta=np.radians(90-(b2[0])), phi=np.radians(l2[0]-120), marker="*", color="r", markersize=5 )
        plt.savefig(figname, bbox_inches='tight')
        plt.close()
        return 0
    

    def mollweide_projection(l, b, l2, b2, title, bmin, bmax, nside, smooth, q=[0], **kwargs):

        """
        Makes mollweide plot using healpix
        Parameters:
        ----------- 
        l : numpy.array in degrees 
        b : numpy.array in degrees [-90, 90]
        """
    

        mwlmc_indices = hp.ang2pix(nside,  (90-b)*np.pi/180., l*np.pi/180.)
        npix = hp.nside2npix(nside)
    
        idx, counts = np.unique(mwlmc_indices, return_counts=True)
        degsq = hp.nside2pixarea(nside, degrees=True)
        # filling the full-sky map
        hpx_map = np.zeros(npix, dtype=float)
        if q[0] != 0 :    
            counts = np.zeros_like(idx, dtype=float)
            k=0
            for i in idx:
                pix_ids = np.where(mwlmc_indices==i)[0]
                counts[k] = np.mean(q[pix_ids])
                k+=1
            hpx_map[idx] = counts
        else :
        hpx_map[idx] = counts/degsq

        map_smooth = hp.smoothing(hpx_map, fwhm=smooth*np.pi/180)
    
        if ((bmin == 'auto') & (bmax == 'auto')):
            bmin = np.min(map_smooth)
            bmax = np.max(map_smooth)

        if 'cmap' in kwargs.keys():
            cmap = kwargs['cmap']
        else:
            cmap='viridis'
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        plt.close()
        projview(
        map_smooth,
        coord=["G"],
        graticule=True,
        graticule_labels=True,
        rot=(0, 0, 0),
        unit=" ",
        #xlabel="Galactic Longitude (l) ",
        ylabel="Galactic Latitude (b)",
        cb_orientation="horizontal",
        min=bmin,
        max=bmax,
        latitude_grid_spacing=45,
        projection_type="mollweide",
        title=title,
        cmap=cmap,
        fontsize={
                "xlabel": 25,
                "ylabel": 25,
                "xtick_label": 20,
                "ytick_label": 20,
                "title": 25,
                "cbar_label": 20,
                "cbar_tick_label": 20,
                },
        )
        
        newprojplot(theta=np.radians(90-(b2)), phi=np.radians(l2), marker="o", color="yellow", markersize=5, lw=0, mfc='none')
        if 'l3' in kwargs.keys():
            l3 = kwargs['l3']
            b3 = kwargs['b3']
            newprojplot(theta=np.radians(90-(b3)), phi=np.radians(l3), marker="o", color="yellow", markersize=5, lw=0)
        elif 'l4' in kwargs.keys():
            l4 = kwargs['l4']
            b4 = kwargs['b4']
            newprojplot(theta=np.radians(90-(b4)), phi=np.radians(l4), marker="*", color="r", markersize=8, lw=0)

        #newprojplot(theta=np.radians(90-(b2[0])), phi=np.radians(l2[0]-120), marker="*", color="r", markersize=5 )
        #newprojplot(theta=np.radians(90-(b2[1])), phi=np.radians(l2[1]-120), marker="*", color="w", markersize=2 )
        
        if 'figname' in kwargs.keys():
            print("* Saving figure in ", kwargs['figname'])
            plt.savefig(kwargs['figname'], bbox_inches='tight')
            plt.close()
        #return 0

            


    def particle_slice(self, pos, nbins, norm, grid_size, cmap='magma'):
        """
        Compute a density slice 

        """
        hbxy = np.histogram2d(pos[:,0], pos[:,1], bins=nbins, normed=norm)
        hbxz = np.histogram2d(pos[:,0], pos[:,2], bins=nbins, normed=norm)
        hbyz = np.histogram2d(pos[:,1], pos[:,2], bins=nbins, normed=norm)

        extent1 = [np.min(hbxy[1]), np.max(hbxy[1]), np.min(hbxy[2]), np.max(hbxy[2])]
        extent2 = [np.min(hbxz[1]), np.max(hbxz[1]), np.min(hbxz[2]), np.max(hbxz[2])]
        extent3 = [np.min(hbyz[1]), np.max(hbyz[1]), np.min(hbyz[2]), np.max(hbyz[2])]

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        im1= ax[0].imshow(hbxy[0].T, norm=LogNorm(), origin='lower', extent=extent1, cmap=cmap, aspect='equal')
        im2 = ax[1].imshow(hbxz[0].T, norm=LogNorm(), origin='lower', extent=extent2, cmap=cmap, aspect='equal')
        im3 = ax[2].imshow(hbyz[0].T, norm=LogNorm(), origin='lower', extent=extent3,  cmap=cmap, aspect='equal')
        
        fig.colorbar(im1, ax=ax[0], orientation='horizontal')
        fig.colorbar(im2, ax=ax[1], orientation='horizontal')
        fig.colorbar(im3, ax=ax[2], orientation='horizontal')
    
        ax[0].set_xlabel('x[kpc]')
        ax[0].set_ylabel('y[kpc]')
        ax[1].set_xlabel('x[kpc]')
        ax[1].set_ylabel('z[kpc]')
        ax[2].set_xlabel('y[kpc]')
        ax[2].set_ylabel('z[kpc]')
        ax[0].set_xlim(grid_size[0], grid_size[1]) 
        ax[0].set_ylim(grid_size[2], grid_size[3]) 
        ax[1].set_xlim(grid_size[0], grid_size[1]) 
        ax[1].set_ylim(grid_size[4], grid_size[5]) 
        ax[2].set_xlim(grid_size[2], grid_size[3]) 
        ax[2].set_ylim(grid_size[4], grid_size[5]) 
        return fig

    def animate(self, snapshot, snap_format, init, final, fig_name=0, npart=10000):
        for k in range(final-init+1):
            pos, vel, mass = load_snapshot(snapshot+"_{:03d}.hdf5".format(k), snap_format)
            scatter(pos, npart, figure_name=fig_name+"{:03d}.png".format(k))
        return 0

