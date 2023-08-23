import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import healpy as hp
from healpy.newvisufunc import projview, newprojplot



class Visuals:
    """
    Class for plotting several halo quantities.

    """
    def __init__(self, pos):
        self.pos = pos 		

    #def quiver_plot(self, **kwargs):
    #	return 0

    def all_sky_galactic(self, lbins, bbins , quantity, **kwargs):
        m = Basemap(projection = 'moll',lon_0=0, lat_0=0)
        fig = figure(figsize=(6, 4.5))
        x = np.linspace(-180, 180, lbins)
        y = np.linspace(-90, 90, bins)

        X, Y = meshgrid(x, y)
        llmc, blmc = m(llmc, blmc)

        if 'contours' in kwargs:
            levels = [1, 1.2]

            im = m.contour(X, Y, quantity.T/np.mean(quantity) , levels=levels,
                extent=(-180, 180, -90, 90), latlon=True, colors='k', alpha=0.7)
            im = m.contourf(X, Y, (sigma.T-np.mean(sigma)), 100, origin='lower',
                            cmap='Spectral_r', latlon=True)

        m.scatter(llmc, blmc, marker='*', s=80, c='k', facecolors='none', alpha=0.6)
        m.drawmeridians(np.arange(-180, 180, 90), linewidth=1.5, labels=[True, True, True])
        m.drawparallels(np.arange(-90, 90, 45), linewidth=1.5)
        xlabel('$l[^{\circ}]$')
        ylabel('$b[^{\circ}]$')
        #title(r'$\Delta {} ({}kpc)$'.format(component, dist))
        title(r'$ {} ({}kpc)$'.format(component, dist))
        cbar_ax2 = fig.add_axes([0.05, 0.1, 0.9, 0.03])
        cbar2 = colorbar(im, cax=cbar_ax2, orientation='horizontal')
        #cbar_ticks2 = np.arange(-15, 16, 5)
        cbar2.set_ticks(cbar_ticks2)
        #cbar2.set_label('$\mathrm{[km/s]}$')
        cbar2.set_label('$v_r$')
        savefig(figname + '.pdf', bbox_inches='tight')
        savefig(figname + '.png', bbox_inches='tight')
        return 0

    def scatter(self, pos, npart_sample, box_size=[-100, 100, -100, 100], figure_name = 0):
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
        else:
            plt.show()
        # Loop over snapshots


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


    def particle2Dhist(self, nbins, projection_axis=[[0,1]], norm=LogNorm(), cmap='magma', **kwargs):
        """
        Computes 2d histograms from particle's coordinates 

        pos_init :  
        nbins : 

        **kwargs:

        text : str
            
        labels : list of strings [n, 2] : [[xlabel1, ylabel1], [xlabel2, ylabel2]...]
             axis labels for each desired projection
        gridsize : [n, 4] : [[min(x1), max(x1), min(y1), max(y1)], ...]
            grid size for ploting hitogram with imshow

        figname: str
            name of the figure 

        # TODO:
        -Write doc string
        -include text coorindated in kwargs

        """

        kwargs_keys = kwargs.keys()
        
        nprojections = len(projection_axis)
        h_all = []
        extent_all = []

        for n in range(nprojections):
            p1, p2 = projection_axis[n]
            h = np.histogram2d(self.pos[:,p1], self.pos[:,p2], bins=nbins)
            h_all.append(h)
            extent = [np.min(h[1]), np.max(h[1]), np.min(h[2]), np.max(h[2])]
            extent_all.append(extent)
       
        fig, ax = plt.subplots(1, nprojections, figsize=(4*nprojections, 4))
        if nprojections==1:
            axs = [ax]
        else:
            axs = ax.flatten()
        
        for n in range(nprojections):
            p1, p2 = projection_axis[n]
            if 'pos_ref' in kwargs.keys():
                pos0 = kwargs['pos_ref']
                h0 = np.histogram2d(pos0[:,p1], pos0[:,p2], bins=nbins)
                if 'vmin' & 'vmax' in kwargs.keys():
                    vmin = kwargs['vmin']
                    vmax = kwargs['vmax']
                    im = axs[n].imshow((h_all[n][0]/h0).T-1, vmin=vmin, vmax=vmax, origin='lower', extent=extent_all[n], cmap=cmap, aspect='equal')
                else: 
                    im = axs[n].imshow((h_all[n][0]/h0).T-1, origin='lower', extent=extent_all[n], cmap=cmap, aspect='equal')
            else:
               
                im = axs[n].imshow(h_all[n][0].T, norm=norm, origin='lower', extent=extent_all[n], cmap=cmap, aspect='equal')
            fig.colorbar(im, ax=axs[n], orientation='horizontal') 
            if 'text' in  kwargs.keys() :
                # TODO add text posiitons to kwargs! 
                axs[0].text(-200, 200, kwargs['text'])
        
            if 'labels' in kwargs.keys():
                labels = kwargs['labels']
                axs[n].set_xlabel(labels[n][0] + r'$\rm{[kpc]}$')
                axs[n].set_ylabel(labels[n][1] + r'$\rm{[kpc]}$')

            if 'gridsize' in kwargs.keys():
                gridsize=kwargs['gridsize'] 
                axs[n].set_xlim(gridsize[n][0], grid_size[n][1]) 
                axs[n].set_ylim(gridsize[n][2], grid_size[n][3]) 
            
        if 'figname' in kwargs.keys():
            plt.savefig(kwargs['figname'], bbox_inches='tight')

        return fig

    def animate(self, snapshot, snap_format, init, final, fig_name=0, npart=10000):
        for k in range(final-init+1):
            pos, vel, mass = load_snapshot(snapshot+"_{:03d}.hdf5".format(k), snap_format)
            scatter(pos, npart, figure_name=fig_name+"{:03d}.png".format(k))
        return 0

