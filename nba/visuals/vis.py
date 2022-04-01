import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import healpy as hp
import sys


class Visuals:
    """
    Class for plotting several halo quantities.

    """
    def __init__(self):
        x=3 		

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
            print("here")
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



    def mollweide_projection(self, l, b, nside, smooth=5):
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
        return twd_map

    def particle_slice(self, pos, nbins, norm, grid_size, cmap='magma'):
        """
        Compute a density slice 

        """
        hbxy = np.histogram2d(pos[:,0], pos[:,1], bins=nbins, normed=norm)
        hbxz = np.histogram2d(pos[:,0], pos[:,2], bins=nbins, normed=norm)
        hbyz = np.histogram2d(pos[:,1], pos[:,2], bins=nbins, normed=norm)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        im1= ax[0].imshow(hbxy[0].T, norm=LogNorm(), origin='lower', extent=grid_size, cmap=cmap, aspect='auto')
        im2 = ax[1].imshow(hbxz[0].T, norm=LogNorm(), origin='lower', extent=grid_size, cmap=cmap, aspect='auto')
        im3 = ax[2].imshow(hbyz[0].T, norm=LogNorm(), origin='lower', extent=grid_size,  cmap=cmap, aspect='auto')
        fig.colorbar(im1, ax=ax[0], orientation='horizontal')
        fig.colorbar(im2, ax=ax[1], orientation='horizontal')
        fig.colorbar(im3, ax=ax[2], orientation='horizontal')
    
        ax[0].set_xlabel('x[kpc]')
        ax[0].set_ylabel('y[kpc]')
        ax[1].set_xlabel('x[kpc]')
        ax[1].set_ylabel('z[kpc]')
        ax[2].set_xlabel('y[kpc]')
        ax[2].set_ylabel('z[kpc]')
    
        return fig

    def animate(self, snapshot, snap_format, init, final, fig_name=0, npart=10000):
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
