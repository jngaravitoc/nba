"""
Plotting routines used for the orbital poles project

Dependencies:
  - Matplitlib
  - Healpy
  - pynbody 

author: Nico Garavito-Camargo
github: jngaravitoc



"""


import numpy as np
import matplotlib.pyplot as plt
import pynbody
import sys
from pynbody import filt
from astropy import units as u
import nba
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

sys.path.append("/mnt/home/ecunningham/python")
plt.style.use('~/matplotlib.mplstyle')
plt.rcParams['font.size'] = 35

def multipanel_plot(hf, hs, satellite_faceon, snap, sim, figname):
    """
    Parameters:
    -----------
        hf : pynbody halo format for the dark matter
        hs : pynbody halo format for the stellar halo  
        satellite_faceon : pynbody halo format 
            satellites position to plot orbit
        snap : int
            snap number
        sim : str
            simulation name
        figname : str
            figure name 
    Return:
    -------
        figure
    """
    times = '/mnt/ceph/users/firesims/fire2/metaldiff/{}_res7100/snapshot_times.txt'.format(sim)
    t_snap = np.loadtxt(times, usecols=3)

    fig, ax = plt.subplots(2, 3, figsize=(20, 13))
    dmproj = pynbody.plot.image(hf.d, width=1200, cmap='Greys', subplot=ax[0][0], show_cbar=False, av_z=True)
    dmproj2 = pynbody.plot.image(hs.d, width=1200, cmap='Greys', subplot=ax[1][0], show_cbar=False, av_z=True)

    dmproj = pynbody.plot.image(hf.s, width=300, cmap='inferno', subplot=ax[0][1], show_cbar=False, av_z=True)
    dmproj2 = pynbody.plot.image(hs.s, width=300, cmap='inferno', subplot=ax[1][1], show_cbar=False, av_z=True)

    dmproj = pynbody.plot.image(hf.s, width=100, cmap='inferno', subplot=ax[0][2], show_cbar=False, av_z=True)
    dmproj2 = pynbody.plot.image(hs.s, width=100, cmap='inferno', subplot=ax[1][2], show_cbar=False, av_z=True)


    ax[0][0].plot(satellite_faceon.dark['pos'][:snap-300,0], satellite_faceon.dark['pos'][:snap-300,1], c='k', ls='--', alpha=0.6, lw=1)
    ax[1][0].plot(satellite_faceon.dark['pos'][:snap-300,0], satellite_faceon.dark['pos'][:snap-300,2], c='k', ls='--', alpha=0.6, lw=1)

    ax[0][1].plot(satellite_faceon.dark['pos'][:snap-300,0], satellite_faceon.dark['pos'][:snap-300,1], c='w', ls='--', alpha=0.6, lw=1)
    ax[1][1].plot(satellite_faceon.dark['pos'][:snap-300,0], satellite_faceon.dark['pos'][:snap-300,2], c='w', ls='--', alpha=0.6, lw=1)

    ax[0][2].plot(satellite_faceon.dark['pos'][:snap-300,0], satellite_faceon.dark['pos'][:snap-300,1], c='w', ls='--', alpha=0.6, lw=1)
    ax[1][2].plot(satellite_faceon.dark['pos'][:snap-300,0], satellite_faceon.dark['pos'][:snap-300,2], c='w', ls='--', alpha=0.6, lw=1)



    ax[0][1].set_ylabel('')
    ax[0][2].set_ylabel('')
    ax[1][1].set_ylabel('')
    ax[1][2].set_ylabel('')

    ax[0][0].set_xlabel('$x\mathrm{[kpc]}$')
    ax[0][1].set_xlabel('$x\mathrm{[kpc]}$')
    ax[0][2].set_xlabel('$x\mathrm{[kpc]}$')

    ax[1][0].set_xlabel('$x\mathrm{[kpc]}$')
    ax[1][1].set_xlabel('$x\mathrm{[kpc]}$')
    ax[1][2].set_xlabel('$x\mathrm{[kpc]}$')


    ax[0][0].set_ylabel('$y\mathrm{[kpc]}$')
    ax[1][0].set_ylabel('$z\mathrm{[kpc]}$')

    ax[0][0].set_title('$\mathrm{Dark\ Matter}$', fontsize=16)
    ax[0][1].set_title('$\mathrm{Stars\ outer\ halo}$', fontsize=16)
    ax[0][2].set_title('$\mathrm{Stars\ inner\ halo}$', fontsize=16)
    fig.suptitle('$t={:.2f}$'.format(t_snap[snap]) + r' $\rm{Gyr;}\ \rm{Sim:\ }$ ' + '{}'.format(sim), fontsize=18, y=0.95)

    ax[0][0].text(-500, 400, r"$\rm{Face-on}$", c='k', fontsize=18)
    ax[1][0].text(-500, 400, r"$\rm{Edge-on}$", c='k', fontsize=18)

    ax[0][0].set_xlim(-600, 600)
    ax[1][0].set_xlim(-600, 600)

    ax[0][1].set_xlim(-150, 150)
    ax[1][1].set_xlim(-150, 150)

    ax[0][2].set_xlim(-50, 50)
    ax[1][2].set_xlim(-50, 50)

    ax[0][0].set_ylim(-600, 600)
    ax[1][0].set_ylim(-600, 600)

    ax[0][1].set_ylim(-150, 150)
    ax[1][1].set_ylim(-150, 150)

    ax[0][2].set_ylim(-50, 50)
    ax[1][2].set_ylim(-50, 50)
    plt.savefig(figname + "_{:03d}.png".format(snap), bbox_inches='tight')
    plt.close()
    
    
def mollweide_projection(l, b, l2, b2, title, bmin, bmax, nside, smooth, q=[0], **kwargs):

    """
    Makes mollweide plot using healpix
    Parameters:
    ----------- 
    l : numpy.array in degrees 
    b : numpy.array in degrees [-90, 90]
    """
 
    times = '/mnt/ceph/users/firesims/fire2/metaldiff/{}_res7100/snapshot_times.txt'.format('m12b')

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

        

def plot_2dcorrfunc(w, w0, t0, t1, title, figname, hlines=[],  vmin=-0.1, vmax=0.1):
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    if type(w0) == np.ndarray :
        im = plt.imshow((w+1)/(w0+1) - 1, origin='lower', extent=[0, 180, t0, t1],
                    vmin=vmin, vmax=vmax, aspect='auto', cmap='Spectral')
        cbar = plt.colorbar(im)
        cbar.set_label(r'$\tilde{\omega} (\theta)$')
    else : 
        im = plt.imshow(w, origin='lower', extent=[0, 180, t0, t1],
                    vmin=vmin, vmax=vmax, aspect='auto', cmap='Spectral')
        cbar = plt.colorbar(im)
        cbar.set_label(r'$\omega (\theta)$')
        
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$t\ \rm{[Gyr}]$')
    ax.set_title(title)
    ax.set_xticks([0, 60, 120, 180])
    for n in range(len(hlines)):
        ax.axhline(hlines[n], ls='--', c='k', lw=1)

    plt.savefig(figname, bbox_inches='tight', dpi=300)

