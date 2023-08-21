"""
Routines for defining pynbody halos and perfoming rotations 

Dependencies:
  - pynbody

author: Nico Garavito-Camargo
github: jngaravitoc

"""


import numpy as np
import pynbody
from pynbody import filt

def pynbody_halo(particles, mask=0, masks=0):
    
    if type(mask)==int :
       ndark = len(particles['dark']['mass'])
       mask = np.ones(ndark, dtype=bool)
    else :
       ndark = len(mask[mask!=0])

    if type(masks)==int :
       nstar = len(particles['star']['mass'])
       masks = np.ones(nstar, dtype=bool)
    else :
       nstar = len(masks[masks!=0])


    print("* Building halo with {} and {} particles".format(int(ndark), int(nstar)))

    halo_pynb = pynbody.new(dark=int(ndark), star=int(nstar), order='dark,star')
    
    halo_pynb.dark['pos'] = particles['dark'].prop('host.distance')[mask]
    halo_pynb.dark['vel'] = particles['dark'].prop('host.velocity')[mask]
    halo_pynb.dark['mass'] = particles['dark'].prop('mass')[mask]

    halo_pynb.star['pos'] = particles['star'].prop('host.distance')[masks]
    halo_pynb.star['vel'] = particles['star'].prop('host.velocity')[masks]
    halo_pynb.star['mass'] = particles['star'].prop('mass')[masks]

    halo_pynb.dark['pos'].units = 'kpc'
    halo_pynb.dark['vel'].units = 'km s**-1'
    halo_pynb.dark['mass'].units = 'Msol'

    halo_pynb.star['pos'].units = 'kpc'
    halo_pynb.star['vel'].units = 'km s**-1'
    halo_pynb.star['mass'].units = 'Msol'
    
    return halo_pynb

def pynbody_subhalos(particles, mask=0):

    if type(mask)==int :
       ndark = len(particles['mass'])
       mask_array = np.ones(ndark, dtype=bool)
    else :
       ndark_init = len(particles['mass'])
       mask_clean = mask[mask>=0]
       print(len(np.unique(mask_clean)), len(mask_clean))
       mask_clean = np.unique(mask_clean)
       ndark = ndark_init - len(mask_clean)
       mask_array = np.ones(ndark_init, dtype=bool)
       mask_array[mask_clean] = 0
       print(ndark, ndark_init, len(mask_array[mask_array==True]))

    # Sub star removing satellite satellites with the mask_array
    sub_stars = particles['star.number'][mask_array]
    stars = np.where(sub_stars!=-1)
    #ndark = len(particles['mass'])
    nstar = len(stars[0])
    print('N satellites = {}'.format(nstar))
    halo_pynb = pynbody.new(dark=int(ndark), star=int(nstar), order='dark,star')
    halo_pynb.dark['pos'] = particles['host.distance'][mask_array]
    halo_pynb.dark['vel'] = particles['host.velocity'][mask_array]
    halo_pynb.dark['mass'] = particles['mass'][mask_array]

    halo_pynb.star['pos'] = particles['host.distance'][mask_array][stars]
    halo_pynb.star['vel'] = particles['host.velocity'][mask_array][stars]
    halo_pynb.star['mass'] = particles['mass'][mask_array][stars]
    halo_pynb.dark['pos'].units = 'kpc'
    halo_pynb.dark['vel'].units = 'km s**-1'
    halo_pynb.dark['mass'].units = 'Msol'

    halo_pynb.star['pos'].units = 'kpc'
    halo_pynb.star['vel'].units = 'km s**-1'
    halo_pynb.star['mass'].units = 'Msol'
    
    return halo_pynb

def pynbody_satellite(particles, **kwargs):
    ndark = len(particles['mass'])
    halo_pynb = pynbody.new(dark=int(ndark))
    halo_pynb.dark['pos'] = particles['position']
    halo_pynb.dark['vel'] = particles['velocity']
    halo_pynb.dark['mass'] = particles['mass']

    if 'treeind' in kwargs:
        halo_pynb.dark['treeind'] = particles['treeind']
    
    return halo_pynb

def make_pynbody_rotations(halo):
    cen = halo[pynbody.filt.Sphere("5 kpc")]
    Lh = pynbody.analysis.angmom.ang_mom_vec(cen)
    Tx_faceon = pynbody.analysis.angmom.calc_faceon_matrix(Lh)
    Tx_sideon = pynbody.analysis.angmom.calc_sideon_matrix(Lh)
    return Tx_faceon, Tx_sideon


        
