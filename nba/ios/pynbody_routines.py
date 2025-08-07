"""
Routines for defining pynbody halos and perfoming rotations 

Dependencies:
  - pynbody

author: Nico Garavito-Camargo
github: jngaravitoc

"""
from nba import requires_library

@requires_library("pynbody")
def createPBhalo(dark_part=None, stellar_part=None):
   """
   Create a pynbody halo with optional dark and/or stellar particles.

   Parameters
   ----------
   dark_part : dict or None
      Dictionary with keys 'pos', 'vel', 'mass' for dark matter particles.

   stellar_part : dict or None
      Dictionary with keys 'pos', 'vel', 'mass' for stellar particles.

   Returns
   -------
   pynbody.snapshot : pynbody snapshot with the specified particle types.
   """
   import pynbody
   particle_counts = {}
   if dark_part is not None:
      ndark = len(dark_part['pos'])
      particle_counts['dark'] = ndark
   else:
      ndark = 0

   if stellar_part is not None:
      nstar = len(stellar_part['pos'])
      particle_counts['star'] = nstar
   else:
      nstar = 0

   if ndark == 0 and nstar == 0:
      raise ValueError("At least one of `dark_part` or `stellar_part` must be provided.")

   order = ','.join(particle_counts.keys())
   print(f"* Building halo with {ndark} dark and {nstar} star particles")

   halo_pynb = pynbody.new(**particle_counts, order=order)

   # Assign data and units
   if dark_part is not None:
      halo_pynb.dark['pos'] = dark_part['pos']
      halo_pynb.dark['vel'] = dark_part['vel']
      halo_pynb.dark['mass'] = dark_part['mass']
      halo_pynb.dark['pos'].units = 'kpc'
      halo_pynb.dark['vel'].units = 'km s**-1'
      halo_pynb.dark['mass'].units = '1e10  Msol'
      if 'pot' in dark_part.keys():
         halo_pynb.dark['phi'] = dark_part['pot']
         halo_pynb.dark['phi'].units = 'km**2 s**-2'

   if stellar_part is not None:
      halo_pynb.star['pos'] = stellar_part['pos']
      halo_pynb.star['vel'] = stellar_part['vel']
      halo_pynb.star['mass'] = stellar_part['mass']
      halo_pynb.star['pos'].units = 'kpc'
      halo_pynb.star['vel'].units = 'km s**-1'
      halo_pynb.star['mass'].units = 'Msol'

   return halo_pynb

@requires_library("pynbody")
def makePBrotation(halo):
   import pynbody
   cen = halo[pynbody.filt.Sphere("5 kpc")]
   Lh = pynbody.analysis.angmom.ang_mom_vec(cen)
   Tx_faceon = pynbody.analysis.angmom.calc_faceon_matrix(Lh)
   Tx_sideon = pynbody.analysis.angmom.calc_sideon_matrix(Lh)
   return Tx_faceon, Tx_sideon


## Old routines that could be useful in the future when working with FIRE data!
"""

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
"""