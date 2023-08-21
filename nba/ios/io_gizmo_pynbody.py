"""
Master script to analyze the FIRE simulations for the orbital poles project
(github/jngaravitoc/poles_in_cosmos)


This script has been tested with sims: m12b, m12i

Main functionalities:
   - Make plots
    - Density plots of the DM and stellar distribution in several projections 
    - Mollweide plots of particles and subhalos positions in Galactocentric
      coordinates.
    - Mollweide plots of the orbital poles
   - Perform analysis
    - Correlation function analysis

Dependencies:
  - scipy
  - numpy 
  - Gizmo Analysis
  - Halo tools
  - pynbody
  - Astropy
  - nba 

Author: Nico Garavito-Camargo
Github: jngaravitoc

TODO:
- Remove satellite subhalos

"""

#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import sys
import pynbody

sys.path.append("/mnt/home/ecunningham/python")
#plt.style.use('~/matplotlib.mplstyle')
import gizmo_analysis as ga
import halo_analysis as halo

# personal n-body analysis library 
import nba

# local libraries
import pynbody_routines  as pr 
import plotting as pl


## Tracking subhalos using their index at 300th snap (using only merger tree)
def return_tracked_pos(halo_tree, tr_ind_at_init, pynbody_halo=False, init_snap=300, final_snap=600):
    # Adapted from Arpit's function
    h_index = tr_ind_at_init
    tree_ind = []
    tree_ind.append(h_index) 
    for _ in range(init_snap, final_snap):
        try :
            h_index = halo_tree['descendant.index'][h_index]
        except IndexError :
            break 
        tree_ind.append(h_index)

    tree_ind = np.array(tree_ind)
    positive = np.where(tree_ind>0)
    tree_ind = tree_ind[positive]
    position = halo_tree['host.distance'][tree_ind]
    nsnaps = halo_tree['snapshot'][tree_ind]
    mass = halo_tree['mass'][tree_ind]
    velocity = halo_tree['host.velocity'][tree_ind]
    cat_ind = halo_tree['catalog.index'][tree_ind]
    
    if pynbody_halo == True:
        sat = {'position': position,
               'mass': mass,
               'velocity': velocity,
               'treeind' : tree_ind, 
            } 
        return pr.pynbody_satellite(sat, treeind='treind')

    elif pynbody_halo == False:

        return {'position': position,
                 'mass' : mass,
                'velocity' : velocity,
                'treeind' : tree_ind,
                'catind' : cat_ind,
                'snapshot' : nsnaps,
               }


## Tracking subhalos using their index at 300th snap (using only merger tree)
def return_tracked_pos_back(halo_tree, tr_ind_at_init, pynbody_halo=False, nsnaps=300):
    # Adapted from Arpit's function
    h_index = tr_ind_at_init
    tree_ind = []
    
    for _ in range(nsnaps):
        tree_ind.append(h_index)
        h_index = halo_tree['progenitor.main.index'][h_index]
    tree_ind = np.array(tree_ind)
    position = halo_tree['host.distance'][tree_ind]
    nsnaps = halo_tree['snapshot'][tree_ind]
    mass = halo_tree['mass'][tree_ind]
    velocity = halo_tree['host.velocity'][tree_ind]
    #vel_rad = halt['host.velocity.rad'][tree_ind]
    #vel_tan = halt['host.velocity.tan'][tree_ind]
    if pynbody_halo == True:
        sat = {'position': position,
               'mass': mass,
               'velocity': velocity,
            }
        return pr.pynbody_satellite(sat)

    elif pynbody_halo == False:

        return {'position': position,
                'snaps' : nsnaps,
                'mass' : mass,
                'velocity' : velocity,
               }
    
    
def poles_subhalos(snap, rmin=20, rmax=400, satellites=False):
    f = 1* (u.km/u.s).to(u.kpc/u.Gyr)
    m12_halo = halo.io.IO.read_catalogs('index', snap, sim_directory)
    dist = np.sqrt(np.sum(m12_halo['host.distance']**2, axis=1))
    rcut = np.where((dist>rmin) & (dist<rmax))
                    
    m12_300 = nba.kinematics.Kinematics(m12_halo['host.distance'][rcut], m12_halo['host.velocity'][rcut]*f)
    l, b = m12_300.orbpole()

    lpol = Angle(l * u.deg)
    lpolw = lpol.wrap_at(360 * u.deg).degree  
    
    if satellites == True :
        stellar_subhalos = m12_halo['star.mass'][rcut]!=-1

        return lpolw[stellar_subhalos], b[stellar_subhalos]
    else :
        return lpolw, b


# Get all the subhalos
def get_halo_satellite(sim, mass_rank):
    sim_directory = "/mnt/ceph/users/firesims/fire2/metaldiff/{}_res7100/".format(sim)
    m12_subhalos = halo.io.IO.read_catalogs('index', 300, sim_directory)
    halt = halo.io.IO.read_tree(simulation_directory=sim_directory)
    hsub = pr.pynbody_subhalos(m12_subhalos)
    sat_id = np.argsort(hsub.dark['mass'])[mass_rank]
    sat_tree_id = m12_subhalos['tree.index'][sat_id]
    satellite = fa.return_tracked_pos(halt, sat_tree_id, pynbody_halo=True)
    f = 1* (u.km/u.s).to(u.kpc/u.Gyr)
    m12_sat = nba.kinematics.Kinematics(satellite['pos'], satellite['vel']*f)
    l, b = m12_sat.orbpole()
    lpol = Angle(l * u.deg)
    lpolw = lpol.wrap_at(360 * u.deg).degree  
    return lpolw, b



class FIRE:
  """
  Class to load FIRE snapshots and subhalos using pynbody format.

  """

  def __init__(self, sim, remove_satellite=False, remove_subs=False, only_sat=False, rm_stellar_sat=False):

    self.sim_directory = "/mnt/ceph/users/firesims/fire2/metaldiff/{}_res7100/".format(sim)
    
    if sim == 'm12c':
        sat_path = '/mnt/home/ecunningham/ceph/latte/{}_res7100/massive_stream/dm_inds.npy'.format(sim)
        self.sat_ids = np.load(sat_path) 
    
    elif sim == 'm12f':
        sat_path = '/mnt/home/ecunningham/ceph/latte/m12f_res7100/massive_stream/dm_particle_inds.npy'
        stars_path = '/mnt/home/ecunningham/ceph/latte/{}_res7100/massive_stream/z0_stream_inds.npy'.format(sim)
        self.stars_ids = np.load(stars_path)
        self.sat_ids = np.load(sat_path) 

    elif sim == 'm12b':
        sat_path = '/mnt/home/ecunningham/ceph/latte/{}_res7100/massive_stream/dm_inds.npy'.format(sim)
        subs_path = '/mnt/home/nico/ceph/FIRE/{}_385_unbound_dark_indices.npy'.format(sim)
        self.subs_ids = np.load(subs_path)
        stars_path = '/mnt/home/ecunningham/ceph/latte/{}_res7100/massive_stream/new_z0_inds.npy'.format(sim)
        self.stars_ids = np.load(stars_path)
        self.sat_ids = np.load(sat_path) 

    elif sim == 'm12r':
        sat_path = '/mnt/home/ecunningham/ceph/latte/{}_res7100/dm_inds_m1.npy'.format(sim)
        sat_path2 = '/mnt/home/ecunningham/ceph/latte/{}_res7100/dm_inds_m2.npy'.format(sim)
        sat_path3 = '/mnt/home/ecunningham/ceph/latte/{}_res7100/dm_inds_m3.npy'.format(sim)
        
        sat_ids1 = np.load(sat_path)
        sat_ids2 = np.load(sat_path2)
        sat_ids3 = np.load(sat_path3)
        sat_ids = np.hstack((sat_ids1, sat_ids2)) 
        self.sat_ids = np.hstack((sat_ids, sat_ids3))

    elif sim == 'm12i':
        sat_path = '/mnt/home/ecunningham/ceph/latte/m12i_res7100/arpit_merger/bound_dm_inds_356.npy'.format(sim)
        stars_path = '/mnt/home/ecunningham/ceph/latte/m12i_res7100/arpit_merger/z0_stream_inds.npy'.format(sim)
        self.stars_ids = np.load(stars_path)
        self.sat_ids = np.load(sat_path) 
   
    elif sim == 'm12m':
        sat_path = "/mnt/home/ecunningham/ceph/latte/m12m_res7100/massive_stream/dm_inds.npy"
        self.sat_ids = np.load(sat_path)

    elif sim == 'm12w':
        sat_path = "/mnt/home/ecunningham/ceph/latte/m12w_res7100/massive_stream/dm_inds.npy"
        self.sat_ids = np.load(sat_path)

    #self.sat_ids = np.load(sat_path)
    self.sim = sim
    self.rm_sat = remove_satellite
    self.rm_subs = remove_subs
    self.only_sat = only_sat
    self.rm_stellar_sat = rm_stellar_sat
    

    if (self.rm_sat == True) | (self.only_sat == True):
        assert self.rm_sat != self.only_sat, '! You have to either select satellite subs or remove it'
    

  def rotated_halo(self, snap, part_sample=1, rotate=True):
    """
    Reads a halo an return its particles data in faceon projection in pynbody halo format.

    The function also performs the following analysis:
        - Remove DM particles associated with a massive satellite as identified by Emily Cunninghan
        - Remove Stellar particles associated with a massive satellite
        - Remove substructure as identified by 
        - Return only DM particles associated with the massive satellite.
        
    Parameters:
    ----------


    Returns:
    --------

    """

    # Read snapshot
    p = ga.io.Read.read_snapshots(['dark', 'star'], 'index', snap, self.sim_directory, 
                              assign_hosts=True, particle_subsample_factor=1, sort_dark_by_id=True, assign_pointers=True)
   
    # Removing satellite substructure
    npart = len(p['dark'].prop('mass'))
    mask_sub = np.ones(npart, dtype=bool)
    nparts = len(p['star'].prop('mass'))
    mask_subs = np.ones(nparts, dtype=bool)

    if self.rm_sat == True :
        mask_sub[self.sat_ids]=0
        print("* Removing DM particles from massive satellite")       
        
    if self.rm_subs == True :
        mask_sub[self.subs_ids]=0    
        print("* Removing DM substructure in halo")     

    if self.only_sat == True :
        mask_sub = np.zeros(npart, dtype=bool)
        mask_sub[self.sat_ids]=1    
        print("* Only plotting DM particles from the Satellite")     
    
    if self.rm_stellar_sat == True :
        pointers = p.Pointer.get_pointers(species_name_from='star', species_names_to='star')
        stream_inds=pointers[self.stars_ids]
        stream_inds=stream_inds[stream_inds>0]
        mask_subs = np.ones(nparts, dtype=bool)
        mask_subs[stream_inds]=0

    #Building pynbody halo format
    hfaceon = pr.pynbody_halo(p, mask=mask_sub, masks=mask_subs)
    del(p)

    if rotate == True:
        pynbody.analysis.angmom.faceon(hfaceon, cen=(0,0,0))

    return hfaceon


  def get_catids_satsubhalos(self):
    """
    Returns subhalos from a satellite galaxy

    Parameters:
    sim_dir 
    sim : str
        m12b, m12c ...

    Returns: 

    """

    # Read simulations tree 
    halt = halo.io.IO.read_tree(simulation_directory=self.sim_directory)

    # Selects id at peak mass snapshot (smax)
    if self.sim =='m12b':
        self.smax = 300 # snapshot of peak mass
        # 0 is host
        self.sat_index_peak_mass = -2
        nsnaps = 600-self.smax+1

    elif self.sim == 'm12c':
        self.smax = 300
        # 0 is host
        self.sat_index_peak_mass = -4
        nsnaps = 600-self.smax+1

    elif self.sim == 'm12f':
        self.smax = 280
        # 0 is host
        self.sat_index_peak_mass = -4
        nsnaps = 600-self.smax+1

    elif self.sim == 'm12i':
        # 0 is host
        self.sat_index_peak_mass = -11
        self.smax = 300
        nsnaps = 600-self.smax+1

    elif self.sim == 'm12m':
        # 0 is host
        self.sat_index_peak_mass = -19
        self.smax = 300
        nsnaps = 600-self.smax+1

    elif self.sim == 'm12r':
        self.smax = 250
        # 0 is host
        self.sat_index_peak_mass = -2
        nsnaps = 600-self.smax+1

    elif self.sim == 'm12w':
        self.smax = 260 
        # 0 is host
        self.sat_index_peak_mass = -3
        nsnaps = 600-self.smax+1

    # Indices of all subhalos and snapshot where the satellite has maximum mass
    snap_peak_mass = np.where(halt['snapshot'] == self.smax)[0]
    # Find satellite at snap of max mass given its sat_index_peak_mass
    sat_id_peak_mass = np.argsort(halt['mass'][snap_peak_mass])[self.sat_index_peak_mass]
    #print(sat_id_peak_mass)
    # Find subhalos that have the central local index of the satellite
    subhalos_sat = np.where((halt['central.local.index'][snap_peak_mass] == snap_peak_mass[sat_id_peak_mass]) & (halt['mass'][snap_peak_mass]>1e6))[0]
    #subhalos_sat = np.where((halt['central.local.index'][snap_peak_mass] == snap_peak_mass[sat_id_peak_mass]) )[0]
    nsub_sat = len(subhalos_sat)

    print('-> Number of satellite subhalos at peak mass identified by halo finder {} at snapshot {}'.format(nsub_sat, self.smax))
    print('-> Total number of subhalos at snap {}'.format(len(snap_peak_mass)))

    # Find tree subhalos of satellite index at peak mass
    sat_indices = np.zeros(nsub_sat, dtype=int)
    sat_indices = snap_peak_mass[subhalos_sat]
    cat_idx = np.zeros((nsnaps, nsub_sat), dtype=int)
    cat_idx[0] = halt['catalog.index'][sat_indices]

    for k in range(0, nsnaps-1):
        sat_indices = halt['descendant.index'][sat_indices]
        cat_idx[k+1] = halt['catalog.index'][sat_indices]
    return cat_idx


  def subhalos_rotated(self, snap):
    """
    Sat_index = -2 for m12b. Second massive subhhalo at snap 300
    """
    p = ga.io.Read.read_snapshots(['dark', 'star'], 'index', snap, self.sim_directory, 
                              assign_hosts=True, particle_subsample_factor=1, sort_dark_by_id=True)
    #Building pynbody subhalos in halo format
    subhalos = halo.io.IO.read_catalogs('index', snap, self.sim_directory)
    # Tree
    halt = halo.io.IO.read_tree(simulation_directory=self.sim_directory)


    subhalos_ids = self.get_catids_satsubhalos()
    if self.rm_sat == True :
        print("-> Removing satellite subhalos")
        assert (snap-self.smax) >= 0, "snapshot number {} has to be greater than {}".format(snap, self.smax)

        subhalos_snap = subhalos_ids[snap-self.smax]
        print(len(subhalos_snap), len(subhalos['mass']))
        hsub = pr.pynbody_subhalos(subhalos, mask=subhalos_snap)
        hsub_faceon = hsub
    else : 
        hsub = pr.pynbody_subhalos(subhalos)
        hsub_faceon = hsub

    subhalos_init = halo.io.IO.read_catalogs('index', self.smax, self.sim_directory)
    hsub_init = pr.pynbody_subhalos(subhalos_init)
    # Satellite orbit
    sat_id = np.argsort(hsub_init.dark['mass'])[self.sat_index_peak_mass]
    sat_tree_id = subhalos_init['tree.index'][sat_id]
    satellite = return_tracked_pos(halt, sat_tree_id, pynbody_halo=True)
    satellite_faceon = satellite


    h_rotations = pr.pynbody_halo(p)
    del(p)

    faceon, edgeon = pr.make_pynbody_rotations(h_rotations)

    pynbody.transformation.transform(hsub_faceon, faceon)
    pynbody.transformation.transform(satellite_faceon, faceon)
  
    return hsub_faceon, satellite_faceon
  
  def subhalos(self, snap):
      m12_subhalos = halo.io.IO.read_catalogs('index', snap, self.sim_directory)
      hsub = pr.pynbody_subhalos(m12_subhalos)
      
      if self.rm_sat == True :
        subhalos_ids = self.get_catids_satsubhalos()
        print("-> Removing satellite subhalos")
        assert (snap-self.smax) >= 0, "snapshot number {} has to be greater than {}".format(snap, self.smax)
        subhalos_snap = subhalos_ids[snap-self.smax]
        hsub = pr.pynbody_subhalos(m12_subhalos, mask=subhalos_snap)
      
      if self.only_sat == True :
        subhalos_ids = self.get_catids_satsubhalos()
        print("-> Selecting satellite subhalos")
        assert (snap-self.smax) >= 0, "snapshot number {} has to be greater than {}".format(snap, self.smax)
        subhalos_snap = subhalos_ids[snap-self.smax]
        hsub = pr.pynbody_subhalos(m12_subhalos, mask=subhalos_snap)
      return hsub
  

 
  def get_halo_satellite(self, mass_rank):
      """
      Get satellites info from snap 300 to 600
      """
      m12_subhalos = halo.io.IO.read_catalogs('index', 300, self.sim_directory)
      halt = halo.io.IO.read_tree(simulation_directory=self.sim_directory)
      hsub = pr.pynbody_subhalos(m12_subhalos)
      sat_id = np.argsort(hsub.dark['mass'])[mass_rank]
      sat_tree_id = m12_subhalos['tree.index'][sat_id]
      satellite = return_tracked_pos(halt, sat_tree_id, pynbody_halo=True)
      return satellite
