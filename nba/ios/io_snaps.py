import numpy as np
from nba.ios.read_snap import load_snapshot as readsnap
from nba.ios.gadget_reader import is_parttype_in_file
import nba.com as com

def halo_ids(pids, list_num_particles, gal_index):
    """
    If in an snapshot there are several halos and the particle ids are arranged
    by halos i.e first N ids are of halo 1 second P ids are of halo 2 etc.. Then
    this function could be used to return properties of a halo given its ids order.

    Parameters
    -----------
    pids: numpy.array
        array with all the DM particles ids
    list_num_particles: list
        A list with the number of particles of all the galaxies in the ids.
        [1500, 200, 50] would mean that the first halo have 1500 particles, the
        second halo 200, and the third and last halo 50 particles.

    gal_index: int
        Index of the needed halo or galaxy.


    Return
    --------
    list of arrays
        With the properties of the desired halo


    """

    #assert len(xyz)==len(vxyz)==len(pids), "your input parameters have different length"
    assert type(gal_index) == int, "your galaxy type should be an integer"
    assert gal_index >= 0, "Galaxy type can't be negative"

    sort_indexes = np.sort(pids)
    Ntot = len(pids)

    if gal_index ==0:
        N_cut_min = sort_indexes[0]
        N_cut_max = sort_indexes[int(sum(list_num_particles[:gal_index+1])-1)]

    elif gal_index == len(list_num_particles)-1:
        N_cut_min = sort_indexes[int(sum(list_num_particles[:gal_index]))]
        N_cut_max = sort_indexes[-1]

    else:
        N_cut_min = sort_indexes[int(sum(list_num_particles[:gal_index]))]
        N_cut_max = sort_indexes[int(sum(list_num_particles[:gal_index+1]))]

    # selecting halo ids
    halo_ids = np.where((pids>=N_cut_min) & (pids<=N_cut_max))[0]
    assert len(halo_ids) == list_num_particles[gal_index], 'Something went wrong selecting the satellite particles'

    return halo_ids


def load_halo(snap, N_halo_part, q, com_frame=0, galaxy=0,
              snapformat=3, com_method='shrinking'):
    """
    Returns the halo properties.

    Parameters:
    path : str
        Path to the simulations
    snap : str
        name of the snapshot
    N_halo_part : list
        A list with the number of particles of all the galaxies in the ids.
        [1500, 200, 50] would mean that the first halo have 1500 particles, the
        second halo 200, and the third and last halo 50 particles.
    q : list
        Particles properties: ['pos', 'vel', 'pot', 'mass'] etc.. IDs are necessay and loaded by default.
    com_frame : int
        In which halo the coordinates will be centered 0 -> halo 1, 1 -> halo 2 etc..
    galaxy : int
        Halo particle data to return  0 -> halo 1, 1 -> halo 2 etc...
    snapformat: int
        0 -> Gadget binnary, 2 -> ASCII, 3 -> Gadget HDF5
    com_method : str
        Method to compute the COM 'shrinking', 'diskpot', 'mean_pos'

    Returns:
    --------
    pos : numpy.array
    vel : numpy.array
    pot : numpy.array
    mass : numpy.array

    TODO: Leave vel and mass as args when computing COM.

    """
    # Load data
    print("Loading snapshot: " + snap)
    # TBD: Define more of quantities, such as acceleration etc.. unify this beter with ios

    all_ids = readsnap(snap, snapformat, 'pid', 'dm')
    ids = halo_ids(all_ids, N_halo_part, galaxy)

    all_pos = readsnap(snap, snapformat, 'pos', 'dm')
    pos = all_pos[ids]
    all_vel = readsnap(snap, snapformat, 'vel', 'dm')
    vel = all_vel[ids]
    if 'pot' in q:
        all_pot = readsnap(snap, snapformat, 'pot', 'dm')
        pot = all_pot[ids]
    if 'mass' in q:
        all_mass = readsnap(snap, snapformat, 'mass', 'dm')
        mass = all_mass[ids]

    #if galaxy == 1:
    print("Loading halo {} particle data".format(galaxy))

    print("Computing coordinates in halo {} reference frame".format(com_frame))

    if com_frame == galaxy:
        pos_com, vel_com = com.get_com(pos, vel, mass, com_method, snapname=snap, snapformat=snapformat)
        new_pos = com.re_center(pos, pos_com)
        new_vel = com.re_center(vel, vel_com)

    else:
        ids_rf = halo_ids(all_ids, N_halo_part, com_frame)

        pos_com, vel_com = com.get_com(all_pos[ids_rf], all_vell[ids_rf], all_mass[ids_rf], com_method)
        new_pos = com.re_center(pos, pos_com)
        new_vel = vel.re_center(vel, vel_com)

    del(pos, vel)

    return new_pos, new_vel#, pot, mass, ids, pos_com, vel_com
