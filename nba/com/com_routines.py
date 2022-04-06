#!/sr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .com_methods import mean_pos, re_center, shrinking_sphere
#Function that computes the center of mass for the halo and disk and
# the corresponsing orbits for the host and satellite simultaneously


def mix_com(pos, vel, mass):
    """
    Experimental function that mixes several COM methods.
    description:
        1. Pick a 200 kpc volume and center using mean_pos
        2. Pick a 50 kpc in around the previously found COM and compute new com using mean_pos
        3. Use the shriniking algorith to compute COM in a 50 kpc sphere centered
            in previously found COM
        4. Repeat 3) in using shrinking sphere around a 20 kpc region.

    TODO: Organize this Function and test this functon.
    """
    # TODO: organize this Function
    # Guess com to re-center halo and precisely compute the COM
    rlmc = np.sqrt(np.sum(pos**2, axis=1))
    #truncate = np.where(rlmc < 600)[0]
    # First COM guesslif com_frame == 'sat':lif com_frame == 'sat':
    print('Computing coordinates in the satellite COM frame')
    print('Computing coordinates in the satellite COM frame')
    pos1 = np.copy(pos)
    vel1 = np.copy(vel)

    #com1 = com.COM(pos1[truncate], vel1[truncate], np.ones(len(pos[truncate])))
    com1 = mean_pos(pos1, vel1, np.ones(len(pos)))
    pos_recenter = re_center(pos1, com1[0])
    vel_recenter = re_center(vel1, com1[1])

    rlmc = np.sqrt(pos_recenter[:,0]**2 + pos_recenter[:,1]**2 +  pos_recenter[:,2]**2)
    truncate = np.where(rlmc < 200)[0]

    com2 = mean_pos(pos_recenter[truncate], vel_recenter[truncate], np.ones(len(truncate))*mass[0])
    pos_recenter2 = re_center(pos_recenter, com2[0])
    vel_recenter2 = re_center(vel_recenter, com2[1])

    rlmc = np.sqrt(pos_recenter2[:,0]**2 + pos_recenter2[:,1]**2 + pos_recenter2[:,2]**2)
    truncate2 = np.where(rlmc < 50)[0]

    com3 = shrinking_sphere(pos_recenter2[truncate2], vel_recenter2[truncate2],
                                np.ones(len(truncate2))*mass[0])

    pos_recenter3 = re_center(pos_recenter2, com3[0])
    vel_recenter3 = re_center(vel_recenter2, com3[1])


    rlmc = np.sqrt(pos_recenter3[:,0]**2 + pos_recenter3[:,1]**2 + pos_recenter3[:,2]**2)
    truncate3 = np.where(rlmc < 20)[0]

    com4 = shrinking_sphere(pos_recenter3[truncate3], vel_recenter3[truncate3],
                                np.ones(len(truncate3))*mass[0])

    print(com1)
    print(com2)
    print(com3)
    print(com4)
    pos_cm = com1[0] + com2[0] + com3[0] + com4[0]
    vel_cm = com1[1] + com2[1] + com3[1] + com4[1]
    print(pos_cm, vel_cm)
    return pos_com, vel_com
