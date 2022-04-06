#!/sr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def re_center(vec, cm):
    """
    Subtract a vector from a each dimension of another vector, this is done to recenter a halo
    positions and velocities to its center of mass.

    Input:
    ------
    vec : numpy.array
        A numpy array to which substract the vector cm
    cm : numpy array
        A numpy 1d array with

    Output:
    -------

    numpy.array
        A new vector with a subtracted vector in each dimension.
    """
    #assert len(vec)==len(cm), "Make sure the len of your N-vector is the same as your 1d vector"

    new_vec = np.copy(vec)
    for i in range(len(cm)):
      new_vec[:,i] = vec[:,i] - cm[i]
    return new_vec





def com_disk_potential(xyz, vxyz, Pdisk):
    V_radius = 2
    vx = vxyz[:,0]
    vy = vxyz[:,1]
    vz = vxyz[:,2]
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    min_pot = np.where(Pdisk==min(Pdisk))[0]
    x_min = x[min_pot]
    y_min = y[min_pot]
    z_min = z[min_pot]
    # This >2.0 corresponds to the radius in kpc of the particles that
    # I am taking into account to compute the CM
    avg_particles = np.where(np.sqrt((x-x_min)**2.0 + (y-y_min)**2.0 + (z-z_min)**2.0)<V_radius)[0]
    x_cm = sum(x[avg_particles])/len(avg_particles)
    y_cm = sum(y[avg_particles])/len(avg_particles)
    z_cm = sum(z[avg_particles])/len(avg_particles)
    vx_cm = sum(vx[avg_particles])/len(avg_particles)
    vy_cm = sum(vy[avg_particles])/len(avg_particles)
    vz_cm = sum(vz[avg_particles])/len(avg_particles)
    return np.array([x_cm, y_cm, z_cm]), np.array([vx_cm, vy_cm, vz_cm])

def velocities_com(cm_pos, pos, vel, r_cut=20):
    """
    Function to compute the COM velocity in a sphere of 20 kpc
    """
    # Compute the distance with respect to the COM
    R_cm = ((pos[:,0]-cm_pos[0])**2 + (pos[:,1]-cm_pos[1])**2 + (pos[:,2]-cm_pos[2])**2)**0.5
    # Select the particles inside 15 kpc
    index = np.where(R_cm < r_cut)[0]
    # Compute the velocities of the COM:
    velx_cm = np.sum(vel[index,0])/len(vel[index,0])
    vely_cm = np.sum(vel[index,1])/len(vel[index,1])
    velz_cm = np.sum(vel[index,2])/len(vel[index,2])

    return np.array([velx_cm, vely_cm, velz_cm])



def mean_pos(xyz, vxyz, m):
    """
    Returns the COM positions and velocities.

    .. math:: \vec{R} = \sum_i^N m_i \vec{r_i} / N

    """


    # Number of particles
    N = np.sum(m)


    xCOM = np.sum(xyz[:,0]*m)/N
    yCOM = np.sum(xyz[:,1]*m)/N
    zCOM = np.sum(xyz[:,2]*m)/N

    vxCOM = np.sum(vxyz[:,0]*m)/N
    vyCOM = np.sum(vxyz[:,1]*m)/N
    vzCOM = np.sum(vxyz[:,2]*m)/N

    return np.array([xCOM, yCOM, zCOM]), np.array([vxCOM, vyCOM, vzCOM])


def shrinking_sphere(xyz, vxyz, m, delta=0.025):
    """
    Compute the center of mass coordinates and velocities of a halo
    using the Shrinking Sphere Method Power et al 2003.
    It iterates in radii until reach a convergence given by delta
    of 1% of the total number of particles while there are more than 1000
    particles.


    Parameters
    -----------
    xyz: numpy.array
        cartesian coordinates with shape (n,3)
    vxys: numpy.array
        cartesian velocities with shape (n,3)
    delta: float, optional
        Convergence of the COM computation in the same units of `xyz', D=0.025

    Returns
    --------

    rcm: numpy.array
        Arrays containing the coordinate of the center of mass with respect
        to a (0,0,0) point.

    vcm: numpy.array
        Arrays containing the velocities of the center of mass with respect
        to a (0,0,0) point.

    References
    -----------
    .. [1] Power, C., ``The inner structure of ΛCDM haloes - I. A numerical convergence study", MNRAS, vol. 338, no. 1, pp. 14–34, 2003. doi:10.1046/j.1365-8711.2003.05925.x.


    """
    N_i = len(xyz)
    N = N_i


    xCM = 0.0
    yCM = 0.0
    zCM = 0.0


    rCOM, vCOM = mean_pos(xyz, vxyz, m)
    xCM_new, yCM_new, zCM_new = rCOM
    vxCM_new, vyCM_new, vzCM_new = vCOM



    while (((np.sqrt((xCM_new-xCM)**2 + (yCM_new-yCM)**2 + (zCM_new-zCM)**2) > delta) & (N>N_i*0.01)) | (N>1000)):
        xCM = xCM_new
        yCM = yCM_new
        zCM = zCM_new
        # Re-centering sphere
        R = np.sqrt((xyz[:,0]-xCM_new)**2 + (xyz[:,1]-yCM_new)**2 + (xyz[:,2]-zCM_new)**2)
        Rmax = np.max(R)
        # Reducing Sphere by its 2.5%
        index = np.where(R<Rmax*0.975)[0]
        xyz = xyz[index]
        vxyz = vxyz[index]
        m = m[index]
        N = len(xyz)
        #Computing new CM coordinates and velocities
        rCOM, vCOM = mean_pos(xyz, vxyz, m)
        xCM_new, yCM_new, zCM_new = rCOM
        vxCM_new, vyCM_new, vzCM_new = vCOM

    vxCM_new, vyCM_new, vzCM_new = velocities_com([xCM_new, yCM_new, zCM_new], xyz, vxyz)
    return np.array([xCM_new, yCM_new, zCM_new]), np.array([vxCM_new, vyCM_new, vzCM_new])

