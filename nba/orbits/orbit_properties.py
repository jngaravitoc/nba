import numpy as np

def orbit_nearest_point(pos, vel, pos_obs, vel_obs):
    """
    Computes the closes value of an orbit to a given 3d point.
    
    Parameters:
    -----------
    pos : numpy.array
        orbits positions
    vel : numpy.array
        orbits velocities
    pos_obs : numpy.array
        orbital point
    vel_obs : numpy.array
        orbital velocity point

    Returns:
    --------

    index : int
        Index of point of close proximity
    nearest_pos : numpy.array
        Nearest positions point in the orbit
    nearest_vel : numpy.array
        Velocity at nearest position point
    
    """
    delta_r = np.sqrt(np.sum((pos-pos_obs)**2, axis=1))
    delta_v = np.sqrt(np.sum((vel-vel_obs)**2, axis=1))
    
    min_r = np.argmin(delta_r)   
  
    return min_r, pos[min_r], vel[min_r]
