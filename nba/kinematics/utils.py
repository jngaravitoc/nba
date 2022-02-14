import bfe
import sys
import scipy.linalg as la
from astropy.coordinates import Angle
from astropy import units as u
import numpy as np
import random
#from random import sample

def orbpole(pos, vel):
    # Hacked from Ekta's code
    # r x v
    uu = np.cross(pos, vel)
    # |r.v|
    uumag = la.norm(uu, axis=1)
    u = uu.T/uumag
    b = np.arcsin(u[2])
    sinl = u[1]/np.cos(b)
    cosl = u[0]/np.cos(b)
    ll = np.arctan2(sinl,cosl)
    
    gl = np.degrees(ll)
    gb = np.degrees(b)
    return gl, gb
