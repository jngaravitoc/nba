"""
Script that contains functions to compute dark matter halo structural quantities for a given cosmology

TODO:
- Double check units. Many codes have different units to the ones define in Astropy. 
- Add tests 
- finish documentation

"""

import numpy as np
from scipy.optimize import bisect
from astropy import units
from astropy import constants   


def H(z, H_0=None, Omega0=0.27):
    """
    Function that computes the Hubble constant as a function of redshift z 
    for a given cosmology.

    Parameters:
    ----------- 
    z: float
        Redshift       
    H_0: float
        Hubble constant in units of km/s/Mpc
    Omega0: float
        Density parameter at z=0
    Returns:
    --------    
    Hubble parameter at redshift z in units of km/s/Mpc
    """
    
    if H_0 is None:
        H_0 = 67.8 * units.km / (units.s * units.Mpc)
        H_0 = 3.2407789E-18  / units.s * 0.7
        H_0 = H_0.to(units.km / units.s / units.Mpc)

    print('Using H_0 = ', H_0)

    Lambda0 = 1. - Omega0
    return H_0*(Omega0*(1+z)**3 - (Omega0+Lambda0-1)*(1+z)**2 + Lambda0)**0.5

def Omega_z(z, H_0=67.8 * units.km / (units.s * units.Mpc), Omega0=0.27):
    """
    function that compute the density parameter at redshift z
    for a given cosmology.
    Parameters:
    -----------
    z: float    
        Redshift
     H_0: float
        Hubble constant in units of km/s/Mpc
    Omega0: float
        Density parameter at z=0
    Returns:    
    --------
    Density parameter at redshift z
    
    """
    
    return Omega0 * (1+z)**3 * (H_0/H(z, H_0, Omega0))**2

def rho_crit(z, H_0=67.8 * units.km / (units.s * units.Mpc), Omega0=0.27):
    """
    Function that compute the critical density at redshift z
    for a given cosmology.
    Parameters:
    -----------
    z: float    
        Redshift    
     H_0: float
        Hubble constant in units of km/s/Mpc
    Omega0: float
        Density parameter at z=0
    Returns:
    --------
    Critical density at redshift z
    """
    H2 = H(z, H_0, Omega0)**2
    rho = 3*H2 / (8*np.pi*G)
    return rho

def Dvir(z, H_0=67.8 * units.km / (units.s * units.Mpc), Omega0=0.27):
    """
    Virial overdensity as a function of redshift from the solution of the top hat model:
    Bryan and Norman (1998) or euquation B1 in Klypin et al. (2011)
    Parameters:
    -----------
    z: float
        Redshift
     H_0: float
        Hubble constant in units of km/s/Mpc
    Omega0: float
        Density parameter at z=0
    Returns:
    --------
    Virial overdensity at redshift z
     
    
    """
    Omegaz = Omega_z(z, H_0, Omega0)
    x = Omegaz - 1
    Deltavir =  ((18*np.pi**2) +  (82*x) - 39*x**2) / Omegaz
    return Deltavir

def rvir(Mvir, z, H_0=67.8 * units.km / (units.s * units.Mpc), Omega0=0.27, physical_units=True):
    """
    Function that computes the virial radius for a given mass and redshift and cosmology.
    Parameters:
    -----------
    Mvir: float
        Virial mass in units of Msun
    z: float        
        Redshift
        H_0: float
        Hubble constant in units of km/s/Mpc
    Omega0: float
        Density parameter at z=0
    physical_units: boolean
        If True, the function returns the virial radius in physical units (kpc)
            Following Besla 2007 or Equation A1 in Van Der marel 2012
        If False, the function returns the virial radius in units of Rvir/Rvir(z=0)
    Returns:    
    --------
    Virial radius in units of kpc or Rvir(z=0)  
    

    """

    if physical_units == False: 

        Mvir = Mvir * units.Msun
        Deltavir = Dvir(z, H_0, Omega0)
        pcrit = rho_crit(z, H_0, Omega0)
        Rvir = (3*Mvir / (4 * np.pi * Deltavir * pcrit * Omega0))**(1/3.)
        Rvir = Rvir.to(units.kpc)
    
    if physical_units == True:
        h = 0.704
        Rvvir = 206/h * (Dvir(z, H_0, Omega0) * Omega0 / 97.2)**(-1.0/3.0) * (Mvir*h/(1E12))**(1.0/3.0)
        Rvir = Rvir * units.kpc
    return Rvir


def concentration(Mvir, h=0.7, subhalos=False):
    """
    Function that computes the concentration for a given virial mass following Kylpin et al. 2011 Eqns 10 and 11
    Parameters:
    -----------
    Mvir: float
        Virial mass in units of Msun
    h: float
        Hubble constant in units of 100 km/s/Mpc
    subhalos: boolean   
        If True, the function returns the concentration for subhalos
        If False, the function returns the concentration for host halos
    Returns:
    --------
    Concentration for a given virial mass

    """
    if subhalos == True:
        c = 12 * (Mvir * h/ 1E12)**(-0.12)
    else:
        c = 9.60 * (Mvir * h/ 1E12)**(-0.075)
    return c


def fx(x):
    """
    function that computes the f function of the NFW profile. See Equation A3 in Van der Marel 2012/
    Parameters:
    -----------
    x: float
        concentration x = r/r_s; where r_s is the scale radius of the NFW profile 
    Returns:
    --------
    f(x) function of the NFW profile

    """
    f = np.log(1.+x) - (x / (1. + x))
    return f


def cx(c_guess, c_given, given):
    """
    function that relates the concetrations at the viral radius rvir and r200 for the NFW profile 
    See Eqn. A6 in the Van der Marel 2012 paper. 
    Parameters:
    -----------
    cvir: float
        concentration at the virial radius
    c200: float
        concentration at r200   
    Returns:
    --------
    y : floaot
        value tu minimize to find the desried concentration
    """
    q = 2.058
    
    if given == 'virial':
        y = (c_guess / c_given) - (fx(c_guess) / (q * fx(c_given)))**(1./3.)
    if given == '200':
        y = (c_given / c_guess) - (fx(c_given) / (q * fx(c_guess)))**(1./3.)
    return y

# Function to compute the c200
def cvirc200(cvir=None, c200=None):
    """
    
    """
    if c200 == None:
        c_guess = bisect(cx, 0.1, cvir, args=((cvir, 'virial')))     
    if cvir == None:
        c_guess = bisect(cx, 0.1, 2*c200, args=((c200, '200')))
        
    return c_guess

def m200mvir(c200, cvir):
    """
    x = M200/Mvir

    """
    x = fx(c200) / fx(cvir)
    return x


def NFW_virial(M200, c200):
    cvir = cvirc200(c200=c200)
    Mvir = M200 / m200mvir(c200, cvir)
    return Mvir, cvir

def NFW_200(Mvir, cvir):
    c200 = cvirc200(cvir=cvir)
    M200 = Mvir * m200mvir(c200, cvir)
    return M200, c200


# function that computes the a/rs
def ars(c):
    """
    c 
    """
    x = 1 / ((2.0*fx(c))**(-0.5) - (1.0/c))
    return x


#Function that computes Mh/Mvir
def mhmvir(ar, cvir):
    x = ar**2 / (2.0*fx(cvir))
    return x



def v200(M):
    M = M * units.Msun
    G = constants.G
    H = 3.2407789E-18  / units.s * 0.7 
    v = (M * 10 * G * H)**(1.0/3.0)
    v = v.to(units.km/units.s)
    return v

def vvir(M):
    M = M * units.Msun
    G = constants.G
    H = 3.2407789E-18  / units.s * 0.7
    v = (M*np.sqrt(48.6)*G*H)**(1.0/3.0)
    v = v.to(units.km / units.s)
    return v

def R200(v200):
    H = 3.2407789E-18  / units.s * 0.7
    r200 = v200 / (10.0 * H)
    r200 = r200.to(units.kpc)
    return r200

def ars_v(c200, r200):
    a = r200 / c200 * np.sqrt(2 * np.log(1 + c200) - c200/(1+c200))
    return a


