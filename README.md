# NBA

N-body Analysis (NBA) is a python package designed to analyze N-body simulations of galaxies.


## Installation:


NBA can be installed by cloning the repository and installing it locally:

```
$ git clone https://github.com/jngaravitoc/nba.git`
$ cd nba/
$ python -m pip install .
```

## Snapshots input format: 

At the moment the code has readers for Gadget2/3/4, and ASCII files. However, readers for other simulations 
output can easily be implemented in `nba/ios/`

## Parallelization:

Many of the routines in NBA can easily be parallelized with python packages such as [Schwimmbad](https://schwimmbad.readthedocs.io/en/latest/index.html). An example showing how to compute
the orbit of two halos can be [found here](https://github.com/jngaravitoc/nba/blob/main/tutorials/compute_orbit_parallel.py). 


## Routines:

At the moment the code has modules to perform the following analysis: 
- Center of mass computations
- Orbit calculations
- Density estimation 
- Compute halo shapes and angular momentum. 
- Kinematics properties such as anisotropy parameter, velocity dispersion, and orbital poles.
- Coordinate transformations using [astropy](https://github.com/astropy/astropy).
- Basic visualization in cartesian and mollweide projections.
- Compute structural properties of DM halos given a cosmology.
- tools to compute virial and r200 quantities and to transform between those. 

## Dependencies:

- scipy
- matplotlib
- astropy
- pygadgetreader 
- healpy (visualization)
