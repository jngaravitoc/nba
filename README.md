# NBA

N-body Analysis (NBA) is a Python package designed to analyze N-body simulations of galaxies.


## Installation:


NBA can be installed by cloning the repository and installing it locally:

```
$ git clone https://github.com/jngaravitoc/nba.git`
$ cd nba/
$ python -m pip install .
```

## Snapshots input format: 

Currently, the code supports readers for Gadget-4 and ASCII files. However, readers for other simulation outputs can easily be implemented in `nba/ios/`

## Parallelization:

Many of the routines in `nba` can easily be parallelized with Python packages such as [Schwimmbad](https://schwimmbad.readthedocs.io/en/latest/index.html). An example showing how to compute
the orbit of two halos can be [found here](https://github.com/jngaravitoc/nba/blob/main/tutorials/compute_orbit_parallel.py). 


## Routines:

At the moment, the code has modules to perform the following analysis: 
- Center of mass computations
- Density estimation 
- Compute halo shapes and angular momentum. 
- Kinematics properties such as anisotropy parameter, velocity dispersion, and orbital poles.
- Coordinate transformations using [astropy](https://github.com/astropy/astropy).
- Basic visualization in Cartesian and Mollweide projections.
- Compute structural properties of DM halos given a cosmology.
- Tools to compute virial and r200 quantities and to transform between those. 

## Dependencies:

- scipy
- matplotlib
- astropy
- healpy (visualization)
