# NBA

N-body Analysis (NBA) is a Python package to analyze N-body simulations of galaxies:
halo centering, density and kinematic profiles, orbits, and simple sky maps.

## Installation

```
$ git clone https://github.com/jngaravitoc/nba.git
$ cd nba/
$ python -m pip install .
```

Python >= 3.10 is required. Optional dependencies are installed with extras:

```
$ python -m pip install ".[extra]"   # healpy, pynbody, FIRE tools (gizmo_analysis, halo_analysis)
$ python -m pip install ".[dev]"     # pytest, flake8
```

## Quick start

```python
import numpy as np
import nba

# Read the MW dark matter halo from a GC21 snapshot (random sub-sample of 500k particles)
reader = nba.ios.ReadGC21("/path/to/snapshots/", "MWLMC5_100M_b0_vir_OM3_G4_110.hdf5")
halo = reader.read_halo(['pos', 'vel', 'mass'], halo='MW', ptype='dm', randomsample=500_000)

# Find the center of mass with a shrinking sphere and re-center the halo
center = nba.com.CenterHalo(halo)
com, vcom = center.shrinking_sphere_numba()
center.recenter(com, vcom)

# Density profile
edges = np.linspace(0, 300, 100)
r, rho = nba.structure.Profiles(halo['pos'], edges).density(mass=halo['mass'])
```

More examples are in [`tutorials/`](tutorials/).

## Modules

| Module | What it does | Extra dependencies |
|---|---|---|
| `nba.ios` | Snapshot readers: `ReadGadgetSim` (Gadget-4 HDF5), `ReadGC21` (Garavito-Camargo+21 MW-LMC), `ReadSheng24` (Sheng+24) | |
| `nba.com` | `CenterHalo`: mean, potential-minimum and shrinking-sphere centers | |
| `nba.structure` | `Profiles`: density, enclosed mass and potential profiles | |
| `nba.kinematics` | `Kinematics`: velocity dispersions, anisotropy beta, angular momentum and orbital poles | scikit-learn (nearest-neighbor maps) |
| `nba.orbits` | `orbit`: center-of-mass orbits from a series of snapshots | |
| `nba.cosmology` | `Cosmology`: virial and r200 quantities, NFW conversions | |
| `nba.visuals.mollweide` | HEALPix density maps and Mollweide plots | healpy |
| `nba.utils` | `Grid3D`: grids in Cartesian, cylindrical and spherical coordinates | |

Importing `nba.visuals.plotting` additionally requires pynbody.

Readers for other simulation formats can be added in `nba/ios/snap_reader.py`.

## Parallelization

Many routines can be parallelized with [Schwimmbad](https://schwimmbad.readthedocs.io/en/latest/index.html)
(`pip install schwimmbad`). See [compute_orbit_parallel.py](tutorials/compute_orbit_parallel.py)
for an example that computes the orbits of the MW and the LMC.

## Tests

```
$ python -m pip install ".[dev]"
$ pytest
```

## Known issues

- `nba.kinematics` and `nba.structure.density_tools` have undefined names in some less-used
  routines (reported as warnings by flake8 in CI).
- `CenterHalo.recenter` modifies the arrays it was created with in place.
- `ReadGC21.read_halo` splits the MW and the LMC dark matter assuming 100M MW particles.
- `legacy/` holds code that no longer works with the current API; it is not installed.

## License

MIT, see [LICENSE](LICENSE).
