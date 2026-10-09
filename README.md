# NBA

N-body Analysis (NBA) is a Python package to analyze N-body simulations of galaxies:
halo centering, density and kinematic profiles, orbits, and simple sky maps.

## Installation

```
$ git clone https://github.com/jngaravitoc/nba.git
$ cd nba/
$ python -m pip install .
```

Python >= 3.10 is required. The `devel` branch is the development version (0.4.0.dev0); see
[CHANGELOG.md](CHANGELOG.md) for the changes since 0.3.0, some of which are not backwards compatible. Optional dependencies are installed with extras:

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

## Centering halos and computing orbits

`nba.com.CenterHalo` has three centering methods:

| Method | Use it for |
|---|---|
| `shrinking_sphere_numba` (and the NumPy reference `shrinking_sphere`) | Any halo, and the only reliable choice for satellites ([Power et al. 2003](https://ui.adsabs.harvard.edu/abs/2003MNRAS.338...14P)) |
| `min_potential` | Host galaxies only. Snapshots store the total potential, so for a satellite it finds its stripped particles in the host's potential well |
| `mean_pos` | A first guess, or a satellite before it is stripped |

The shrinking sphere options after `delta` are keyword-only. The most useful ones:

- `softening`: stop before the sphere is smaller than 4 × the softening length.
- `center0`, `r0`: start from a sphere of radius `r0` around a previous center, so the sphere stays on the
  halo when debris or another halo overlaps it.
- `rvel_factor` or `nvel`: compute the velocity from a region tied to the final sphere (a multiple of its
  radius, or the nearest particles) instead of the fixed `rcut_vel = 20` sphere, which picks up debris.
- `return_info=True`: also return the final radius, the number of particles and the mean density in the
  final sphere, which show when the center is no longer well defined.

`nba.orbits.orbit` and `iter_orbit` follow a halo through a series of snapshots. For a satellite, the setup
recommended by the tests on the GC21 MWLMC5 simulation ([`reports/`](reports/)) is:

```python
from nba.orbits import orbit, read_orbit

pos, vel = orbit("/path/to/snapshots/", "MWLMC5_100M_b0_vir_OM3_G4_{:03d}.hdf5", range(451),
                 halo="LMC", com_method="shrinking_sphere_numba",
                 softening=0.08, r0=15, rvel_factor=5, min_density_ratio=0.01,
                 time_offset=lambda k: 8.0 if k >= 400 else 0.0,  # restart that reset the time
                 outfile="lmc_orbit_{method}.ecsv")
```

- `r0` tracks the center from one snapshot to the next.
- `time_offset` corrects snapshots whose header time was reset by a restart.
- **Warnings** flag snapshots where the center may be wrong:
  - it jumps much further than |v| Δt (`jump_factor`);
  - it moves inconsistently with its velocity for several snapshots (`velocity_tol`, `velocity_window`);
  - the density in the final sphere falls below a fraction of its first value (`min_density_ratio`), as the
    satellite dissolves;
  - the time decreases.
- **`outfile`** writes one self-describing [ECSV](https://docs.astropy.org/en/stable/io/ascii/ecsv.html) file
  per method:
  - units on every column;
  - the per-snapshot diagnostics;
  - every parameter as used, and the warnings;
  - the nba version and commit. Add `full_provenance=True` to also record the user, host and command.

  Read it back with `read_orbit(path)`, which returns an astropy Table.

## Modules

| Module | What it does | Extra dependencies |
|---|---|---|
| `nba.ios` | Snapshot readers: `ReadGadgetSim` (Gadget-4 HDF5), `ReadGC21` (Garavito-Camargo+21 MW-LMC), `ReadSheng24` (Sheng+24) | |
| `nba.com` | `CenterHalo`: shrinking-sphere, potential-minimum (hosts only) and mean centers | |
| `nba.structure` | `Profiles`: density, enclosed mass and potential profiles | |
| `nba.kinematics` | `Kinematics`: velocity dispersions, anisotropy beta, angular momentum and orbital poles | scikit-learn (nearest-neighbor maps) |
| `nba.orbits` | `orbit`, `iter_orbit`: center-of-mass orbits from a series of snapshots; `write_orbit`, `read_orbit`: ECSV orbit files | |
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

`tests/test_centering_lmc.py` tests the centering on a real subsample of the LMC in GC21 MWLMC5. Its particle
file (42 MB) is not in the repository, and these tests are skipped without it. To run them, set
`NBA_TEST_DATA` to the folder that holds the file. See [`tests/data/README.md`](tests/data/README.md).

## Known issues

- `nba.kinematics` and `nba.structure.density_tools` have undefined names in some less-used
  routines (reported as warnings by flake8 in CI).
- `CenterHalo.recenter` modifies the arrays it was created with in place, unless `copy=True`.
- `ReadGC21.read_halo` splits the MW and the LMC dark matter assuming 100M MW particles
  (`ReadGC21.npart_mw`).
- Once a satellite has dissolved, it has no physical center. In MWLMC5 this happens after about 7 Gyr,
  and the shrinking sphere then follows phase-mixed debris. Use the warnings and `info["density"]` to
  decide where to stop trusting the orbit.
- `legacy/` holds code that no longer works with the current API; it is not installed.

## License

MIT, see [LICENSE](LICENSE).
