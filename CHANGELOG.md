# Changelog

All notable changes to this project will be documented in this file.

Versions before 0.3.0 were originally numbered 1.0.1, 1.1 and 1.2; they were renumbered 0.1.0,
0.1.1 and 0.2.0 to match the package metadata and the pre-1.0 state of the API.

## [Unreleased]
### Changed
- `CenterHalo.shrinking_sphere`, `shrinking_sphere_numba` and `ssphere_numba`: all parameters
  after `delta` are keyword-only (the two methods took `min_npart` and `rcut_vel` in a different
  order, so positional calls gave different results)
- An empty velocity region in the shrinking sphere raises a ValueError instead of returning NaN
- `CenterHalo.mean_pos`: `rmax=None` (the new default) means no upper limit, so `rmin` alone can
  be given; masses are optional
- `iter_orbit`/`orbit` refuse `min_potential` and `diskpot` for `halo='LMC'`: the snapshot
  potential is the total potential, so for a satellite they find the host's potential well
- `ReadGC21.read_halo` splits the MW and LMC with an O(N) ID threshold instead of sorting all IDs
- `tutorials/compute_lmc_orbits.py` no longer runs `min_potential`; it uses `softening=0.08` and
  `r0=15` by default
### Added
- Shrinking sphere: `npart_frac` (None removes the 1% cap on `min_npart`), `rvel_factor` (velocity
  from a multiple of the final radius) and `nmin`/`density` in `info`
- `CenterHalo.min_potential(return_info=True)` and a warning when fewer than 10 particles are
  averaged
- `iter_orbit`: `min_npart`, `npart_frac`, `nvel`, `rvel_factor`, `time_offset`, `jump_factor`
  and `return_info`; warnings when the center jumps, the time decreases or no softening is given
- `ReadGC21.read_halo(seed=...)`
### Fixed
- The shrinking sphere's `npart` uses r < R, as the loop does, when it stops before shrinking
- `ReadGC21.read_halo(randomsample=n)` returns exactly n particles (it drew with replacement)
- `nba.ios.snap_reader` no longer calls `logging.basicConfig` on import; header fields are logged
  at DEBUG level

## [0.3.0] - 2026-10-02
### Added
- Test suite (`tests/`) and flake8 configuration
- `nba.orbits.orbit` rewritten on top of `ReadGC21` and `CenterHalo`; it accepts several centering
  methods at once, and `nba.orbits.iter_orbit` yields the centers snapshot by snapshot (each
  snapshot is read once)
- `tutorials/compute_lmc_orbits.py` and `tutorials/lmc_centering.ipynb` to compare the centering
  methods on the LMC
- `nba.visuals.mollweide` (HEALPix density maps and Mollweide plots)
- `__version__` in `nba/__init__.py`
- `.gitignore`
### Fixed
- `nba.kinematics` (duplicate argument in `slice_NN`) and `nba.cosmology`
  (class was commented out) can be imported again
- Packaging: dynamic version in `pyproject.toml`; numba and scikit-learn are core dependencies
- CI installs the package, runs on `devel` and pull requests, and no longer fails without tests
- Tutorial scripts and notebooks updated to the current API
- `Kinematics.profiles` bins now match `nba.structure.Profiles` (they were offset by half a bin and
  extended past `rmax`); `Kinematics.dr` holds the bin centers and `profiles` no longer overwrites
  `pos`/`vel`
### Removed
- Non-working modules moved to `legacy/` (not installed)
- Cell outputs stripped from tutorial notebooks

## [0.2.0] - 2025-12-22
### Added
- Included functionality to read Sheng+24 data
- Added 3D grid generation functionality as part of utils
### Fixed:
- Consistent binning strategy in Profile for enclosed mass, potential, and
  density 

## [0.1.1] - 2025-05-31
### Added 
- Included test units for the main code functionality
- Included basic documentation
- Move out analysis files out of the repo
### Fixed
- Issue #10

## [0.1.0] - 2025-05-27
### Added 

- Changelog file
- Including scf_utils scipt into __init__.py
### Fixed
- Issue #27 


## [0.0.0] - 2024-04-26
### Initial Release
- First stable release of the library.

