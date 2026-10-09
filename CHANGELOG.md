# Changelog

All notable changes to this project will be documented in this file.

Versions before 0.3.0 were originally numbered 1.0.1, 1.1 and 1.2; they were renumbered 0.1.0,
0.1.1 and 0.2.0 to match the package metadata and the pre-1.0 state of the API.

## [Unreleased]
### Breaking changes
- `CenterHalo.shrinking_sphere`, `shrinking_sphere_numba` and `ssphere_numba`: all parameters after
  `delta` are keyword-only (the two methods took `min_npart` and `rcut_vel` in a different order, so
  positional calls gave different results)
- An empty velocity region in the shrinking sphere raises a ValueError instead of returning NaN
- `iter_orbit`/`orbit` refuse `min_potential` and `diskpot` for `halo='LMC'`: the snapshot potential is
  the total potential, so for a satellite they find the host's potential well
- Shrinking sphere `info['npart']` counts the particles within `info['radius']` of the returned center,
  like `info['density']` (it counted the last sphere, around the previous center)
- `nba.ios.snap_reader` no longer calls `logging.basicConfig` on import; header fields are logged at
  DEBUG level
- `tutorials/compute_lmc_orbits.py` requires the snapshot folder, no longer runs `min_potential`, and
  defaults to `softening=0.08`, `r0=15`, `rvel_factor=5` and `min_density_ratio=0.01`

### Added
- Documentation (Sphinx), published at https://jngaravitoc.github.io/nba/ from `main` by a GitHub
  Actions workflow: installation, getting started, user guide pages for `nba.ios` and `nba.com` with
  tested examples, API pages, the tutorials and this changelog. Build it locally with
  `make -C docs html` after `pip install -e ".[docs]"` (new `docs` extra)
- `nba.orbits.write_orbit`, `read_orbit` and `provenance`: self-describing ECSV orbit files with units,
  per-snapshot diagnostics, every parameter as used, the warnings and the provenance (nba version and
  commit; the user, host and command only with `full_provenance=True`). `orbit(..., outfile=...)`
  writes one per method
- `iter_orbit` checks: warnings when the center jumps (`jump_factor`), moves inconsistently with its
  velocity for several snapshots (`velocity_tol`, `velocity_window`; with `nvel` or `rvel_factor`),
  or its final sphere falls below a fraction of its initial density (`min_density_ratio`); also when
  the time decreases or no softening is given. `warning_log` records the warnings
- `iter_orbit` options: `min_npart`, `npart_frac`, `nvel`, `rvel_factor`, `time_offset` (restarts that
  reset the time) and `return_info` (`info` gains `velocity_error` and `density_ratio`)
- Shrinking sphere: `npart_frac` (None removes the 1% cap on `min_npart`), `rvel_factor` (velocity from
  a multiple of the final radius), and `nmin` and `density` in `info`
- `CenterHalo.min_potential(return_info=True)`, and a warning when fewer than 10 particles are averaged
- `CenterHalo.mean_pos`: `rmax=None` (the new default) means no upper limit, so `rmin` alone can be
  given; masses are optional
- `ReadGC21.read_halo(seed=...)`; `ReadGC21.npart_mw` as a class attribute
- `nba.com` exports `ssphere_numba`
- `tests/test_centering_lmc.py`: tests on a real LMC subsample (MWLMC5_b0, snapshot 150), skipped unless
  the particle file is found (`NBA_TEST_DATA`, see `tests/data/README.md`)

### Changed
- `ReadGC21.read_halo` splits the MW and LMC with an O(N) ID threshold instead of sorting all IDs
- `tutorials/Reading_GC21_MWLMC_snapshots.ipynb` and `tutorials/lmc_centering.ipynb` were rewritten for
  the current API and are committed with their outputs, since the documentation shows them without
  running them (see `tutorials/README.md`)
- numpydoc docstrings in `nba.ios` and `nba.com`
- `pyyaml` is a dependency (ECSV orbit files)
- Committed files no longer contain local paths, user or host names

### Fixed
- `ReadGC21.read_halo(randomsample=n)` returns exactly n particles (it drew with replacement)
- The shrinking sphere's `npart` uses r < R, as the loop does, when it stops before shrinking
- `Reading_GC21_MWLMC_snapshots.ipynb` centered the MW on its first disk particle
  (`min_potential(disk_pot=True)`) and read the LMC from the isolated MW simulation

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

