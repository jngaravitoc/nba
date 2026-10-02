# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Test suite (`tests/`) and flake8 configuration
- `nba.orbits.orbit` rewritten on top of `ReadGC21` and `CenterHalo`
- `nba.visuals.mollweide` (HEALPix density maps and Mollweide plots)
- `__version__` in `nba/__init__.py`
- `.gitignore`
### Fixed
- `nba.kinematics` (duplicate argument in `slice_NN`) and `nba.cosmology`
  (class was commented out) can be imported again
- Packaging: dynamic version in `pyproject.toml`; numba and scikit-learn are core dependencies
- CI installs the package, runs on `devel` and pull requests, and no longer fails without tests
- Tutorial scripts and notebooks updated to the current API
### Removed
- Non-working modules moved to `legacy/` (not installed)
- Cell outputs stripped from tutorial notebooks

## [1.2] - 2025-12-22
### Added
- Included functionality to read Sheng+24 data
- Added 3D grid generation functionality as part of utils
### Fixed:
- Consistent binning strategy in Profile for enclosed mass, potential, and
  density 

## [1.1] - 2025-05-31
### Added 
- Included test units for the main code functionality
- Included basic documentation
- Move out analysis files out of the repo
### Fixed
- Issue #10

## [1.0.1] - 2025-05-27
### Added 

- Changelog file
- Including scf_utils scipt into __init__.py
### Fixed
- Issue #27 


## [0.0.0] - 2024-04-26
### Initial Release
- First stable release of the library.

