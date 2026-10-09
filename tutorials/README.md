# Tutorials

| File | What it shows | Data |
|---|---|---|
| `Reading_GC21_MWLMC_snapshots.ipynb` | Reading a GC21 snapshot, selecting the MW and the LMC, centering them | GC21 MWLMC5_b0 snapshot 110 |
| `lmc_centering.ipynb` | Following the LMC through MWLMC5_b0 with `nba.orbits.orbit`, and when its center is reliable | GC21 MWLMC5_b0, all 451 snapshots |
| `density_profiles.ipynb`, `disk_particles.ipynb`, `example_cosmology_functions.ipynb` | Profiles, disk particles, cosmology functions | GC21 snapshots |
| `compute_lmc_orbits.py`, `compute_orbit.py`, `compute_orbit_parallel.py`, `halo_kinematics.py`, `orbital_poles_computation.py` | Scripts for orbits and kinematics | GC21 snapshots |

## Notebooks in the documentation

The first two notebooks are shown in the documentation (`docs/tutorials.rst`). The simulations are not public,
so the documentation build does not run them: they are committed **with their outputs**, unlike the other
notebooks. After changing them, or a change in nba that affects their results, run them again and commit
them:

```
export NBA_GC21_DIR=/path/to/GC21/MWLMC5_b0/out
export NBA_ORBIT_DIR=/path/to/lmc_orbits    # where lmc_centering.ipynb writes its orbit files
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    Reading_GC21_MWLMC_snapshots.ipynb lmc_centering.ipynb
```

This needs the `docs` extra (`pip install -e ".[docs]"`), about 7 GB of memory, and about 1.5 hours the first
time, while `lmc_centering.ipynb` computes the orbits; once they exist, the orbit files are read instead.
Before committing, check that the outputs show no local paths, user or host names, and that each notebook is
a few MB at most.
