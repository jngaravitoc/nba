# Re-test of the nba centering changes on MWLMC5_b0

**Tested version:** nba `devel` at `8c1c3a1`, which includes `1ec6fc4` and `e39177f`.
**Compared against:** the baseline from `3ff4881`, saved in `data/baseline_3ff4881/`.
**Covers:** the five items in `nba/reports/com_methods_changes.md` ("What to test"), plus two extra
experiments: r0 sensitivity and a larger final sphere.
**Outputs:** `data/retest_8c1c3a1/`. **Scripts:** `scripts/` and `slurm/`. Every computation ran as a SLURM job.

## Summary

| Item | Result |
|---|---|
| 1. Tracked re-run | **Pass, with one limit.** Tracking (`r0=15`, `softening=0.08`) removes the jumps up to about 7 Gyr, and `rvel_factor` fixes the velocity. After about 7.1 Gyr the LMC has dissolved and no centre is physical, and the new jump check does not catch it (details below). |
| 2. Old arguments give the old results | **Pass.** Bit-identical, both through `CenterHalo` and through `iter_orbit`. |
| 3. Reader | **Pass.** Same particles in the same order. `randomsample` is fixed. Reads are about 25–30% faster when the file is cached. |
| 4. Host centring | **Pass.** MW `min_potential` and `diskpot` are bit-identical to `3ff4881`. |
| 5. Our scripts | **Pass.** Only keyword arguments are used, and nothing relies on NaN velocities. |
| Unit tests | **Pass.** `pytest`: 88 passed, 1 skipped. `scripts/check_api.py`: 25/25 API checks pass. |
| Proposal | Self-describing orbit files (`write_orbit`/`read_orbit`), with a working prototype in this project (Section 6). |
| New test | A real-data test of the shrinking sphere, ready to move into `nba/tests/` (Section 7). It passes on `8c1c3a1` (27 passed, 3 expected failures) and found one small inconsistency in `info`. |

**Recommended setup for the other GC21 satellites:**

```python
iter_orbit(..., halo="LMC", com_method="shrinking_sphere_numba",
           softening=0.08, r0=15, rvel_factor=5, time_offset=..., return_info=True)
```

Only trust the centre while the mean density in the final sphere (`info["density"]`) is at least about 1% of
its initial value (see 1c).

## 1. Tracked re-run (developers' call)

Runs (output folders in `data/retest_8c1c3a1/`):

| Label | Folder | Settings |
|---|---|---|
| B1 | `tracked_r15_0-450` | `lmc_orbits.py`, r0 = 15, softening 0.08, velocity sweep |
| B2 | `iter_orbit_rvf5_0-450` | `nba.orbits.iter_orbit` with the developers' exact call, `rvel_factor=5` |
| B3 | `tracked_r10_0-450` | r0 = 10 |
| B4 | `tracked_r30_0-450` | r0 = 30 |
| B5 | `tracked_r15_n10000_fracnone_0-450` | r0 = 15, `min_npart=1e4`, `npart_frac=None` |

**Library path checks (B2)**
- **Same centres through `iter_orbit`:** B2's positions equal B1's exactly (max |Δ| = 0).
- **`rvel_factor` velocity is correct:** it matches an independent calculation (the sweep's `rvf5` column) to
  5×10⁻⁵ km/s, the precision of the text files, and the `info["nvel"]` counts are identical.
- **`time_offset` works:** time runs continuously through the restart (7.799 → 7.822 → 7.845 Gyr at snaps
  399–401).
- **The time-decrease warning works:** run on snaps 395–405 without `time_offset`, `iter_orbit` warned once, at
  snap 400 ("the time decreases (7.9765625 -> 0.0); use time_offset…").
- **No warnings in the full B2 run:** no jump, time or softening warnings over all 451 snapshots.

### 1a. Jumps (`MWLMC5_b0_lmc_runs_comparison.png` and `.txt`)

| Run | Largest step, t < 6 Gyr | Largest step, t ≥ 6 Gyr | Steps > 10 kpc | Most recent ≤ 1 kpc agreement with r0 = 15 |
|---|---|---|---|---|
| baseline (no tracking) | 9.1 | **64.7** | 13 | – (differs from 6.04 Gyr on) |
| r0 = 15 | 9.1 | 13.4 | 2 | – |
| r0 = 10 | 9.1 | 11.5 | 1 | 7.26 Gyr |
| r0 = 30 | 9.1 | 40.2 | 6 | 6.87 Gyr |
| r0 = 15, min_npart = 10⁴ | 9.1 | 13.9 | 1 | 7.30 Gyr |

Steps are in kpc. The maximum step before 6 Gyr (9.1 kpc) is pericentre motion, not a jump.

**What these results show:**
- **Before 6 Gyr every run gives the same orbit**, to within 0.01 kpc (0.13 kpc with min_npart = 10⁴).
- **r0 = 10–15 removes the 30–65 kpc jumps.** The density maps at the old jump snapshots
  (`tracked_r15_0-450/*_jumps_*`) put the tracked centre on the density peak to within one bin at 6.04, 6.43
  and 7.02 Gyr. The baseline sat 57 and 92 kpc from it at 6.04 and 7.02 Gyr
  (`baseline_maps/*_jumps_*`).
- **r0 = 30 is too large.** Debris clumps fit inside the starting sphere, so 6 jumps remain.

### 1b. Velocity region (`tracked_r15_0-450/MWLMC5_b0_lmc_velsweep*`)

Each region's velocity is compared with dx/dt of the B1 orbit, with snapshots next to jumps left out:

| Region | Median \|v − dx/dt\|, t < 6.8 Gyr | 90th percentile | Particles |
|---|---|---|---|
| `rvel_factor=2` | 1.77 km/s | 3.4 | ~9×10³ |
| `rvel_factor=5` | 1.81 | 3.5 | ~6×10⁴ |
| `rvel_factor=10` | 1.90 | 4.4 | ~2×10⁵ |
| `nvel=1e4` | 1.82 | 3.4 | 10⁴ |
| `rcut_vel=20` (old default) | 4.38 | **32.3** | ~2×10⁶ |

- **The small regions all agree.** `rvel_factor` 2–5 and `nvel` = 10⁴ reach the same residual of about 1.8 km/s.
- **The old 20 kpc sphere fails at pericentres.** It is 10–130 km/s off at 2.1, 4.0, 5.3 and 6.3 Gyr. At the
  6.3 Gyr pericentre it gives 243 km/s, against dx/dt = 354 km/s and 353 km/s from `rvel_factor=5`. This is the
  ~100 km/s velocity step reported earlier, and it is gone with `rvel_factor` or `nvel`.
- **Recommendation: `rvel_factor=5`.** Its residual is as low as `rvel_factor=2` with about 7× more particles,
  so its Poisson noise is ≲0.5 km/s. The snapshot-to-snapshot change, about 7.5 km/s, is real acceleration and
  is the same for every region.

### 1c. `radius` and `density` show when the LMC dissolves, and the centre stops being physical

| t [Gyr] | 0 | 4.0 | 6.0 | 6.8 | 7.0 | 7.2 | 7.5 | 8.8 |
|---|---|---|---|---|---|---|---|---|
| final radius [kpc] | 0.33 | 0.33 | 0.49 | 1.05 | 1.25 | 1.46 | 2.08 | 2.19 |
| density relative to t = 0 | 1 | 0.68 | 0.13 | 0.013 | 0.0074 | 0.0048 | 0.0017 | 0.0014 |

- **The velocity check fails at the same time as the density drops.** From about 7.1 Gyr on, |v − dx/dt| is
  100–600 km/s for every velocity region (median about 230 km/s after 7.3 Gyr). The running 5-snapshot median
  first goes above 20 km/s at 7.06 Gyr for `rvf5` and at 7.12 Gyr for `nvel=1e4`.
- **There is no bound remnant left.** The centre no longer moves with its own particles, and the y–z maps at
  7.57 and 8.37 Gyr show only phase-mixed shells and caustics. The global density peak lies on a shell
  50–90 kpc from the centre.
- **The runs diverge at the same time.** Results with different r0 or min_npart separate after about
  6.9–7.3 Gyr.
- **The diagnostics flag this reliably.** `info["radius"]` and `info["density"]` do flag the breakdown, as
  intended. A threshold of about 1% of the initial density (radius ≳ 1.3 kpc here) matches the point where the
  centre stops being physical.

### Problem found: the jump check doesn't detect this failure
- `jump_factor=5` warns when a step exceeds 5 × |v| × Δt. With |v| ≈ 250 km/s and Δt ≈ 0.02 Gyr, that is about
  25 kpc per snapshot.
- After 7.1 Gyr the centre wanders in 1–10 kpc steps in directions that don't match v. Each step is small, but
  the motion is unphysical.
- So the full B2 run raised **no warnings** while the centre was meaningless for its last ~1.7 Gyr.

### Suggestions for the nba developers
1. **Add a velocity-consistency check** to `iter_orbit`: compare the step with the velocity's prediction,
   |Δx − ½(v_k + v_{k−1})Δt| / (|v|Δt), or |v − dx/dt|. Warn when it stays high over a few snapshots.
   - This catches drift as well as jumps.
   - It needs a velocity region tied to the centre (`rvel_factor`/`nvel`). With the 20 kpc sphere it fires at
     every pericentre.
2. **Add an optional density threshold**, `min_density_ratio`, that warns, or stops tracking, once
   `info["density"]` falls below that fraction of its value in the first snapshot. Here 0.01 is the point where
   the LMC is gone.

### Extra experiments
- **r0:** use 10–15 kpc. They agree to within 0.004 kpc until 7.26 Gyr. 30 kpc is too large and leaves 6 jumps.
  Below about 7 Gyr the result doesn't depend on r0 between 10 and 15 kpc.
- **`min_npart=1e4` with `npart_frac=None`:**
  - The option works: `info["nmin"]` = 10⁴, and the final radius is about 2× larger.
  - The centre agrees with the default to within 0.13 kpc before 6 Gyr.
  - It only slightly reduces late-time scatter: the median step after 6 Gyr is 2.49 vs 2.75 kpc, and steps over
    10 kpc go from 2 to 1.
  - It doesn't delay the breakdown, which happens at the same time. No benefit is worth the departure from Power
    et al. in this run.

## 2. Old arguments give the old results (`retest_8c1c3a1/regression/`)
- **Setup:** all 4 methods with the tutorial defaults at snaps 0, 150, 309 and 450.
- **Result:** compared with `baseline_3ff4881`, max |Δt| = |Δpos| = |Δvel| = 0 for `mean_pos`, both shrinking
  spheres and `min_potential`.
- **Through `iter_orbit`:** the same arguments through `nba.orbits.iter_orbit` also give Δ = 0. `min_potential`
  is left out of this comparison, since `iter_orbit` correctly refuses it for the LMC.
- **Warnings in this run were all correct:** missing softening, `min_potential` averaging only 1 particle, and
  the time decreasing at the restart (no offset given).

## 3. Reader (`retest_8c1c3a1/reader_host/compare.txt`)
- **Selection:**
  - LMC: 15,000,000 particles with IDs 107180001–122180000.
  - MW: 100,000,000 particles with IDs 1–107123907.
  - Both have the **same IDs in the same order** as `3ff4881` (SHA-1 of the ID arrays), at snaps 0 and 450.
- **`randomsample=10**6`:**
  - Now returns exactly 10⁶ distinct LMC particles, and the same seed gives the same sample.
  - `3ff4881` returned only about 967k distinct particles.
- **Speed:** a cached LMC read takes 2.1–2.7 s vs 3.0–3.8 s before, about 25–30% faster. Reading the file is
  now most of the cost, so the gain is smaller than "much faster".

## 4. Host centring
- **Setup:** `iter_orbit(halo="MW", com_method=["min_potential","diskpot"])` at snaps 0, 150, 300 and 450,
  with `8c1c3a1` and `3ff4881`.
- **Result:** positions and velocities are bit-identical. For example, at snap 450 `min_potential` gives
  (24.336, 60.751, −80.268) kpc.

## 5. Our scripts
- Every shrinking-sphere call uses keyword arguments. The only positional calls are the deliberate `TypeError`
  tests in `check_api.py`.
- No script expected NaN velocities.
- `lmc_orbits.py --check-iter-orbit` already leaves out `min_potential` for the LMC.

## 6. Proposal: self-describing orbit files (prototype)

**The problem, as seen in this re-test:**
- Our orbit files record only the method and the snapshot range.
- The six runs here use different `r0`, softening, `min_npart` and velocity regions, and nothing in their
  `.txt` files says which is which.
- For example, the tracked runs B1 and B3–B5 compute their orbit velocity from the old 20 kpc sphere, which
  you can only find out from the job script.
- The per-snapshot diagnostics (`radius`, `density`, particle counts), which Section 1c shows are needed to
  judge the centre, live in separate files that are easy to lose.
- The nba version is not recorded anywhere.

**Proposal:** add `nba.orbits.write_orbit` and `read_orbit`, and let `orbit()`/`iter_orbit` write files
through them. Then the tutorial script and external pipelines would all write the same format. This would be a
library function rather than a standalone script, because copies of a script drift.

### The prototype
It is in this project, not in nba:
- `scripts/orbit_io.py` holds `provenance()`, `write_orbit()` and `read_orbit()`.
- `scripts/export_orbit_ecsv.py` converts a finished run folder.
- `lmc_orbits.py` and `iter_orbit_run.py` now write `<sim>_run_params.json` at the start of a run, holding the
  provenance and every parameter as used, so an export records the run rather than the export. Runs started
  before that file existed are exported with their settings given explicitly and the commit taken from the job
  log.

**Format:** astropy ECSV. astropy is already an nba dependency. ECSV is plain text, with units on every column
and a YAML metadata block in the header, and `read_orbit` returns an `astropy.table.Table` with the units and
`.meta` restored.

**Columns:**
- `snap`, then the current order `t x y z vx vy vz`, so existing column indices stay valid after `snap`.
- `t_code` and `time_offset`. The offset is stored per snapshot because a callable `time_offset` cannot be
  saved.
- The method's `info` dict: `radius`, `npart`, `nmin`, `niter`, `stop`, `density`, `nvel`.

**Metadata:**
- `provenance`: nba version, path, commit, branch and dirty flag; Python and numpy versions; date, user, host,
  SLURM job ID and command.
- `simulation`: folder, pattern, snapshot range, restart offsets, units and the code-time-to-Gyr factor.
- `selection`: reader, halo, particle type and the ID rule.
- `method`.
- `parameters`: every centring parameter after defaults are applied, including a plain-language description of
  the velocity region.
- `warnings`: every warning `iter_orbit` raised, with its snapshot.
- `notes`.

Example: the header of `data/retest_8c1c3a1/tracked_r15_300-335/MWLMC5_b0_lmc_orbit_shrinking_sphere_numba.ecsv`
(41 header lines), exported after the run with the provenance taken from its job log:

```
# %ECSV 1.0
# ---
# datatype:
# - {name: snap, datatype: int64, description: snapshot number}
# - {name: t, unit: Gyr, datatype: float64, description: time}
# - {name: x, unit: kpc, datatype: float64, description: centre position}
#   ... y, z, vx, vy, vz (km / s), t_code, time_offset ...
# - {name: radius, unit: kpc, datatype: float64, description: final shrinking-sphere radius}
#   ... npart, nmin, niter, stop ...
# - {name: density, unit: 1e+10 solMass / kpc3, datatype: float64, description: mean density in the final sphere}
# - {name: nvel, datatype: int64, description: particles used for the velocity}
# meta: !!omap
# - {format: nba orbit file 0.1}
# - provenance: {nba_commit: 8c1c3a1, nba_path: /home/nicolas.garavito/codes/nba/nba, nba_version: 0.3.0,
#     slurm_job_id: '12144', source: job log lmc_orbit_tracked_12144.out}
# - simulation:
#     name: MWLMC5_b0
#     snapshot_dir: /data8/ngaravito/XMC-Atlas-sims/GC21/MWLMC5_b0/out
#     snapshot_pattern: MWLMC5_100M_b0_vir_OM3_G4_{:03d}.hdf5
#     snapshots: 300-335 (36)
#     time_offsets: ['from snap 400: +8 (code units)']
#     units: {length: kpc, mass: 1e10 Msun, time_code_to_Gyr: 0.9777923542981722, velocity: km/s}
#     dm_softening_kpc: 0.08
# - selection: {halo: LMC, ptype: dm, reader: nba.ios.ReadGC21.read_halo,
#     rule: dark matter particles with IDs above the npart_mw = 1e8 lowest (MW)}
# - {method: shrinking_sphere_numba}
# - parameters: {center0: centre found in the previous snapshot, delta: null, min_npart: 1000,
#     npart_frac: 0.01, nvel: null, r0: 15, rcut_vel: 20, rvel_factor: null, shrink_floor_kpc: 0.32,
#     softening: 0.08, velocity_region: particles within rcut_vel = 20 kpc of the centre, ...}
# - warnings: []
# schema: astropy-2.0
snap t x y z vx vy vz t_code time_offset radius npart nmin niter stop density nvel
300 5.86675413 30.9233777 116.501915 -99.8515962 5.09644376 -61.825238 -87.9042929 6.000000004306607 0.0 0.5144677 1059 1000 447 min_npart 0.002227424 407212
```

Usage:

```python
from orbit_io import read_orbit          # proposed: from nba.orbits import read_orbit
orb = read_orbit("..._shrinking_sphere_numba.ecsv")
orb["x"].unit                            # kpc
orb.meta["parameters"]["r0"]             # 15
orb.meta["provenance"]["nba_commit"]     # '8c1c3a1'
```

### What the prototype showed (for the nba version)
1. **Write the file during the run, not afterwards.**
   - The provenance has to describe the run, and an export made later would record the wrong commit.
   - The prototype needs a parameters file written at the start, or the commit from the job log. Inside
     `iter_orbit` this comes for free.
   - Writing during the run also gives exact code times. Rebuilding `t_code` from the 8-digit Gyr column gave
     6.000000004 instead of 6.0.
2. **Plain numpy needs one extra argument.**
   - ECSV writes its column-name line without a `#`.
   - `np.loadtxt(path, comments="#")` therefore fails on that line, and `np.genfromtxt(names=True)` takes its
     names from the first `# %ECSV` line.
   - Both work once the comment lines are skipped: `skiprows=n_comment + 1`, or `skip_header=n_comment` with
     `names=True`.
   - So nba would need to document this, or provide `read_orbit(..., as_array=True)`. Commenting the column-name
     line would make the file plain numpy text but break the ECSV standard. We would keep ECSV.
3. **A callable `time_offset` can't be saved.** Storing the offset applied to each snapshot solves this and also
   makes the restart correction visible in the file.
4. **The git commit is only available for git checkouts.** `provenance()` falls back to the version number and
   records `nba_dirty` so that uncommitted edits are visible. A pip-installed release would rely on the version
   number alone.
5. **Record defaults as values.** `parameters` stores every option after defaults are applied, including
   `rcut_vel` when `rvel_factor` or `nvel` is unset. A file stays interpretable even if a default changes in a
   later release.
6. **Keep the quality diagnostics with the orbit.** Writing `radius` and `density` as columns means the
   "trust the centre while density ≥ 1% of its initial value" test from Section 1c can be applied from the file
   alone. If the checks suggested in Section 1 are added, their warnings would be recorded in `warnings` too.
7. **Size is negligible:** 41 header lines next to 451 data rows, and one file instead of separate orbit and
   diagnostics files.

**Suggested API:**

```python
nba.orbits.write_orbit(path, snap, t, pos, vel, *, t_code=None, time_offset=None, info=None,
                       simulation=None, selection=None, method=None, parameters=None,
                       warnings=None, notes=None)
nba.orbits.read_orbit(path, as_array=False)
nba.orbits.orbit(..., outfile=None)   # writes one file per method when given
```

Here `iter_orbit` would collect its own parameters and warnings, so the caller only adds what nba can't know:
the simulation name, notes, and the restart description.

## 7. A real-data test for nba (`tests/test_centering_lmc.py`)

**What it is:** a pytest module that tests the shrinking sphere on real LMC particles instead of the synthetic
halos in `nba/tests/test_com.py`.
- It uses a 10% random subsample of the MWLMC5_b0 LMC at snapshot 150 (t = 2.93 Gyr). At that time the core is
  intact but surrounded by debris, so a correct centre and the plain mean position differ by 66 kpc.
- It runs in about 35 s and needs no simulation access.
- With nba `8c1c3a1`: **27 passed, 3 expected failures** (`xfail`, strict; see point 3 under the tests below).

### Files and paths
All paths are relative to `/home/nicolas.garavito/projects/nba_centering/`.

| Path | What it is | Size |
|---|---|---|
| `tests/test_centering_lmc.py` | The test module (13 test functions, 30 test cases) | 11 kB |
| `tests/data/MWLMC5_b0_snap150_lmc_subsample.npz` | Particles: `pos` [kpc], `vel` [km/s] (float32, simulation coordinates, not recentred), `pid` (uint32), `mass` (one float32 value, 1e10 Msun) and `meta` (JSON: selection, seed, time, units, provenance) | 42.0 MB |
| `tests/data/MWLMC5_b0_snap150_lmc_centers.json` | Reference results from nba `8c1c3a1` (described below) | 8.8 kB |
| `tests/README.md` | How the files were made and how to regenerate them | – |
| `scripts/make_centering_test_data.py` | Generator. It reads the snapshot with h5py, independently of nba, then computes the references with the installed nba | – |
| `slurm/make_test_data.sh` | Runs the generator; it needs the snapshot at `/data8/ngaravito/XMC-Atlas-sims/GC21/MWLMC5_b0/out/MWLMC5_100M_b0_vir_OM3_G4_150.hdf5` | – |
| `slurm/test_centering.sh` | Runs the tests on the cluster, and logs the nba commit and the checksums of the test and data files | – |

**How the subsample was made:**
- The LMC is the `PartType1` particles with `ParticleIDs >= 107180001`, i.e. above the 10⁸ lowest IDs (the MW).
  That gives 15,000,000 particles, all of one mass, 1.1996922×10⁻⁶ ×10¹⁰ Msun.
- From those, 1,500,000 are drawn with `np.random.default_rng(150).choice(15_000_000, 1_500_000, replace=False)`
  and kept in sorted order.

**What the reference file holds:**
- `meta.configs`: the 5 settings, as keyword arguments for `shrinking_sphere`/`shrinking_sphere_numba`:
  - `default`: `{}`, the tutorial defaults;
  - `recommended`: `softening=0.08`, `rvel_factor=5`;
  - `nvel`: `softening=0.08`, `nvel=10000`;
  - `uncapped`: `softening=0.08`, `rvel_factor=5`, `min_npart=10000`, `npart_frac=None`;
  - `tracked`: `recommended` plus `center0` (the full-data centre at snap 149) and `r0=15`.
- `meta.provenance`: the nba commit and the environment used.
- `subsample["<config>/<method>"]`: `pos`, `vel` and `info` for both methods, plus `mean_pos`.
- `full["<config>/shrinking_sphere_numba"]`: the same settings on all 15M particles.
- `orbit`: the full-data tracked orbit at snaps 149–151 and its finite-difference velocity at snap 150.

### Test functions
**Fixtures** (module scope):
- `ref` loads the JSON file.
- `halo` loads the subsample and expands `mass` to an array.
- `computed` runs every case once: all 5 settings with `shrinking_sphere_numba`, and `default` and `tracked` with
  the NumPy `shrinking_sphere`. The NumPy version takes about 20 s per setting on 1.5M particles, so it runs for
  two settings only.

| Test | What it checks | Tolerance (observed) |
|---|---|---|
| `test_subsample_file` | Shapes and dtypes, 1.5M distinct IDs, LMC IDs only, not recentred | exact |
| `test_reference_centres[config-method]` (7 cases) | **Regression:** `pos`, `vel` and `info` match the stored results | 1e-6 kpc, 1e-5 km/s; integer counts exact; `radius` 1e-9 and `density` 1e-6 relative |
| `test_numpy_and_numba_agree[config]` | NumPy and Numba give identical results | bit-identical |
| `test_ssphere_numba_function` | The module-level `ssphere_numba` equals `CenterHalo.shrinking_sphere_numba` | bit-identical |
| `test_info_consistent_with_particles[config]` | `density` = mass within `radius` / volume; `nvel` matches `rvel_factor`, `nvel` or `rcut_vel`; `npart ≥ nmin` | 1e-6 relative; `npart` within 1% of a recount |
| `test_info_npart_and_density_describe_the_same_sphere[config]` | **Expected failure (strict):** `density × V / m == npart` (point 3 below) | exact |
| `test_nmin` | `nmin = min(min_npart, 1% of N)`, and `npart_frac=None` removes the cap | exact |
| `test_softening_floor` | With softening, the sphere doesn't shrink below 4 × softening | exact |
| `test_subsample_centre_matches_full_data[config]` | **Physics:** 10% of the particles give the full-data centre | 0.3 kpc (≤ 0.071) |
| `test_subsample_velocity_matches_full_data[config]` | Velocity with `rvel_factor`/`nvel` matches the full data | 3 km/s (≤ 1.6) |
| `test_velocity_matches_orbit[config]` | Velocity matches the rate of change of the full-data orbit | 4 km/s (≤ 1.9) |
| `test_tracking_gives_the_same_centre` | `center0` + `r0=15` finds the same centre in fewer steps | 0.05 kpc (0.004) |
| `test_mean_pos_is_pulled_by_debris` | `mean_pos` matches its reference and lies far from the centre | > 30 kpc (66) |

**How the two kinds of test behave:**
- The regression tolerances only allow for a different summation order, so any change in the results fails. If
  the change was intended, regenerate the reference file.
- The physical tests survive implementation changes and catch a wrong centre.

**Inconsistency the test found** (point 3; documented by the strict expected failure):
- In `nba/com/com_methods.py`, `_ssphere_power` and `_ssphere_kernel` return `npart` as the number of particles
  in the final sphere around the **previous** centre, whose barycentre becomes the returned centre.
- `_shrinking_sphere` then computes `info["density"]` from the particles within `radius` of the **returned**
  centre.
- So the two describe slightly different spheres: 1042 vs 1041 particles here, and 10183 vs 10169 for
  `uncapped`.
- Suggested fix: compute both around the returned centre, or say in the docstring which sphere each describes.
- When it is fixed, the test reports XPASS as a failure, and the `xfail` marker should be removed.

### How to move it into nba
1. **Test module:** copy `tests/test_centering_lmc.py` to `nba/tests/`. It only needs `numpy`, `pytest` and
   `nba`, and finds its data through `Path(__file__).parent / "data"`.
   - Suggested change: let an environment variable override the data folder, e.g.
     `DATA = Path(os.environ.get("NBA_TEST_DATA", Path(__file__).parent / "data"))`.
   - The module already skips itself when the files are missing (`pytestmark = pytest.mark.skipif(...)`), so
     CI without the data still passes.
2. **Data:** the reference JSON (8.8 kB) can go in `nba/tests/data/` directly. The particle file (42 MB) is
   about 10× the size of nba's whole git history (4.4 MB), so I'd **not commit it** to the repository. Options:
   - **Git LFS:** transparent for people who clone the repo, but it needs LFS installed and uses LFS quota.
   - **A release asset or Zenodo record**, downloaded on first use and checked against its MD5
     (`3b21a2d2b0cb80c7fd731c8e47fab4c4`), e.g. with `pooch`. This keeps the repo small, but needs network
     access the first time.
   - **A smaller subsample.** 150k particles would be 4 MB. The centre would still be well defined at snap 150,
     but the physical tolerances would need re-measuring and the regression values regenerating.
3. **Packaging:** nothing changes, because `[tool.setuptools.packages.find]` includes only `nba*`, so the
   tests and their data stay out of the wheel.
4. **Speed:** 35 s in total, of which about 30 s is the NumPy version. If that's too slow for CI, mark the two
   NumPy cases with a custom `slow` marker, registered in `pyproject.toml` under `[tool.pytest.ini_options]`.
5. **Regenerating after an intended change:** run `scripts/make_centering_test_data.py` (on the cluster:
   `sbatch slurm/make_test_data.sh`) with the new nba, and commit the new reference file.
   - The particle file is reproducible from the seed and the selection rule, so it does not change.
   - The generator writes both files, though; if the particle file is hosted outside the repo, keep the hosted
     copy and its checksum.

## Notes
- **In this re-test (fixed):**
  - In `slurm/tracked_orbit.sh`, the final `[ "$SWEEP" = 1 ] && …` made the jobs exit with status 1 when
    `SWEEP=0`. SLURM marked B3–B5 FAILED although all their outputs were complete and merged. Fixed with
    `if … fi`.
  - `check_api.py` initially compared r² against R². That counts a boundary particle after rounding, while nba
    compares r against R, which is correct. Fixed in the test, not in nba.
- **Remaining limitations of nba:**
  - `iter_orbit` has no velocity-consistency or density-threshold check (suggested above).
  - Iterative unbinding was not implemented, as stated in the developers' report. After about 7 Gyr in
    MWLMC5_b0 there is nothing bound left to centre on, so unbinding would mainly give an objective end time,
    M_bound(t) → 0, rather than a better centre.
