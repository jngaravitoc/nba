# COM methods for a disrupting satellite: assessment and suggestions

Test case: LMC in GC21 **MWLMC5_b0** (15M LMC DM particles, 451 snapshots, 0–8.80 Gyr),
nba 0.3.0 at commit `3ff4881` (local checkout, installed editable).
All methods were run with the tutorial defaults (`compute_lmc_orbits.py`): `rcut_pot=2`,
`rcut_vel=20`, `min_npart=1000`, no `delta`, `r0`, `center0`, `softening` or `nvel`.

Evidence files (all in `data/`):

| File | Content |
|---|---|
| `MWLMC5_b0_lmc_orbit_<method>.txt` | orbits, `t x y z vx vy vz` (Gyr, kpc, km/s) |
| `MWLMC5_b0_lmc_diagnostics.txt` | per snapshot: shrinking-sphere final radius / npart / niter / stop reason / particles used for the velocity; `min_potential` anchor particle and number averaged; timings |
| `MWLMC5_b0_lmc_orbit_distance_velocity.png` | distance and speed vs time, all methods |
| `MWLMC5_b0_lmc_orbit_offsets.png` | \|Δr\|, \|Δv\| of each method from `shrinking_sphere_numba` (log) |
| `MWLMC5_b0_lmc_density_yz_{wide,zoom}_*.png` | y–z density maps at snaps 0/150/300/450, centred on each method's COM (±100 kpc and ±10 kpc) |
| `MWLMC5_b0_lmc_density_peak_offsets.txt` | COM → smoothed projected density peak offsets |

Notes on the evidence:
- The density peak is the maximum of the smoothed y–z map. 0.07 kpc (zoom) and 0.35 kpc
  (wide) are one bin, so those values mean the peak is at the COM.
- A zoom offset component of ±9.95 means the peak is at or beyond the ±10 kpc edge.

## 1. How each method does on this run

### `shrinking_sphere` / `shrinking_sphere_numba`: the only usable method here, until the core dissolves

**The two versions agree exactly.** Over all 451 snapshots the difference is
max |Δpos| = max |Δv| = 0, with identical final radius, `npart` and `niter`. Both also match
`nba.orbits.iter_orbit` exactly. These are good signs for the new version.

**Until about 6 Gyr the method works well.**
- The orbit is smooth, with first pericentre at 2.23 Gyr and 24.9 kpc.
- At 0, 2.93 and 5.87 Gyr the COM is on the projected density peak to within one bin
  (0.07 kpc).
- Every snapshot stopped on `min_npart`, with about 1000 particles.

**The core then loses density.** The final radius holding 1000 particles grows from
0.23 kpc (t = 0) to 0.5 kpc (5.9 Gyr) and to 2.1 kpc (7.8 Gyr). The central density falls by
roughly 600×, so the core stops being a strong maximum.

**After about 6 Gyr the centre jumps.** It moves 30–65 kpc in a single snapshot, where the
typical step is 2.6 kpc:

| snap | t [Gyr] | jump [kpc] | \|v\| before → after [km/s] |
|---|---|---|---|
| 309 | 6.04 | 59.9 | 127 → 238 |
| 312 | 6.10 | 36.2 | 237 → 216 |
| 329 | 6.43 | 44.0 | 111 → 239 |
| 359 | 7.02 | 64.7 | 206 → 204 |
| 387 / 388 | 7.57 / 7.59 | 55.5 / 32.5 | 99 → 192 → 236 |
| 426 / 428 | 8.33 / 8.37 | 40.6 / 45.7 | 115 → 200 / 206 |

- Several jumps land within 14–18 kpc of the MW centre. The sphere is switching between the
  remnant and debris piled up in the MW centre, which are now of similar density.
- At 8.80 Gyr the zoom map has no core left, and the wide map shows a phase-mixed spiral.
  From here on, "the LMC centre" is poorly defined for any method. The jumps are the method
  choosing a different local maximum each snapshot, not numerical noise.

**Problem: the velocity comes from a fixed 20 kpc sphere.**
- At t = 0 this is 3.7M particles. The position comes from about 1000 particles within
  0.23 kpc, so the velocity is averaged over a region 80× larger than the one that defines
  the centre.
- Late in the run it is about 250k particles, mostly debris. The speed jumps of about
  100 km/s at the position jumps come from this sphere moving onto a different stream.

**Cost:** the NumPy version takes about 218 s per snapshot on 15M particles. The Numba
version takes 14–19 s, about 13× faster for identical results.

### `mean_pos`: unsuitable after the first pericentre approach

- **It is only right at t = 0**, where it is 0.3 kpc from the shrinking-sphere centre.
- **By 1 Gyr it is 6.4 kpc off, and by 2 Gyr 48 kpc**, as stripped mass builds up.
- **From 2.9 Gyr on it sits in low-density space.** It is 66, 126 and 52 kpc from the
  density peak at snaps 150, 300 and 450.
- **Its orbit is smooth but wrong:** after about 2.6 Gyr it stays at 10–35 kpc, below
  80 km/s. That is the debris centroid sinking into the MW, not the LMC.
- **Use it only as a starting guess or for the t = 0 check.**

### `min_potential`: follows the MW centre, not the LMC, from t = 0

**Cause:**
- The snapshot `Potential` field is the total MW+LMC potential.
- The GC21 LMC initial halo is very extended. At t = 0, 36 LMC particles are already within
  10 kpc of the MW centre, and 3,839 within 50 kpc.
- Those particles have Φ ≈ −235,000 (km/s)², against about −61,000 in the LMC core, so the
  lowest-potential "LMC" particle is always one sitting in the MW's well.

**Evidence:**
- At t = 0 the anchor particle is at (4.5, 3.5, −2.8) kpc, 269 kpc from the LMC.
- At 8.80 Gyr it is 0.4 kpc from the MW's own potential minimum.
- In the orbit plot its distance rises smoothly from 5 to 100 kpc. This is the MW centre
  drifting in the simulation frame.
- In the density maps the COM sits 28–140 kpc from the LMC peak at all four snapshots, and
  the t = 0 zoom map is almost empty.

**Its velocity is noise:**
- With `rcut=2` it averages between 1 particle (t = 0) and 257 particles.
- Its speed ranges from 15 to 507 km/s, and the median snapshot-to-snapshot change is
  39 km/s.

**This is not an implementation bug.** It is an assumption that doesn't hold: "the
global potential minimum of the selected particles is the centre of this halo" is true for
the host, not for a satellite in a total-potential snapshot. The ID-based selection was
checked separately and is clean: one particle mass, contiguous IDs ≥ 107180001.

## 2. Bugs, edge cases and inconsistencies in `com_methods.py`

Each item was checked by running the code (synthetic arrays or this run), not just read.

1. **The two shrinking-sphere methods take their parameters in a different order**
   ([com_methods.py:402](../../../codes/nba/nba/com/com_methods.py#L402) vs
   [:412](../../../codes/nba/nba/com/com_methods.py#L412)).
   - `shrinking_sphere(delta, min_npart, rcut_vel, ...)` vs
     `shrinking_sphere_numba(delta, rcut_vel, min_npart, ...)`, and `ssphere_numba` follows
     the Numba order.
   - So the call `(None, 500)` sets `min_npart=500` in one and `rcut_vel=500` in the other,
     without any warning.
   - Fix: make everything after `delta` keyword-only (`*,`).
2. **`min_npart` cannot be raised above 1% of the particles**: `nmin = max(1, min(min_npart, int(0.01*N)))`.
   - `min_npart=10**9` on N = 20000 still stops at 200 particles.
   - This follows Power et al., but it means a larger, less noisy final sphere can't be
     requested for a dissolving core (Section 3.4).
   - The docstring says so, but nothing is returned or warned when the value is clipped.
3. **`min_potential` has no locality and no diagnostics.**
   - Both modes, `rcut` around the global argmin and the `npart` lowest-potential particles,
     work on the global potential ordering, so both fail as in Section 1.
   - There is no `center0` or search radius.
   - It doesn't support `return_info`, so the number of particles averaged (here sometimes
     1) can't be seen without recomputing it.
   - With `rcut` the result depends on one particle, so it jumps from snapshot to snapshot.
4. **`mean_pos` interface:**
   - `rmin > 0` with `rmax = 0` raises "rmin must be less than or equal to rmax", so there is
     no way to ask for "everything beyond rmin".
   - It demands `mass` (`_require("vel","mass")`) even though `_weighted_mean` handles
     `weights=None`, which `min_potential` and `velocities_com` both allow.
   - It has no `center0`, so it can't be limited to a region around a previous centre.
5. **Empty selections are handled inconsistently.**
   - `velocities_com` raises `ValueError` when no particles are within `r_cut`.
   - The shrinking sphere's `_com_velocity` silently returns NaN.
   - `min_potential` with an empty `rcut` selection can't happen, since the argmin particle
     is always selected, but would return NaN through `np.mean`.
   - Pick one behaviour and document it.
6. **`npart` uses a different radius test on an immediate stop.** When the sphere stops
   before shrinking (`niter = 0`), `npart` comes from `searchsorted(..., side="right")`
   (r ≤ R), but the loop tests r < R. This is cosmetic, but the `info` count doesn't use the
   same definition.
7. **No softening floor by default.** The final radius is 0.23 kpc at t = 0, below
   4ε = 0.32 kpc (DM softening 0.08 kpc), so the default stops inside the softened core,
   which is the regime the docstring warns about. `iter_orbit` accepts `softening` but
   defaults to None, and so does the tutorial.
8. **The NumPy shrinking sphere is about 13× slower than Numba for identical results.**
   - Each step boolean-indexes and copies the leading block of the sorted arrays.
   - With radius × 0.975 per step, the first ~100 steps each read most of the 15M particles.
   - Either use the Numba kernel only and keep NumPy as a test reference, or crop the sorted
     arrays permanently once the radius has shrunk.

Outside `com_methods.py`, found during the run (also not changed):

- **`nba/ios/snap_reader.py` changes the caller's logging setup.** It calls
  `logging.basicConfig(...)` on import, which overrides any logging the caller sets up, and
  `read_header` logs every header field at INFO.
- **`ReadGC21.read_halo(randomsample=n)` returns fewer than n particles.** It draws with
  `np.random.randint` (with replacement), so duplicate draws collapse in the boolean mask.
  Use `rng.choice(npart, n, replace=False)`.
- **`ReadGC21.read_halo` sorts all 115M DM IDs** and runs `np.isin` against 15M IDs for
  every snapshot. A threshold on the sorted ID array (`pid >= cut`) is equivalent for GC21
  and much cheaper.
- **`iter_orbit` returns the raw header time.** In MWLMC5_b0, snaps 400–450 come from a
  restart and their header time resets to 0, so orbits built with `iter_orbit`, or with the
  tutorial, go back to t = 0 at snap 400. This was corrected here with +8.0 code units. A
  `time_offset` hook, or a warning when time decreases, would catch it.
- **`iter_orbit` passes no `center0` or locality to `min_potential`**, so the tracking
  option `r0` only helps the shrinking sphere.

## 3. Proposed improvements

Rough order of expected benefit for satellites. The trade-off is noted for each.

### 3.1 Track with the previous centre (`center0` + `r0`): available now, run it next
`iter_orbit(..., r0=R)` already starts each shrinking sphere from a sphere of radius R
around the previous centre.

- **Expected benefit:** with R ≈ 10–20 kpc, larger than the 2.6 kpc typical step and the
  ~7 kpc maximum at pericentre, the 30–65 kpc jumps after 6 Gyr should go away, because
  debris in the MW centre is never inside the starting sphere.
- **Trade-off:** if the true centre ever moves more than R in one snapshot, or tracking is
  lost, the method stays on the wrong object with no way to recover. Pair it with a
  consistency check (3.5).
- **Suggested API change:** let `center0`/`r0` also drive `mean_pos` and `min_potential`.

### 3.2 Make `min_potential` local, or use the satellite's own potential
- **(a) Local version:** take the argmin of the potential only among particles within
  `r_search` of a prior centre, such as the shrinking-sphere centre or the previous snapshot.
  This is cheap and fixes the failure in Section 1, but it still uses the host-dominated
  total potential. Across a satellite of a few kpc the host's potential gradient shifts the
  minimum toward the host.
- **(b) Self-potential version:** compute the LMC-only potential of particles near the
  candidate centre, with a tree or direct sum over a subsample within ~20–50 kpc, and take
  its minimum. This is the right quantity for a satellite, at the cost of a potential
  calculation per snapshot.
- **In both versions:**
  - Average the `npart` lowest-potential particles (e.g. 100–1000) rather than one particle
    plus a fixed `rcut`.
  - Return the anchor and the count through `return_info`.

### 3.3 Centre on the bound remnant (iterative unbinding)
Starting from the shrinking-sphere centre:
1. Compute each particle's energy relative to the remnant: the remnant's self-potential plus
   ½|v − v_c|².
2. Remove the unbound particles.
3. Recompute the centre and velocity from the bound set, and repeat until it converges.

- **Expected benefit:**
  - Removes stream debris from both the position and the velocity.
  - Gives a bound mass M_bound(t), which shows when the satellite has really dissolved.
  - Gives an objective point to stop reporting a centre, which matters here after
    ~6–7 Gyr.
- **Trade-off:**
  - Costs far more than any current method; it needs a self-potential.
  - Needs care at pericentre, where tides make "bound" ambiguous.
  - A tidal-radius cut can replace infinity as the boundary.

### 3.4 Shrinking-sphere stopping and velocity choices
- **Velocity region:** set the velocity region from the final sphere, using
  `nvel ≈ 10–100 × npart` or a radius of a few × the final radius, instead of a fixed
  `rcut_vel = 20 kpc`.
  - Expected benefit: the velocity describes the same particles that define the centre.
    This removes the ~100 km/s velocity steps from debris streams in the 20 kpc sphere.
  - Trade-off: more Poisson noise in v from fewer particles. 10^4–10^5 particles still give
    sub-km/s noise for σ ~ 50–100 km/s.
- **Default softening:** default `softening` to the snapshot's softening, or require it in
  `iter_orbit`.
  - Expected benefit: no centre noise from inside the softened core.
  - Trade-off: the final sphere is slightly larger.
- **Configurable `min_npart` cap:** allow `min_npart` above 1% of N, for example with
  `cap_fraction=None`.
  - Expected benefit: as the core loses density, a larger final particle count gives a
    steadier centre.
  - Trade-off: departs from Power et al. and is biased if the core is asymmetric.
- **Return a centring-quality measure in `info`**, such as the final radius or the mean
  density inside it.
  - Expected benefit: a jump in the final radius (0.5 → 1.8 kpc at snap 309 here) flags the
    snapshots where the centre is ambiguous.

### 3.5 Consistency checks inside `iter_orbit`
- Warn when the centre moves more than k × |v| Δt between snapshots. Here a 60 kpc step
  with |v| ≈ 130 km/s and Δt ≈ 0.02 Gyr is about 20× the expected step.
- Warn when the header time decreases.
- **Trade-off:** none for the results; this is cheap bookkeeping, and the thresholds would
  need defaults.

### 3.6 Performance
- Use the Numba kernel as the default `shrinking_sphere` and keep NumPy as a test reference.
- Replace `np.isin` with an ID threshold in `ReadGC21`.
- The full run (451 snapshots × 4 methods) took 48 min on 2 nodes with 64 workers. About
  half of that was I/O contention and most of the rest was the NumPy shrinking sphere.
  Without the NumPy run, the same job would be I/O-bound at roughly half the time.

## 4. Recommended setup for the remaining GC21 runs

- `shrinking_sphere_numba` with `softening=0.08`, plus `iter_orbit(r0≈15)` tracking.
- A velocity from `nvel` particles instead of the 20 kpc sphere. `nvel` is supported by
  `CenterHalo` but not yet passed through by `iter_orbit`.
- Report the final radius per snapshot and stop trusting the centre when it grows
  sharply.
- Keep `mean_pos` only for the t = 0 check.
- Don't use `min_potential` for satellites until 3.2 is in place.

The orbit pipeline here already takes the simulation folder, name, snapshot pattern and
restart time offsets as arguments, so it can be pointed at the other GC21 runs without
changes. Each new run still needs a check for the restart/`bak-*` pattern.
