# COM methods: changes after the MWLMC5_b0 assessment

This note lists the changes made in response to `com_methods_suggestions.md`, what to re-test,
and what was left out. The changes are on the `devel` branch, on top of `3ff4881`.

## Summary

- **`min_potential` is now for host galaxies only.** `iter_orbit`/`orbit` refuse
  `min_potential` and `diskpot` for `halo='LMC'`. Use the shrinking sphere for satellites.
- **Bugs from Section 2 of the report are fixed**: argument order, empty selections, the `npart`
  definition, the `min_npart` cap, the `mean_pos` interface, and the reader's logging, sampling
  and ID selection.
- **New options for satellites**: a velocity region tied to the final sphere, a quality measure
  in `info`, and jump and time checks in `iter_orbit`.
- **Defaults are unchanged** for the velocity (`rcut_vel=20`) and the softening (`None`), so
  existing calls give the same centres. `iter_orbit` now warns when no softening is given.

## Changes that can break existing code

| Change | What to do |
|---|---|
| In `shrinking_sphere`, `shrinking_sphere_numba` and `ssphere_numba`, every parameter after `delta` is keyword-only. | Pass `min_npart=`, `rcut_vel=`, etc. by name. A positional call now raises `TypeError`. |
| An empty velocity region in the shrinking sphere raises `ValueError`. It used to return NaN. | Catch the error, or use `nvel`/`rvel_factor` so the region can't be empty. |
| `iter_orbit`/`orbit` raise `ValueError` for `min_potential` or `diskpot` with `halo='LMC'`. | Drop these methods from satellite runs. Calling `CenterHalo.min_potential` directly still works, but see below. |
| `mean_pos(rmax=...)` defaults to `None` (no upper limit) instead of `0`. | Nothing to do: `rmax=0` still means no limit. |
| `snap_reader` no longer calls `logging.basicConfig` on import. | Configure logging in your own script if you relied on the INFO messages. Header fields are now logged at DEBUG. |

## Changes by report item

### `com_methods.py`

| Report item | Change |
|---|---|
| 2.1 Argument order | Keyword-only parameters after `delta` (see above). |
| 2.2 `min_npart` cap | New `npart_frac=0.01`. `npart_frac=None` removes the 1% cap, so `min_npart` is used as given. `info["nmin"]` reports the value used. |
| 2.3 `min_potential` | It is documented and enforced as a host-only method rather than made local. New `return_info=True` returns `anchor` (the lowest-potential particle), `pot_min` and `npart` (the number averaged). It warns when it averages fewer than 10 particles. |
| 2.4 `mean_pos` | `rmin` can be given alone (everything beyond `rmin`). `mass` is optional; without it the mean is unweighted. The existing `center` argument sets the origin of the radii. |
| 2.5 Empty selections | All methods now raise `ValueError` on an empty selection. |
| 2.6 `npart` on an immediate stop | It counts r < R, the same test the loop uses. |
| 2.7 Softening | Default unchanged. `iter_orbit` warns when a shrinking-sphere method runs without `softening`. |
| 2.8 / 3.6 NumPy speed | Not optimised. The NumPy version is documented as the test reference, and `shrinking_sphere_numba` is the one to use for production runs. |
| 3.4 Velocity region | New `rvel_factor`: the velocity uses the particles within `rvel_factor` × the final sphere radius. `nvel` still selects a fixed number of nearest particles. Only one of the two can be given. |
| 3.4 Quality measure | `info["density"]` is the mean density within the final radius of the centre. Together with `info["radius"]`, it flags snapshots where the centre is ambiguous. |

### `halo_orbit.py` (`iter_orbit`, `orbit`)

| Report item | Change |
|---|---|
| 3.1 / Section 4 pass-through | New keyword arguments `min_npart`, `npart_frac`, `nvel` and `rvel_factor` are passed to the shrinking sphere. `orbit` accepts the same arguments. |
| 3.5 Jump check | `jump_factor=5.0` warns when a centre moves more than `jump_factor` × \|v\| × Δt between consecutive snapshots, with \|v\| the larger of the two speeds. This assumes velocity × time gives a position, as in Gadget code units. `jump_factor=None` turns it off. |
| 3.5 Time check, restarts | A warning when the time decreases. New `time_offset`, a constant or a function of the snapshot number, is added to the header time. For MWLMC5_b0: `time_offset=lambda k: 8.0 if k >= 400 else 0.0`. |
| Diagnostics | `return_info=True` makes each centre `(pos, vel, info)`. `info` is the method's own dict; for `mean_pos` it only holds `npart`. The default output is unchanged. |

### `snap_reader.py` (`ReadGC21.read_halo`)

| Report item | Change |
|---|---|
| Logging | No `logging.basicConfig` on import. Header fields are logged at DEBUG. |
| `randomsample` | It draws without replacement, so it returns exactly n particles (or all of them if n is larger). New `seed` argument. |
| ID selection | MW/LMC split by an ID threshold found with `np.partition`. This is O(N), with no sort of the 115M IDs and no `np.isin`. It gives the same selection for any set of unique IDs. `ReadGC21.npart_mw` is now a class attribute. |

### Tutorial

`tutorials/compute_lmc_orbits.py` no longer runs `min_potential`. It now defaults to
`softening=0.08` and `r0=15`, and has a `--rvel-factor` option.

## Not changed

- **Locality or self-potential for `min_potential` (3.2):** dropped, since the method is now
  host-only.
- **Iterative unbinding (3.3):** not implemented for now.
- **Defaults:** `rcut_vel=20` and `softening=None` were kept, so existing results are reproducible.
  Set the new options explicitly.

## What to test

1. **Re-run MWLMC5_b0** with the setup recommended in Section 4 of the report:
   ```python
   iter_orbit(path, snapname, snapshots, halo="LMC", com_method="shrinking_sphere_numba",
              softening=0.08, r0=15, rvel_factor=5,   # or nvel=...
              time_offset=lambda k: 8.0 if k >= 400 else 0.0, return_info=True)
   ```
   Check that:
   - the 30–65 kpc jumps after 6 Gyr are gone, or at least trigger the jump warning;
   - the ~100 km/s velocity steps are gone with `rvel_factor` or `nvel`;
   - `info["radius"]` and `info["density"]` show when the core dissolves, and agree with the
     snapshots where you stop trusting the centre;
   - the time runs continuously through snap 400.
2. **Check that nothing changed without the new options.** With the old arguments
   (`rcut_vel=20`, no `r0`, no softening), the shrinking-sphere orbits should match the
   previous `MWLMC5_b0_lmc_orbit_*.txt` files exactly.
3. **Reader:**
   - `read_halo(..., halo="LMC")` selects the same particles as before (15M, IDs ≥ 107180001);
   - each snapshot reads faster;
   - `randomsample=n` returns n distinct particles.
4. **Host centring:** `min_potential` and `diskpot` with `halo="MW"` still behave as before.
5. **Your own scripts:** look for positional arguments after `delta` in shrinking-sphere calls,
   and for code that expected NaN velocities.

The unit tests (`pytest tests`) pass: 88 passed, 1 skipped. They include 21 new tests for the
changes above.
