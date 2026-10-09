# Test data

## LMC centering test (`tests/test_centering_lmc.py`)

| File | In the repository | Content |
|---|---|---|
| `MWLMC5_b0_snap150_lmc_centers.json` (9 kB) | yes | Reference centres, velocities and `info` dicts from the shrinking sphere and `mean_pos`, for five settings, on the subsample and on all 15M LMC particles; and the full-data orbit around snapshot 150. Made with nba `8c1c3a1` (`meta.provenance`); `meta.updates` lists later changes. |
| `MWLMC5_b0_snap150_lmc_subsample.npz` (42 MB) | **no** | 1.5M of the 15M LMC dark matter particles of GC21 MWLMC5_b0 at snapshot 150 (t = 2.93 Gyr), drawn without replacement with `np.random.default_rng(150)`, in simulation coordinates. MD5 `3b21a2d2b0cb80c7fd731c8e47fab4c4`. |

The particle file is too large for the repository and is ignored by git. To run the tests, put it in this folder,
or point the `NBA_TEST_DATA` environment variable to the folder that holds it:

```
NBA_TEST_DATA=/path/to/folder pytest tests/test_centering_lmc.py
```

Without it the 30 tests are skipped. With it they take about 30 s, most of it in the NumPy shrinking sphere.

**Regenerating.** Both files are made by `scripts/make_centering_test_data.py` in the `nba_centering`
project, which needs the MWLMC5_b0 snapshots. The particle file is reproducible from the seed and the
selection rule (`ParticleIDs >= 107180001`). The reference file is recomputed with the installed nba, so after
an intended change in the results, regenerate it and commit it.
