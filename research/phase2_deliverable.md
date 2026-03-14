# Phase 2 Deliverable (Chapter 5.1)

This deliverable implements a modular pipeline that transforms raw order-book ticks into a filtered latent logit series suitable for EM calibration in Phase 3.

## What Is Implemented

1. Data conditioning:
   - canonical probability from bid/ask microstructure
   - clipping to `p in [epsilon, 1-epsilon]`
   - isolated spike removal
   - uniform cadence resampling with LOCF for missing bins

2. Observation model:
   - `y_t = logit(p_tilde_t) = x_t + eta_t`
   - heteroskedastic microstructure noise:
     `sigma_eta^2(t) = a0 + a1*s_t^2 + a2*d_t^{-1} + a3*r_t + a4*iota_t^2`
   - robust coefficient fitting via Huber-weighted ridge least squares

3. State filtering:
   - 1D Kalman filter (local-level state)
   - 1D RTS smoother to recover latent `x_hat_t`

4. Diagnostics:
   - innovation mean/std and median absolute innovation
   - Ljung-Box whiteness test

## New Modules

- `backend/calibration/types.py`
- `backend/calibration/preprocessing.py`
- `backend/calibration/filtering.py`
- `backend/calibration/pipeline.py`
- `backend/phase2.py`

## Runtime Fixes Included

- `backend/main.py`: fetchers are now imported lazily, so missing fetcher files do not crash module import.
- `backend/database/interface.py`: optional `psycopg2` import with explicit runtime error only when PostgreSQL mode is used.

## How To Run

From workspace root:

`python -m backend.phase2 --csv data/ticks.csv --out artifacts/phase2`

Or load from PostgreSQL:

`python -m backend.phase2 --token-id <TOKEN_ID> --since 2026-03-01T00:00:00+00:00 --out artifacts/phase2`

## Output Artifacts

- `artifacts/phase2/phase2_conditioned.csv`
- `artifacts/phase2/phase2_filtered.csv`
- `artifacts/phase2/phase2_summary.json`

These outputs are the direct input to Phase 3 (EM diffusion/jump separation and RN drift enforcement).
