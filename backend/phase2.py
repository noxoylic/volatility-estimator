"""
Phase 2 deliverable runner:
- Data conditioning (canonical mid, clipping, cadence, outlier cleanup)
- Heteroskedastic noise model fit
- 1D Kalman filter + smoother for latent logit
- Diagnostics and artifact export

Examples:
    python -m backend.phase2 --csv data/ticks.csv --out artifacts/phase2
    python -m backend.phase2 --token-id <TOKEN_ID> --since 2026-03-01T00:00:00+00:00
"""

from __future__ import annotations

import argparse

from backend.calibration.filtering import FilterConfig
from backend.calibration.pipeline import (
    load_rows_from_storage,
    run_phase2_pipeline,
    write_phase2_artifacts,
)
from backend.calibration.preprocessing import PreprocessConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter-5 Phase-2 calibration")
    parser.add_argument("--csv", help="Path to ticks CSV (optional)")
    parser.add_argument("--token-id", help="Token id for DB load when --csv is not provided")
    parser.add_argument("--since", help="ISO timestamp lower-bound for DB fetch")
    parser.add_argument("--limit", type=int, default=50000, help="Max DB rows (default 50k)")
    parser.add_argument("--out", default="artifacts/phase2", help="Output artifact folder")

    parser.add_argument("--cadence", type=float, default=1.0, help="Uniform cadence in seconds")
    parser.add_argument("--epsilon", type=float, default=1e-5, help="Probability clip epsilon")
    parser.add_argument("--spike-threshold", type=float, default=0.08, help="Spike detection threshold")
    parser.add_argument("--spike-revert-threshold", type=float, default=0.01, help="Spike revert threshold")

    parser.add_argument("--min-noise", type=float, default=1e-8, help="Noise variance lower clip")
    parser.add_argument("--max-noise", type=float, default=5e-2, help="Noise variance upper clip")
    parser.add_argument("--lags", type=int, default=10, help="Ljung-Box lags")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_rows_from_storage(
        csv_path=args.csv,
        token_id=args.token_id,
        since_iso=args.since,
        limit=args.limit,
    )

    preprocess_cfg = PreprocessConfig(
        epsilon=args.epsilon,
        cadence_seconds=args.cadence,
        spike_threshold=args.spike_threshold,
        spike_revert_threshold=args.spike_revert_threshold,
    )
    filter_cfg = FilterConfig(
        min_noise_var=args.min_noise,
        max_noise_var=args.max_noise,
        ljung_box_lags=args.lags,
    )

    result = run_phase2_pipeline(rows, preprocess_cfg=preprocess_cfg, filter_cfg=filter_cfg)
    write_phase2_artifacts(result, args.out)

    print("Phase 2 complete")
    print(f"Conditioned points: {len(result.conditioned)}")
    print(f"Filtered points: {len(result.filtered)}")
    print(f"Ljung-Box Q: {result.diagnostics.ljung_box_q:.4f}")
    print(f"Ljung-Box p-value: {result.diagnostics.ljung_box_pvalue:.4f}")
    print(f"Whiteness pass: {result.diagnostics.pass_whiteness}")
    print(f"Artifacts written to: {args.out}")


if __name__ == "__main__":
    main()
