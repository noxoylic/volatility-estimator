from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

from backend.calibration.filtering import FilterConfig, run_phase2_filter
from backend.calibration.preprocessing import PreprocessConfig, condition_ticks, ticks_from_dict_rows
from backend.calibration.types import Phase2Result


def _read_csv_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_rows_from_storage(
    *,
    csv_path: str | None = None,
    token_id: str | None = None,
    since_iso: str | None = None,
    limit: int = 50000,
) -> List[dict]:
    if csv_path:
        return _read_csv_rows(Path(csv_path))

    if not token_id:
        raise ValueError("token_id is required when csv_path is not provided")

    from backend.database.interface import DatabaseInterface

    db = DatabaseInterface()
    db.connect()
    try:
        since = datetime.fromisoformat(since_iso) if since_iso else None
        return db.get_ticks(token_id=token_id, since=since, limit=limit)
    finally:
        db.close()


def run_phase2_pipeline(
    rows: Sequence[dict],
    preprocess_cfg: PreprocessConfig | None = None,
    filter_cfg: FilterConfig | None = None,
) -> Phase2Result:
    pp_cfg = preprocess_cfg or PreprocessConfig()
    flt_cfg = filter_cfg or FilterConfig()

    raw_ticks = ticks_from_dict_rows(rows)
    conditioned = condition_ticks(raw_ticks, pp_cfg)
    filtered_rows, diagnostics, beta = run_phase2_filter(conditioned, flt_cfg)

    return Phase2Result(
        conditioned=conditioned,
        filtered=filtered_rows,
        diagnostics=diagnostics,
        noise_coefficients=beta,
    )


def write_phase2_artifacts(result: Phase2Result, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    conditioned_path = out / "phase2_conditioned.csv"
    with conditioned_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp",
            "token_id",
            "canonical_p",
            "logit_y",
            "spread",
            "depth",
            "trade_rate",
            "imbalance",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.conditioned:
            payload = asdict(row)
            payload["timestamp"] = row.timestamp.isoformat()
            writer.writerow(payload)

    filtered_path = out / "phase2_filtered.csv"
    with filtered_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp",
            "token_id",
            "observed_logit",
            "filtered_logit",
            "smoothed_logit",
            "filtered_p",
            "smoothed_p",
            "innovation",
            "innovation_var",
            "measurement_var",
            "process_var",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.filtered:
            payload = asdict(row)
            payload["timestamp"] = row.timestamp.isoformat()
            writer.writerow(payload)

    summary_path = out / "phase2_summary.json"
    summary = {
        "counts": {
            "conditioned": len(result.conditioned),
            "filtered": len(result.filtered),
        },
        "noise_coefficients": result.noise_coefficients,
        "diagnostics": asdict(result.diagnostics),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
