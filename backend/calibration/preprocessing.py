from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Tuple

from backend.calibration.types import ConditionedPoint, RawTick


@dataclass
class PreprocessConfig:
    epsilon: float = 1e-5
    cadence_seconds: float = 1.0
    spike_threshold: float = 0.08
    spike_revert_threshold: float = 0.01
    min_depth: float = 1e-9


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float, eps: float) -> float:
    pc = max(eps, min(1.0 - eps, p))
    return math.log(pc / (1.0 - pc))


def _coerce_timestamp(value) -> datetime:
    if isinstance(value, datetime):
        ts = value
    else:
        txt = str(value).replace("Z", "+00:00")
        ts = datetime.fromisoformat(txt)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def ticks_from_dict_rows(rows: Iterable[dict], token_id_fallback: str = "unknown") -> List[RawTick]:
    out: List[RawTick] = []
    for row in rows:
        bid = float(row.get("best_bid", 0.0) or 0.0)
        ask = float(row.get("best_ask", 0.0) or 0.0)
        spread = float(row.get("spread", ask - bid) or (ask - bid))
        depth_bid = float(row.get("depth_bid", 0.0) or 0.0)
        depth_ask = float(row.get("depth_ask", 0.0) or 0.0)
        out.append(
            RawTick(
                timestamp=_coerce_timestamp(row["timestamp"]),
                token_id=str(row.get("token_id", token_id_fallback)),
                best_bid=bid,
                best_ask=ask,
                spread=spread,
                depth_bid=depth_bid,
                depth_ask=depth_ask,
                trade_size=float(row.get("trade_size", 0.0) or 0.0),
                trade_rate=float(row.get("trade_rate", 0.0) or 0.0),
                imbalance=float(row.get("imbalance", 0.0) or 0.0),
            )
        )
    out.sort(key=lambda x: x.timestamp)
    return out


def compute_canonical_probability(tick: RawTick, eps: float, min_depth: float) -> float:
    # Skip invalid books. Chapter 5 recommends dropping crossed/locked states.
    if tick.best_bid <= 0.0 or tick.best_ask <= 0.0 or tick.best_bid >= tick.best_ask:
        return float("nan")

    book_mid = 0.5 * (tick.best_bid + tick.best_ask)

    # Inverse-spread reliability plus optional trade-size reinforcement.
    w_book = 1.0 / max(tick.spread, eps)
    w_trade = max(0.0, tick.trade_size)

    if w_trade > 0.0:
        # If explicit trade price is unavailable, use midpoint proxy with trade weight.
        p = (w_book * book_mid + w_trade * book_mid) / (w_book + w_trade)
    else:
        p = book_mid

    return max(eps, min(1.0 - eps, p))


def remove_isolated_spikes(points: List[ConditionedPoint], cfg: PreprocessConfig) -> List[ConditionedPoint]:
    if len(points) < 3:
        return points[:]

    keep = [True] * len(points)
    max_gap = timedelta(seconds=1.5 * cfg.cadence_seconds)

    for i in range(1, len(points) - 1):
        p_prev = points[i - 1].canonical_p
        p_now = points[i].canonical_p
        p_next = points[i + 1].canonical_p

        jump = abs(p_now - p_prev)
        revert = abs(p_next - p_prev)
        tgap = points[i + 1].timestamp - points[i].timestamp

        if jump > cfg.spike_threshold and revert < cfg.spike_revert_threshold and tgap <= max_gap:
            keep[i] = False

    return [pt for pt, k in zip(points, keep) if k]


def _bucket_start(ts: datetime, cadence_seconds: float) -> datetime:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = (ts - epoch).total_seconds()
    bucket = math.floor(seconds / cadence_seconds) * cadence_seconds
    return epoch + timedelta(seconds=bucket)


def resample_uniform(points: List[ConditionedPoint], cfg: PreprocessConfig) -> List[ConditionedPoint]:
    if not points:
        return []

    by_bucket: Dict[datetime, List[ConditionedPoint]] = defaultdict(list)
    for pt in points:
        by_bucket[_bucket_start(pt.timestamp, cfg.cadence_seconds)].append(pt)

    buckets = sorted(by_bucket.keys())
    start = buckets[0]
    end = buckets[-1]

    out: List[ConditionedPoint] = []
    last: ConditionedPoint | None = None
    cur = start

    while cur <= end:
        bucket_points = by_bucket.get(cur)
        if bucket_points:
            w_sum = 0.0
            p_sum = 0.0
            spread_sum = 0.0
            depth_sum = 0.0
            rate_sum = 0.0
            imb_sum = 0.0

            for bp in bucket_points:
                w = 1.0 / max(bp.spread, cfg.epsilon)
                w_sum += w
                p_sum += w * bp.canonical_p
                spread_sum += bp.spread
                depth_sum += bp.depth
                rate_sum += bp.trade_rate
                imb_sum += bp.imbalance

            p = p_sum / max(w_sum, cfg.epsilon)
            spread = spread_sum / len(bucket_points)
            depth = depth_sum / len(bucket_points)
            rate = rate_sum / len(bucket_points)
            imb = imb_sum / len(bucket_points)
            token_id = bucket_points[-1].token_id

            pt = ConditionedPoint(
                timestamp=cur,
                token_id=token_id,
                canonical_p=max(cfg.epsilon, min(1.0 - cfg.epsilon, p)),
                logit_y=_logit(p, cfg.epsilon),
                spread=spread,
                depth=max(depth, cfg.min_depth),
                trade_rate=max(rate, 0.0),
                imbalance=imb,
            )
            out.append(pt)
            last = pt
        elif last is not None:
            # LOCF to maintain a uniform state-space grid.
            out.append(
                ConditionedPoint(
                    timestamp=cur,
                    token_id=last.token_id,
                    canonical_p=last.canonical_p,
                    logit_y=last.logit_y,
                    spread=last.spread,
                    depth=last.depth,
                    trade_rate=last.trade_rate,
                    imbalance=last.imbalance,
                )
            )

        cur += timedelta(seconds=cfg.cadence_seconds)

    return out


def condition_ticks(raw_ticks: List[RawTick], cfg: PreprocessConfig) -> List[ConditionedPoint]:
    conditioned: List[ConditionedPoint] = []

    for tick in raw_ticks:
        p = compute_canonical_probability(tick, eps=cfg.epsilon, min_depth=cfg.min_depth)
        if math.isnan(p):
            continue

        depth = max(tick.depth_bid + tick.depth_ask, cfg.min_depth)
        conditioned.append(
            ConditionedPoint(
                timestamp=tick.timestamp,
                token_id=tick.token_id,
                canonical_p=p,
                logit_y=_logit(p, cfg.epsilon),
                spread=max(tick.spread, cfg.epsilon),
                depth=depth,
                trade_rate=max(tick.trade_rate, 0.0),
                imbalance=tick.imbalance,
            )
        )

    conditioned = remove_isolated_spikes(conditioned, cfg)
    return resample_uniform(conditioned, cfg)
