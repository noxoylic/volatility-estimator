from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Iterable, List, Sequence, Tuple

from backend.calibration.types import ConditionedPoint, Diagnostics, FilterPoint, Phase2Result


@dataclass
class FilterConfig:
    min_noise_var: float = 1e-8
    max_noise_var: float = 5e-2
    huber_k: float = 1.5
    huber_iters: int = 6
    ridge: float = 1e-6
    process_floor: float = 1e-8
    process_window: int = 20
    ljung_box_lags: int = 10


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        div = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= div

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def _huber_weights(residuals: Sequence[float], k: float) -> List[float]:
    scale = max(median(abs(r) for r in residuals), 1e-9)
    cutoff = k * scale
    out = []
    for r in residuals:
        ar = abs(r)
        out.append(1.0 if ar <= cutoff else cutoff / ar)
    return out


def _build_features(points: Sequence[ConditionedPoint]) -> Tuple[List[List[float]], List[float]]:
    x_mat: List[List[float]] = []
    y_vec: List[float] = []

    if len(points) < 2:
        return x_mat, y_vec

    for i in range(1, len(points)):
        cur = points[i]
        prev = points[i - 1]

        dy = cur.logit_y - prev.logit_y
        target = dy * dy

        s2 = cur.spread * cur.spread
        inv_depth = 1.0 / max(cur.depth, 1e-9)
        rate = cur.trade_rate
        imb2 = cur.imbalance * cur.imbalance

        x_mat.append([1.0, s2, inv_depth, rate, imb2])
        y_vec.append(target)

    return x_mat, y_vec


def fit_noise_model(points: Sequence[ConditionedPoint], cfg: FilterConfig) -> List[float]:
    x_mat, y_vec = _build_features(points)
    if not x_mat:
        return [1e-5, 1.0, 1e-3, 1e-3, 1e-3]

    p = len(x_mat[0])
    w = [1.0] * len(x_mat)

    beta = [0.0] * p
    for _ in range(cfg.huber_iters):
        xtwx = [[0.0] * p for _ in range(p)]
        xtwy = [0.0] * p

        for row, y, wi in zip(x_mat, y_vec, w):
            for i in range(p):
                xtwy[i] += wi * row[i] * y
                for j in range(p):
                    xtwx[i][j] += wi * row[i] * row[j]

        for i in range(p):
            xtwx[i][i] += cfg.ridge

        beta = _solve_linear_system(xtwx, xtwy)
        residuals = [y - _dot(row, beta) for row, y in zip(x_mat, y_vec)]
        w = _huber_weights(residuals, cfg.huber_k)

    return beta


def estimate_measurement_variance(points: Sequence[ConditionedPoint], beta: Sequence[float], cfg: FilterConfig) -> List[float]:
    out: List[float] = []
    for pt in points:
        features = [
            1.0,
            pt.spread * pt.spread,
            1.0 / max(pt.depth, 1e-9),
            pt.trade_rate,
            pt.imbalance * pt.imbalance,
        ]
        v = _dot(features, beta)
        out.append(max(cfg.min_noise_var, min(cfg.max_noise_var, v)))
    return out


def estimate_process_variance(points: Sequence[ConditionedPoint], cfg: FilterConfig) -> List[float]:
    if len(points) <= 1:
        return [cfg.process_floor for _ in points]

    inc = [points[i].logit_y - points[i - 1].logit_y for i in range(1, len(points))]
    sq = [x * x for x in inc]

    out = [cfg.process_floor]
    for i in range(len(inc)):
        lo = max(0, i - cfg.process_window + 1)
        window = sq[lo: i + 1]
        v = sum(window) / max(1, len(window))
        out.append(max(cfg.process_floor, v))

    return out


def run_kalman_smoother(points: Sequence[ConditionedPoint], meas_var: Sequence[float], proc_var: Sequence[float]) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    n = len(points)
    if n == 0:
        return [], [], [], [], []

    y = [pt.logit_y for pt in points]

    x_filt = [0.0] * n
    p_filt = [0.0] * n
    x_pred = [0.0] * n
    p_pred = [0.0] * n
    innovation = [0.0] * n

    x_prev = y[0]
    p_prev = max(proc_var[0], 1e-8)

    for t in range(n):
        q = max(proc_var[t], 1e-12)
        r = max(meas_var[t], 1e-12)

        x_prior = x_prev
        p_prior = p_prev + q

        v = y[t] - x_prior
        s = p_prior + r
        k = p_prior / s

        x_post = x_prior + k * v
        p_post = (1.0 - k) * p_prior

        x_pred[t] = x_prior
        p_pred[t] = p_prior
        innovation[t] = v
        x_filt[t] = x_post
        p_filt[t] = p_post

        x_prev = x_post
        p_prev = max(p_post, 1e-12)

    x_smooth = x_filt[:]
    p_smooth = p_filt[:]

    for t in range(n - 2, -1, -1):
        denom = max(p_pred[t + 1], 1e-12)
        a = p_filt[t] / denom
        x_smooth[t] = x_filt[t] + a * (x_smooth[t + 1] - x_pred[t + 1])
        p_smooth[t] = p_filt[t] + a * a * (p_smooth[t + 1] - p_pred[t + 1])

    return x_filt, x_smooth, p_filt, p_smooth, innovation


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _chi2_sf_wilson_hilferty(x: float, k: int) -> float:
    if x <= 0.0:
        return 1.0
    if k <= 0:
        return 0.0
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return _norm_sf(z)


def ljung_box_test(series: Sequence[float], lags: int) -> Tuple[float, float]:
    n = len(series)
    if n < lags + 2:
        return 0.0, 1.0

    mean = sum(series) / n
    centered = [x - mean for x in series]
    var = sum(x * x for x in centered)
    if var <= 1e-16:
        return 0.0, 1.0

    q = 0.0
    for k in range(1, lags + 1):
        cov = 0.0
        for t in range(k, n):
            cov += centered[t] * centered[t - k]
        rho = cov / var
        q += (rho * rho) / (n - k)

    q *= n * (n + 2)
    pvalue = _chi2_sf_wilson_hilferty(q, lags)
    return q, pvalue


def run_phase2_filter(points: Sequence[ConditionedPoint], cfg: FilterConfig) -> Tuple[List[FilterPoint], Diagnostics, List[float]]:
    if not points:
        empty_diag = Diagnostics(0.0, 1.0, 0.0, 0.0, 0.0, True)
        return [], empty_diag, [0.0] * 5

    beta = fit_noise_model(points, cfg)
    meas_var = estimate_measurement_variance(points, beta, cfg)
    proc_var = estimate_process_variance(points, cfg)

    x_filt, x_smooth, _, _, innovation = run_kalman_smoother(points, meas_var, proc_var)

    rows: List[FilterPoint] = []
    for i, pt in enumerate(points):
        rows.append(
            FilterPoint(
                timestamp=pt.timestamp,
                token_id=pt.token_id,
                observed_logit=pt.logit_y,
                filtered_logit=x_filt[i],
                smoothed_logit=x_smooth[i],
                filtered_p=1.0 / (1.0 + math.exp(-x_filt[i])),
                smoothed_p=1.0 / (1.0 + math.exp(-x_smooth[i])),
                innovation=innovation[i],
                innovation_var=proc_var[i] + meas_var[i],
                measurement_var=meas_var[i],
                process_var=proc_var[i],
            )
        )

    innov_mean = sum(innovation) / len(innovation)
    centered = [x - innov_mean for x in innovation]
    innov_std = math.sqrt(max(sum(x * x for x in centered) / max(1, len(centered) - 1), 0.0))
    abs_med = median(abs(x) for x in innovation)
    lb_q, lb_p = ljung_box_test(innovation, cfg.ljung_box_lags)

    diag = Diagnostics(
        ljung_box_q=lb_q,
        ljung_box_pvalue=lb_p,
        innovation_mean=innov_mean,
        innovation_std=innov_std,
        abs_innovation_median=abs_med,
        pass_whiteness=lb_p > 0.05,
    )

    return rows, diag, list(beta)
