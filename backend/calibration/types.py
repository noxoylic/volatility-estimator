from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class RawTick:
    timestamp: datetime
    token_id: str
    best_bid: float
    best_ask: float
    depth_bid: float
    depth_ask: float
    spread: float
    trade_size: float = 0.0
    trade_rate: float = 0.0
    imbalance: float = 0.0


@dataclass
class ConditionedPoint:
    timestamp: datetime
    token_id: str
    canonical_p: float
    logit_y: float
    spread: float
    depth: float
    trade_rate: float
    imbalance: float


@dataclass
class FilterPoint:
    timestamp: datetime
    token_id: str
    observed_logit: float
    filtered_logit: float
    smoothed_logit: float
    filtered_p: float
    smoothed_p: float
    innovation: float
    innovation_var: float
    measurement_var: float
    process_var: float


@dataclass
class Diagnostics:
    ljung_box_q: float
    ljung_box_pvalue: float
    innovation_mean: float
    innovation_std: float
    abs_innovation_median: float
    pass_whiteness: bool


@dataclass
class Phase2Result:
    conditioned: List[ConditionedPoint]
    filtered: List[FilterPoint]
    diagnostics: Diagnostics
    noise_coefficients: List[float]
