from dataclasses import dataclass
from typing import Optional

@dataclass
class MarketMetadata:
    """
    STATIC DATA: Fetched once per session.
    Represents the Event Contract details (Question, Slug, etc.).
    """
    token_id: str
    condition_id: str
    question: str
    slug: str
    description: Optional[str] = None

class Tick:
    """
    DYNAMIC DATA: The 'Atomic Unit' of the RN-JD Model.
    FLAT structure for zero-overhead conversion to Numpy arrays.
    """
    __slots__ = [
        'timestamp', 'token_id', 
        'best_bid', 'best_ask', 'spread', 
        'depth_bid', 'depth_ask',
        'canonical_p', 'logit_x', 'noise_var'
    ]

    def __init__(self, timestamp, token_id, best_bid, best_ask, depth_bid, depth_ask, 
                 spread, canonical_p, logit_x, noise_var):
        self.timestamp = timestamp
        self.token_id = token_id
        
        # Raw Data (Microstructure)
        self.best_bid = best_bid
        self.best_ask = best_ask
        self.depth_bid = depth_bid
        self.depth_ask = depth_ask
        self.spread = spread
        
        # RN-JD Model Inputs (Section 5.1 of Paper)
        self.canonical_p = canonical_p  # p_t (Canonical Mid)
        self.logit_x = logit_x          # x_t (Log-Odds)
        self.noise_var = noise_var      # sigma_eta^2 (Microstructure Noise)