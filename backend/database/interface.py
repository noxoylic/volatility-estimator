from backend.models.schema import Tick, MarketMetadata

class StorageInterface:
    """
    Abstract interface for persisting the RN-JD data stream.
    """

    def __init__(self, db_config: dict = None):
        self.db_config = db_config
        # Connection setup (psycopg2) would go here
        pass

    def save_metadata(self, meta: MarketMetadata):
        """
        Saves static market info ONCE (Idempotent).
        SQL: INSERT INTO market_meta ... ON CONFLICT DO NOTHING
        """
        pass

    def save_tick(self, tick: Tick):
        """
        Saves the flat tick data stream.
        This aligns with the "Observation model" Eq (10) in the paper.
        """
        if not tick:
            return

        # Placeholder for INSERT logic
        # SQL: INSERT INTO market_ticks (time, token_id, logit_x, noise_var...)
        # print(f"Saving Tick: {tick.timestamp} | x={tick.logit_x:.4f} | noise={tick.noise_var:.6f}")
        pass