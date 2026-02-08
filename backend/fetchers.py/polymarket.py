from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderBookSummary
from typing import Dict, List, Optional

class PolymarketFetcher:
    """
    Interacts with Polymarket CLOB API using py-clob-client.
    """
    HOST = "https://clob.polymarket.com"

    def __init__(self, key: str = None, chain_id: int = 137):
        # Initialize client (No auth needed for public data like books)
        self.client = ClobClient(self.HOST, key=key, chain_id=chain_id)

    def fetch_market_metadata(self, condition_id: str) -> Dict:
        """
        Fetches static details (Question, Slug) using the Gamma API (via client helper).
        """
        try:
            # The client usually provides a way to get market info, or we hit the endpoint directly
            # For simplicity, we assume we get the market object
            market = self.client.get_market(condition_id) # Hypothetical helper or direct call
            return market
        except Exception as e:
            print(f"Error fetching metadata: {e}")
            return {}

    def fetch_orderbook(self, token_id: str) -> Optional[OrderBookSummary]:
        """
        Fetches L2 Orderbook.
        Required for Section 5.1 "Microstructure noise" estimation.
        """
        try:
            # get_order_book returns an OrderBookSummary object
            return self.client.get_order_book(token_id)
        except Exception as e:
            # print(f"Error fetching book for {token_id}: {e}")
            return None