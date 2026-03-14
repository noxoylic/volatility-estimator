"""
Main polling loop: fetches orderbooks at intervals, processes into Ticks,
saves to Postgres (or CSV fallback).

Usage:
    python -m backend.main --source polymarket --token-id <TOKEN_ID> --interval 30
    python -m backend.main --source kalshi --ticker <TICKER> --interval 30
    python -m backend.main --csv  # use CSV fallback instead of Postgres
"""

import argparse
import time
import signal
import sys
from datetime import datetime, timezone

from backend.database.interface import DatabaseInterface, CSVFallback


_running = True


def _handle_sigint(sig, frame):
    global _running
    print("\nShutting down gracefully...")
    _running = False


signal.signal(signal.SIGINT, _handle_sigint)


def poll_polymarket(token_id: str, interval: float, storage):
    try:
        from backend.fetchers.polymarket import PolymarketFetcher
    except ImportError as exc:
        raise RuntimeError(
            "Polymarket fetcher is missing. Add backend/fetchers/polymarket.py"
        ) from exc

    fetcher = PolymarketFetcher()
    print(f"[polymarket] Polling token_id={token_id} every {interval}s")
    print("Press Ctrl+C to stop.\n")

    count = 0
    while _running:
        tick = fetcher.fetch_tick(token_id)
        if tick:
            count += 1
            if hasattr(storage, 'save_tick'):
                storage.save_tick(tick)
            else:
                storage.save_ticks([tick])
            print(f"[{count}] {tick.timestamp.isoformat()} "
                  f"bid={tick.best_bid:.4f} ask={tick.best_ask:.4f} "
                  f"x={tick.logit_x:.4f} noise={tick.noise_var:.6f}")
        else:
            print(f"[{datetime.now(timezone.utc).isoformat()}] No valid tick")

        time.sleep(interval)

    print(f"\nDone. Saved {count} ticks.")


def poll_kalshi(ticker: str, interval: float, storage):
    try:
        from backend.fetchers.kalshi import KalshiFetcher
    except ImportError as exc:
        raise RuntimeError(
            "Kalshi fetcher is missing. Add backend/fetchers/kalshi.py"
        ) from exc

    fetcher = KalshiFetcher()
    # No login needed for read-only endpoints (orderbook, markets)

    print(f"[kalshi] Polling ticker={ticker} every {interval}s")
    print("Press Ctrl+C to stop.\n")

    count = 0
    while _running:
        tick = fetcher.fetch_tick(ticker)
        if tick:
            count += 1
            if hasattr(storage, 'save_tick'):
                storage.save_tick(tick)
            else:
                storage.save_ticks([tick])
            print(f"[{count}] {tick.timestamp.isoformat()} "
                  f"bid={tick.best_bid:.4f} ask={tick.best_ask:.4f} "
                  f"x={tick.logit_x:.4f} noise={tick.noise_var:.6f}")
        else:
            print(f"[{datetime.now(timezone.utc).isoformat()}] No valid tick")

        time.sleep(interval)

    print(f"\nDone. Saved {count} ticks.")


def main():
    parser = argparse.ArgumentParser(description="RN-JD Market Data Poller")
    parser.add_argument("--source", choices=["polymarket", "kalshi"],
                        required=True)
    parser.add_argument("--token-id", help="Polymarket token ID")
    parser.add_argument("--ticker", help="Kalshi market ticker")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Polling interval in seconds (default 30)")
    parser.add_argument("--csv", action="store_true",
                        help="Use CSV fallback instead of Postgres")
    args = parser.parse_args()

    # Storage backend
    if args.csv:
        storage = CSVFallback(output_dir="data")
        print("Using CSV storage (data/ folder)")
    else:
        try:
            storage = DatabaseInterface()
            storage.connect()
            print("Connected to PostgreSQL")
        except Exception as e:
            print(f"Postgres connection failed: {e}")
            print("Falling back to CSV storage")
            storage = CSVFallback(output_dir="data")

    if args.source == "polymarket":
        if not args.token_id:
            print("Error: --token-id is required for polymarket")
            sys.exit(1)
        poll_polymarket(args.token_id, args.interval, storage)
    elif args.source == "kalshi":
        if not args.ticker:
            print("Error: --ticker is required for kalshi")
            sys.exit(1)
        poll_kalshi(args.ticker, args.interval, storage)


if __name__ == "__main__":
    main()
