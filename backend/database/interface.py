import os
import csv
import time
from datetime import datetime, timezone
from typing import List, Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from backend.models.schema import Tick, MarketMetadata

load_dotenv()


def _get_db_config() -> dict:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "dbname": os.getenv("POSTGRES_DB", "rnjd_markets"),
        "user": os.getenv("POSTGRES_USER", "rnjd"),
        "password": os.getenv("POSTGRES_PASSWORD", "changeme"),
    }


class DatabaseInterface:
    """Postgres read/write for RN-JD tick data and market metadata."""

    def __init__(self, db_config: dict = None):
        self.db_config = db_config or _get_db_config()
        self._conn = None

    def connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.db_config)
            self._conn.autocommit = False
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ── Metadata ────────────────────────────────────────────

    def save_metadata(self, meta: MarketMetadata):
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO market_meta (token_id, condition_id, source, question, slug, description)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (token_id) DO NOTHING
                """,
                (meta.token_id, meta.condition_id, meta.source,
                 meta.question, meta.slug, meta.description),
            )
        conn.commit()

    def get_metadata(self, token_id: str) -> Optional[dict]:
        conn = self.connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM market_meta WHERE token_id = %s", (token_id,))
            return cur.fetchone()

    def list_markets(self, source: str = None) -> List[dict]:
        conn = self.connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if source:
                cur.execute("SELECT * FROM market_meta WHERE source = %s ORDER BY created_at DESC", (source,))
            else:
                cur.execute("SELECT * FROM market_meta ORDER BY created_at DESC")
            return cur.fetchall()

    # ── Ticks ───────────────────────────────────────────────

    def save_tick(self, tick: Tick):
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO market_ticks
                    (timestamp, token_id, best_bid, best_ask, spread,
                     depth_bid, depth_ask, canonical_p, logit_x, noise_var)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (tick.timestamp, tick.token_id,
                 tick.best_bid, tick.best_ask, tick.spread,
                 tick.depth_bid, tick.depth_ask,
                 tick.canonical_p, tick.logit_x, tick.noise_var),
            )
        conn.commit()

    def save_ticks_batch(self, ticks: List[Tick]):
        """Bulk insert using executemany for performance."""
        if not ticks:
            return
        conn = self.connect()
        rows = [
            (t.timestamp, t.token_id,
             t.best_bid, t.best_ask, t.spread,
             t.depth_bid, t.depth_ask,
             t.canonical_p, t.logit_x, t.noise_var)
            for t in ticks
        ]
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO market_ticks
                    (timestamp, token_id, best_bid, best_ask, spread,
                     depth_bid, depth_ask, canonical_p, logit_x, noise_var)
                VALUES %s
                """,
                rows,
                page_size=500,
            )
        conn.commit()

    def get_ticks(self, token_id: str, since: datetime = None,
                  limit: int = 10000) -> List[dict]:
        conn = self.connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if since:
                cur.execute(
                    """
                    SELECT * FROM market_ticks
                    WHERE token_id = %s AND timestamp >= %s
                    ORDER BY timestamp ASC LIMIT %s
                    """,
                    (token_id, since, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM market_ticks
                    WHERE token_id = %s
                    ORDER BY timestamp ASC LIMIT %s
                    """,
                    (token_id, limit),
                )
            return cur.fetchall()

    def get_latest_tick(self, token_id: str) -> Optional[dict]:
        conn = self.connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM market_ticks
                WHERE token_id = %s ORDER BY timestamp DESC LIMIT 1
                """,
                (token_id,),
            )
            return cur.fetchone()


class CSVFallback:
    """Write ticks to CSV when Postgres is not available (dev/testing)."""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_ticks(self, ticks: List[Tick], filename: str = "ticks.csv"):
        path = os.path.join(self.output_dir, filename)
        file_exists = os.path.isfile(path)

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "token_id", "best_bid", "best_ask",
                    "spread", "depth_bid", "depth_ask",
                    "canonical_p", "logit_x", "noise_var",
                ])
            for t in ticks:
                writer.writerow([
                    t.timestamp, t.token_id,
                    t.best_bid, t.best_ask, t.spread,
                    t.depth_bid, t.depth_ask,
                    t.canonical_p, t.logit_x, t.noise_var,
                ])

    def read_ticks(self, filename: str = "ticks.csv"):
        """Read ticks back as list of dicts."""
        path = os.path.join(self.output_dir, filename)
        if not os.path.isfile(path):
            return []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)
