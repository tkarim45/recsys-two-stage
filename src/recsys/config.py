"""Paths + recsys defaults."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("RECSYS_ROOT", Path(__file__).resolve().parents[2]))
DATA_PARQUET = ROOT / "data" / "interactions.parquet"
ITEMS_PARQUET = ROOT / "data" / "items.parquet"
REPORTS = ROOT / "reports"

N_FACTORS = 32        # SVD embedding dim
N_CANDIDATES = 50     # stage-1 retrieval depth
TOP_K = 10            # evaluation cutoff
