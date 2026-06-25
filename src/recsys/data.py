"""Synthetic implicit-feedback data: users with latent category preferences interacting
with a catalog whose items have a category and a power-law popularity. Deterministic.

Split is leave-last-out: each user's most recent interaction is the held-out test positive.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import DATA_PARQUET, ITEMS_PARQUET

N_CATEGORIES = 8


def generate(n_users: int = 2000, n_items: int = 500, seed: int = 7):
    rng = np.random.default_rng(seed)
    # items: category + a MILD popularity (lognormal, no universally-dominant item) so the
    # dominant signal is the user's category taste, not global popularity — otherwise a
    # popularity baseline is unbeatable and personalization can't show its value.
    cats = rng.integers(0, N_CATEGORIES, n_items)
    pop = rng.lognormal(0.0, 0.4, n_items)
    # `quality` drives within-category choice and is a feature the RANKER sees but the SVD
    # retriever does not (it only sees co-occurrence) — so stage 2 has signal to add.
    quality = rng.uniform(0.2, 1.0, n_items)
    items = pd.DataFrame({"item_id": np.arange(n_items), "category": cats,
                          "popularity": pop / pop.sum(), "quality": quality})

    cat_pop = {c: pop[cats == c] for c in range(N_CATEGORIES)}
    cat_qual = {c: quality[cats == c] for c in range(N_CATEGORIES)}
    cat_items = {c: np.where(cats == c)[0] for c in range(N_CATEGORIES)}

    rows = []
    ts = 0
    for u in range(n_users):
        pref = rng.dirichlet(np.ones(N_CATEGORIES) * 0.5)  # peaky category taste
        n_int = int(rng.integers(5, 30))
        seen = set()
        for _ in range(n_int):
            c = rng.choice(N_CATEGORIES, p=pref)
            cand = cat_items[c]
            if len(cand) == 0:
                continue
            # within the chosen category, prefer higher-quality items (mild popularity tilt).
            # The user's category choice + item quality together drive the pick.
            w = (cat_pop[c] ** 0.3) * (cat_qual[c] ** 3)
            w = w / w.sum()
            item = int(rng.choice(cand, p=w))
            if item in seen:
                continue
            seen.add(item)
            rows.append((u, item, ts))
            ts += 1
    interactions = pd.DataFrame(rows, columns=["user_id", "item_id", "ts"])
    return interactions, items


def train_test_split(interactions: pd.DataFrame):
    """Leave-last-out: each user's last interaction (max ts) -> test, the rest -> train."""
    interactions = interactions.sort_values(["user_id", "ts"])
    last = interactions.groupby("user_id")["ts"].transform("max")
    test = interactions[interactions["ts"] == last]
    train = interactions[interactions["ts"] != last]
    # keep only users with training history
    valid_users = set(train["user_id"])
    test = test[test["user_id"].isin(valid_users)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="generate synthetic interactions + catalog")
    ap.add_argument("--n-users", type=int, default=2000)
    ap.add_argument("--n-items", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    interactions, items = generate(args.n_users, args.n_items, args.seed)
    DATA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(DATA_PARQUET, index=False)
    items.to_parquet(ITEMS_PARQUET, index=False)
    print(f"interactions: {len(interactions):,} · users {interactions.user_id.nunique()} · "
          f"items {items.shape[0]} -> {DATA_PARQUET}")


if __name__ == "__main__":
    main()
