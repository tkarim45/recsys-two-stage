"""Two-stage recommender + offline evaluation.

Compares three systems on the same leave-last-out split so the lift from each stage is
explicit: popularity baseline → SVD retrieval (stage 1) → SVD + LightGBM ranking (stage 2).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import DATA_PARQUET, ITEMS_PARQUET, N_CANDIDATES, REPORTS, TOP_K
from .data import generate, train_test_split
from .metrics import evaluate_users
from .ranking import Ranker
from .retrieval import PopularityRecommender, SVDRetriever


class RecSys:
    def __init__(self, n_candidates: int = N_CANDIDATES):
        self.n_candidates = n_candidates

    def fit(self, train: pd.DataFrame, items: pd.DataFrame, n_users: int, n_items: int):
        self.n_users, self.n_items = n_users, n_items
        self.pop = PopularityRecommender().fit(train, n_items)
        self.retriever = SVDRetriever().fit(train, n_users, n_items)
        self.ranker = Ranker().fit(train, items, self.retriever, n_users, n_items)
        return self

    def recommend(self, user_id: int, k: int = TOP_K, method: str = "two_stage") -> list[int]:
        if method == "popularity":
            return self.pop.recommend(user_id, k)
        cands = self.retriever.candidates(user_id, self.n_candidates)
        if method == "svd":
            return cands[:k]
        return self.ranker.rank(user_id, cands, k)


def evaluate(train, test, items, n_users, n_items, k: int = TOP_K) -> dict:
    rec = RecSys().fit(train, items, n_users, n_items)
    truth = test.groupby("user_id")["item_id"].agg(set).to_dict()
    out = {}
    for method in ("popularity", "svd", "two_stage"):
        recs = {u: rec.recommend(u, k, method) for u in truth}
        out[method] = evaluate_users(recs, truth, k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="evaluate the two-stage recommender")
    ap.add_argument("--k", type=int, default=TOP_K)
    args = ap.parse_args()

    if DATA_PARQUET.exists():
        interactions, items = pd.read_parquet(DATA_PARQUET), pd.read_parquet(ITEMS_PARQUET)
    else:
        interactions, items = generate()
    n_users = int(interactions.user_id.max()) + 1
    n_items = int(items.item_id.max()) + 1
    train, test = train_test_split(interactions)

    res = evaluate(train, test, items, n_users, n_items, args.k)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "eval.json").write_text(json.dumps(res, indent=2))

    print(f"\nleave-last-out · {res['two_stage']['n_users']} test users · k={args.k}")
    print("-" * 52)
    print(f"{'system':14} {'Recall@k':>10} {'NDCG@k':>9} {'MAP@k':>8}")
    for method, m in res.items():
        print(f"{method:14} {m[f'recall@{args.k}']:>10.4f} {m[f'ndcg@{args.k}']:>9.4f} "
              f"{m[f'map@{args.k}']:>8.4f}")


if __name__ == "__main__":
    main()
