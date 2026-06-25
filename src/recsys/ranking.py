"""Stage 2 — learned ranking. A LightGBM model re-scores the retrieved candidates using
features the retrieval score alone misses: item popularity, the user's category affinity,
and activity. Trained on the user's real interactions (positive) vs sampled negatives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .retrieval import SVDRetriever

N_CATEGORIES = 8


class Ranker:
    name = "two_stage"

    def __init__(self):
        self.cols = ["svd_score", "log_pop", "cat_affinity", "user_activity", "quality"]

    def _user_cat_dist(self, train: pd.DataFrame, items: pd.DataFrame) -> dict:
        merged = train.merge(items[["item_id", "category"]], on="item_id")
        dist = {}
        for u, g in merged.groupby("user_id"):
            v = np.bincount(g["category"].values, minlength=N_CATEGORIES).astype(float)
            dist[u] = v / v.sum() if v.sum() else v
        return dist

    def _features(self, user_id, item_ids, retriever, items_cat, pop, cat_dist, activity, quality):
        item_ids = np.asarray(item_ids)
        svd = retriever.score(user_id, item_ids)
        ud = cat_dist.get(user_id, np.zeros(N_CATEGORIES))
        aff = ud[items_cat[item_ids]] if len(ud) else np.zeros(len(item_ids))
        return np.column_stack([
            svd, np.log1p(pop[item_ids]), aff,
            np.full(len(item_ids), activity.get(user_id, 0)), quality[item_ids],
        ])

    def fit(self, train: pd.DataFrame, items: pd.DataFrame, retriever: SVDRetriever,
            n_users: int, n_items: int, neg_ratio: int = 4, seed: int = 7):
        from lightgbm import LGBMClassifier

        rng = np.random.default_rng(seed)
        self.items_cat = items.set_index("item_id")["category"].reindex(range(n_items)).values
        self.quality = items.set_index("item_id")["quality"].reindex(range(n_items)).fillna(0).values
        self.pop = retriever.popularity
        self.cat_dist = self._user_cat_dist(train, items)
        self.activity = train.groupby("user_id").size().to_dict()
        self.retriever = retriever
        seen = train.groupby("user_id")["item_id"].agg(set).to_dict()

        # Train on the user's interactions (positive) vs sampled negatives. Features:
        # retrieval affinity + popularity + category affinity + activity + item quality.
        X, y = [], []
        for u, pos_items in train.groupby("user_id")["item_id"]:
            pos = pos_items.values
            negs = [i for i in rng.integers(0, n_items, size=len(pos) * neg_ratio)
                    if i not in seen.get(u, set())]
            feats = self._features(u, list(pos) + negs, retriever, self.items_cat,
                                   self.pop, self.cat_dist, self.activity, self.quality)
            X.append(feats)
            y.extend([1] * len(pos) + [0] * len(negs))

        self.model = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                    num_leaves=31, verbosity=-1)
        self.model.fit(pd.DataFrame(np.vstack(X), columns=self.cols), np.array(y))
        return self

    def rank(self, user_id: int, candidates: list[int], k: int) -> list[int]:
        if not candidates:
            return []
        feats = self._features(user_id, candidates, self.retriever, self.items_cat,
                               self.pop, self.cat_dist, self.activity, self.quality)
        scores = self.model.predict_proba(pd.DataFrame(feats, columns=self.cols))[:, 1]
        order = np.argsort(-scores)
        return [candidates[i] for i in order[:k]]
