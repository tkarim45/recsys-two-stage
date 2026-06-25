"""Stage 1 — candidate retrieval. Matrix-factorization (truncated SVD) embeddings score
every item for a user; we keep the top-N as candidates. A popularity recommender is the
baseline the two-stage system must beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .config import N_FACTORS


class PopularityRecommender:
    name = "popularity"

    def fit(self, train: pd.DataFrame, n_items: int):
        counts = train["item_id"].value_counts()
        self.ranked = counts.index.tolist()
        self.seen = train.groupby("user_id")["item_id"].agg(set).to_dict()
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        seen = self.seen.get(user_id, set())
        return [i for i in self.ranked if i not in seen][:k]


class SVDRetriever:
    name = "svd"

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int):
        rows = train["user_id"].values
        cols = train["item_id"].values
        mat = csr_matrix((np.ones(len(train)), (rows, cols)), shape=(n_users, n_items))
        svd = TruncatedSVD(n_components=min(N_FACTORS, n_items - 1, n_users - 1), random_state=7)
        self.user_emb = svd.fit_transform(mat)          # n_users × k
        self.item_emb = svd.components_.T                # n_items × k
        self.seen = train.groupby("user_id")["item_id"].agg(set).to_dict()
        self.popularity = train["item_id"].value_counts().reindex(range(n_items), fill_value=0).values
        return self

    def score(self, user_id: int, item_ids) -> np.ndarray:
        return self.user_emb[user_id] @ self.item_emb[np.asarray(item_ids)].T

    def candidates(self, user_id: int, n: int) -> list[int]:
        scores = self.user_emb[user_id] @ self.item_emb.T
        seen = self.seen.get(user_id, set())
        order = np.argsort(-scores)
        out = [int(i) for i in order if i not in seen]
        return out[:n]
