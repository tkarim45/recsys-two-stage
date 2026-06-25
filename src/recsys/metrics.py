"""Ranking metrics for implicit-feedback recsys (per-user, then averaged)."""
from __future__ import annotations

import numpy as np


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    score, hits = 0.0, 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def evaluate_users(recs_by_user: dict[int, list], truth_by_user: dict[int, set], k: int) -> dict:
    users = [u for u in truth_by_user if truth_by_user[u]]
    if not users:
        return {}
    mean = lambda f: float(np.mean([f(recs_by_user.get(u, []), truth_by_user[u], k) for u in users]))
    return {
        f"recall@{k}": round(mean(recall_at_k), 4),
        f"ndcg@{k}": round(mean(ndcg_at_k), 4),
        f"map@{k}": round(mean(average_precision_at_k), 4),
        "n_users": len(users),
    }
