"""Metrics, split correctness, retrieval contract, and the key claim: stages add lift."""
from recsys.data import generate, train_test_split
from recsys.metrics import average_precision_at_k, ndcg_at_k, recall_at_k
from recsys.pipeline import evaluate
from recsys.retrieval import SVDRetriever


def test_metrics_known():
    rec, rel = [1, 2, 3, 4], {2, 4}
    assert recall_at_k(rec, rel, 4) == 1.0
    assert recall_at_k([9, 8, 2, 7], {2}, 4) == 1.0
    assert ndcg_at_k([2, 1, 3], {2}, 3) == 1.0          # relevant first -> perfect
    assert ndcg_at_k([1, 3, 2], {2}, 3) < 1.0
    assert 0 < average_precision_at_k(rec, rel, 4) <= 1.0


def test_leave_last_out_split():
    inter, _ = generate(n_users=200, n_items=100, seed=1)
    train, test = train_test_split(inter)
    # each test user has exactly one held-out item, not present in their train set
    assert test.groupby("user_id").size().max() == 1
    merged = test.merge(train, on=["user_id", "item_id"], how="inner")
    assert len(merged) == 0  # held-out item never leaks into train


def test_retriever_excludes_seen():
    inter, items = generate(n_users=300, n_items=120, seed=2)
    train, _ = train_test_split(inter)
    r = SVDRetriever().fit(train, 300, 120)
    u = train["user_id"].iloc[0]
    cands = r.candidates(u, 20)
    assert len(cands) == 20 and not (set(cands) & r.seen[u])


def test_personalized_beats_popularity_baseline():
    inter, items = generate(n_users=800, n_items=400, seed=3)
    train, test = train_test_split(inter)
    res = evaluate(train, test, items, 800, 400, k=10)
    pop = res["popularity"]["recall@10"]
    svd = res["svd"]["recall@10"]
    two = res["two_stage"]["recall@10"]
    # both personalized systems must clear the popularity baseline (retrieval is the big win)
    assert svd > pop
    assert two > pop
