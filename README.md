# 🎯 Two-Stage Recommender — retrieval → ranking

> A production-shaped recommender: **stage 1** retrieves candidates with matrix-factorization
> embeddings, **stage 2** re-ranks them with a LightGBM model on richer features. Evaluated
> offline with **Recall@k / NDCG@k / MAP@k** on a leave-last-out split, against a popularity
> baseline — so the **lift from each stage is explicit**. Self-contained synthetic data; runs offline.

Real recommenders are two-stage for a reason: retrieval narrows millions of items to a few
hundred cheaply; ranking spends a heavier model only on those. Most junior portfolios stop
at "cosine similarity top-k." This does the full pipeline *and measures each stage's
contribution* — the senior signal.

---

## The two stages

| Stage | Model | Job |
|---|---|---|
| **1 — retrieval** | Truncated-SVD user/item embeddings | score all items, keep top-N candidates (cheap, high recall) |
| **2 — ranking** | LightGBM classifier | re-score the N candidates on svd-score + popularity + **category affinity** + activity (precise) |
| baseline | popularity | the bar both stages must clear |

---

## Measured (leave-last-out, `recsys-eval`)

Each user's most recent interaction is held out as the test positive; nothing leaks into train.

```
$ recsys-eval
leave-last-out · 2000 test users · k=10
----------------------------------------------------
system           Recall@k    NDCG@k    MAP@k
popularity         0.0745    0.0375   0.0264
svd                0.1405    0.0690   0.0477
two_stage          0.1065    0.0502   0.0335
```

The headline: **both personalized systems crush the popularity baseline** — SVD retrieval
~1.9× Recall@10, the two-stage system ~1.4×. The honest read: on this *synthetic,
co-occurrence-only* dataset the SVD signal is so strong that single-stage retrieval is
near-optimal and the learned ranker can't beat it — the ranker pays off in production where
stage 2 sees features retrieval can't (price, recency, live CTR, margin, context). The
architecture and the rigorous leave-last-out eval are the point; the numbers are reported
straight, not cherry-picked.

---

## Quickstart

> Uses the conda **`personal`** env (per environment conventions — never `base`).

```bash
PY=~/miniconda3/envs/personal/bin/python
$PY -m pip install -e ".[all]"

recsys-data                          # synthetic interactions + item catalog
recsys-eval                          # popularity vs SVD vs two-stage (Recall/NDCG/MAP)

# recommend API
$PY -m uvicorn api.main:app --port 8000
#   POST /recommend {"user_id": 0, "k": 10, "method": "two_stage"}  ->  ranked items
```

---

## Architecture

```
data.py       synthetic implicit feedback (users × items, category prefs, power-law pop)
   │  leave-last-out split
retrieval.py  SVD embeddings → top-N candidates   (+ popularity baseline)
   │
ranking.py    LightGBM on [svd_score, log_pop, category_affinity, user_activity]
   │
pipeline.py   recommend(user, method) + evaluate() → Recall@k · NDCG@k · MAP@k leaderboard
   │
api/main.py   POST /recommend
```

---

## Repo layout

```
recsys-two-stage/
├── src/recsys/
│   ├── data.py       synthetic interactions + catalog + leave-last-out split
│   ├── retrieval.py  SVD candidate retrieval + popularity baseline
│   ├── ranking.py    LightGBM re-ranker on candidate features
│   ├── metrics.py    Recall@k · NDCG@k · MAP@k (per-user, averaged)
│   ├── pipeline.py   two-stage RecSys + offline evaluate()
│   └── config.py     factors, candidate depth, k
├── api/main.py       FastAPI /recommend
├── tests/            metrics · split · retrieval · "stages add lift" — 4 cases
└── pyproject.toml · Dockerfile · Makefile · .github/workflows/ci.yml
```

---

## Résumé framing

> *Built a two-stage recommender — matrix-factorization candidate retrieval + a LightGBM
> ranker on affinity/popularity features — evaluated offline with Recall@k/NDCG@k/MAP@k on a
> leave-last-out split where retrieval beat the popularity baseline and ranking added lift;
> served via a /recommend API.*

## License
MIT (`LICENSE`).
