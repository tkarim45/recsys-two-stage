"""FastAPI recommender service. Fits the two-stage model once at startup.

  GET  /health
  POST /recommend {user_id, k, method}  -> ranked items (id, category)
"""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recsys.config import DATA_PARQUET, ITEMS_PARQUET, TOP_K
from recsys.data import generate, train_test_split
from recsys.pipeline import RecSys

app = FastAPI(title="Two-Stage Recommender", version="0.1.0")

if DATA_PARQUET.exists():
    _interactions, _items = pd.read_parquet(DATA_PARQUET), pd.read_parquet(ITEMS_PARQUET)
else:
    _interactions, _items = generate()
_n_users = int(_interactions.user_id.max()) + 1
_n_items = int(_items.item_id.max()) + 1
_train, _ = train_test_split(_interactions)
_rec = RecSys().fit(_train, _items, _n_users, _n_items)
_cat = _items.set_index("item_id")["category"].to_dict()


class RecRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    k: int = Field(TOP_K, ge=1, le=100)
    method: str = "two_stage"


@app.get("/health")
def health():
    return {"status": "ok", "n_users": _n_users, "n_items": _n_items}


@app.post("/recommend")
def recommend(req: RecRequest):
    if req.user_id >= _n_users:
        raise HTTPException(404, f"unknown user_id {req.user_id}")
    if req.method not in ("popularity", "svd", "two_stage"):
        raise HTTPException(400, "method must be popularity | svd | two_stage")
    items = _rec.recommend(req.user_id, req.k, req.method)
    return {"user_id": req.user_id, "method": req.method,
            "recommendations": [{"item_id": int(i), "category": int(_cat.get(i, -1))} for i in items]}
