# src/utils_split.py
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd

def split_users_by_hash(user_ids, train_frac=0.8):
    """
    Deterministic user split (stable across machines/versions).
    Returns (train_users_set, valid_users_set).
    """
    user_ids = pd.Series(user_ids).astype(str).dropna().unique()

    def h(u: str) -> int:
        return int(hashlib.md5(u.encode("utf-8")).hexdigest(), 16)

    order = np.argsort([h(u) for u in user_ids])
    user_sorted = user_ids[order]
    cut = int(len(user_sorted) * train_frac)

    return set(user_sorted[:cut]), set(user_sorted[cut:])