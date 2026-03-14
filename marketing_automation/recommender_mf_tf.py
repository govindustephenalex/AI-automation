from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MFConfig:
    embedding_dim: int = 64
    epochs: int = 10
    batch_size: int = 2048
    negatives_per_positive: int = 3
    lr: float = 1e-3


class MatrixFactorizationRecommender:
    """
    Implicit-feedback matrix factorization with negative sampling in TensorFlow.

    Fits from interactions:
      customer_id, item_id, value (implicit strength)
    """

    def __init__(self, cfg: Optional[MFConfig] = None):
        self.cfg = cfg or MFConfig()
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: List[str] = []
        self.seen_items_by_user: Dict[str, set[str]] = {}
        self.model = None

    def fit(self, interactions: pd.DataFrame) -> "MatrixFactorizationRecommender":
        try:
            import tensorflow as tf
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency: tensorflow") from exc

        df = interactions[["customer_id", "item_id", "value"]].copy()
        df["customer_id"] = df["customer_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(1.0).astype(float)
        if df.empty:
            raise ValueError("No interactions to train on.")

        users = sorted(df["customer_id"].unique().tolist())
        items = sorted(df["item_id"].unique().tolist())
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {it: i for i, it in enumerate(items)}
        self.idx_to_item = items

        self.seen_items_by_user = {}
        for u, grp in df.groupby("customer_id"):
            self.seen_items_by_user[str(u)] = set(grp["item_id"].astype(str).tolist())

        u_idx = df["customer_id"].map(self.user_to_idx).to_numpy(dtype=np.int32)
        i_idx = df["item_id"].map(self.item_to_idx).to_numpy(dtype=np.int32)

        # Positives: label=1
        pos_u = u_idx
        pos_i = i_idx
        pos_y = np.ones_like(pos_u, dtype=np.float32)

        # Negatives: sample random items for each positive
        rng = np.random.default_rng(42)
        neg_u = np.repeat(pos_u, int(self.cfg.negatives_per_positive)).astype(np.int32)
        neg_i = rng.integers(low=0, high=len(items), size=len(neg_u), dtype=np.int32)
        neg_y = np.zeros_like(neg_u, dtype=np.float32)

        X_u = np.concatenate([pos_u, neg_u], axis=0)
        X_i = np.concatenate([pos_i, neg_i], axis=0)
        y = np.concatenate([pos_y, neg_y], axis=0)

        # Shuffle
        perm = rng.permutation(len(y))
        X_u, X_i, y = X_u[perm], X_i[perm], y[perm]

        ds = tf.data.Dataset.from_tensor_slices(((X_u, X_i), y))
        ds = ds.shuffle(min(200_000, len(y)), seed=42, reshuffle_each_iteration=True).batch(int(self.cfg.batch_size))

        user_in = tf.keras.Input(shape=(), dtype=tf.int32, name="user")
        item_in = tf.keras.Input(shape=(), dtype=tf.int32, name="item")
        user_emb = tf.keras.layers.Embedding(len(users), int(self.cfg.embedding_dim), name="user_emb")(user_in)
        item_emb = tf.keras.layers.Embedding(len(items), int(self.cfg.embedding_dim), name="item_emb")(item_in)
        # Dot product -> probability of interaction
        x = tf.keras.layers.Dot(axes=1)([user_emb, item_emb])
        x = tf.keras.layers.Flatten()(x)
        out = tf.keras.layers.Activation("sigmoid")(x)
        model = tf.keras.Model(inputs=[user_in, item_in], outputs=out)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(self.cfg.lr)), loss="binary_crossentropy")
        model.fit(ds, epochs=int(self.cfg.epochs), verbose=0)
        self.model = model
        return self

    def recommend(self, customer_id: str, k: int = 10) -> List[Tuple[str, float]]:
        if self.model is None:
            raise RuntimeError("Recommender not fit.")
        import tensorflow as tf

        cid = str(customer_id)
        if cid not in self.user_to_idx:
            return []

        u = self.user_to_idx[cid]
        items = np.arange(len(self.idx_to_item), dtype=np.int32)
        users = np.full_like(items, u, dtype=np.int32)
        scores = self.model.predict([users, items], verbose=0).reshape(-1)

        seen = self.seen_items_by_user.get(cid, set())
        candidates = []
        for idx, score in enumerate(scores):
            item_id = self.idx_to_item[int(idx)]
            if item_id in seen:
                continue
            candidates.append((item_id, float(score)))
        candidates.sort(key=lambda t: t[1], reverse=True)
        return candidates[: int(k)]

    def most_popular(self, interactions: pd.DataFrame, k: int = 10) -> List[str]:
        top = interactions.groupby("item_id")["value"].sum().sort_values(ascending=False).head(int(k))
        return [str(i) for i in top.index.tolist()]
