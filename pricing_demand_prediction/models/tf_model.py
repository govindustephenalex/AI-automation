from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TfRegressor:
    params: Optional[Dict[str, Any]] = None
    model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import tensorflow as tf
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency: tensorflow") from exc

        tf.random.set_seed(42)

        p = {
            "hidden_units": [256, 128, 64],
            "dropout": 0.1,
            "lr": 1e-3,
            "epochs": 40,
            "batch_size": 256,
        }
        if self.params:
            p.update(self.params)

        inp = tf.keras.Input(shape=(X.shape[1],))
        x = inp
        for u in p["hidden_units"]:
            x = tf.keras.layers.Dense(int(u), activation="relu")(x)
            if p["dropout"] and float(p["dropout"]) > 0:
                x = tf.keras.layers.Dropout(float(p["dropout"]))(x)
        out = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(p["lr"])),
            loss="mse",
        )

        model.fit(
            X.astype("float32"),
            y.astype("float32"),
            epochs=int(p["epochs"]),
            batch_size=int(p["batch_size"]),
            verbose=0,
            validation_split=0.1,
        )
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit.")
        pred = self.model.predict(X.astype("float32"), verbose=0).reshape(-1)
        return np.asarray(pred, dtype=float)

    def info(self) -> Dict[str, Any]:
        return {"backend": "tf", "params": self.params or {}}
