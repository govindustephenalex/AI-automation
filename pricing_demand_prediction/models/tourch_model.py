from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TorchRegressor:
    params: Optional[Dict[str, Any]] = None
    model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency: torch") from exc

        torch.manual_seed(42)

        p = {
            "hidden_units": [256, 128, 64],
            "dropout": 0.1,
            "lr": 1e-3,
            "epochs": 50,
            "batch_size": 256,
            "weight_decay": 1e-4,
        }
        if self.params:
            p.update(self.params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers: list[nn.Module] = []
        in_dim = int(X.shape[1])
        for u in p["hidden_units"]:
            layers.append(nn.Linear(in_dim, int(u)))
            layers.append(nn.ReLU())
            if p["dropout"] and float(p["dropout"]) > 0:
                layers.append(nn.Dropout(float(p["dropout"])))
            in_dim = int(u)
        layers.append(nn.Linear(in_dim, 1))
        net = nn.Sequential(*layers).to(device)

        opt = torch.optim.AdamW(net.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
        loss_fn = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        n = X_t.shape[0]
        batch = int(p["batch_size"])
        epochs = int(p["epochs"])

        net.train()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                xb = X_t[idx].to(device)
                yb = y_t[idx].to(device)
                opt.zero_grad()
                pred = net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        self.model = net.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit.")
        import torch

        device = next(self.model.parameters()).device
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            pred = self.model(X_t).reshape(-1).cpu().numpy()
        return np.asarray(pred, dtype=float)

    def info(self) -> Dict[str, Any]:
        return {"backend": "torch", "params": self.params or {}}
