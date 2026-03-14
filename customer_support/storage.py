from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class JsonStore:
    """
    Simple JSON store for demo purposes (tickets, FAQ, etc.).
    Not a replacement for a real database.
    """

    path: str

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"tickets": {}, "faq": {}}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)

    def put(self, bucket: str, key: str, value: Any) -> None:
        data = self.load()
        data.setdefault(bucket, {})
        data[bucket][key] = value
        self.save(data)

    def get(self, bucket: str, key: str) -> Optional[Any]:
        return self.load().get(bucket, {}).get(key)

    def list_bucket(self, bucket: str) -> Dict[str, Any]:
        return dict(self.load().get(bucket, {}))
