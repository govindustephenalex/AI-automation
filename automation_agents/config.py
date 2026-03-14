from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AutomationSettings:
    """
    Central paths for shared demo storage.
    """

    data_dir: str = os.getenv("AUTOMATION_DATA_DIR", "data")
    guardians_db: str = os.getenv("GUARDIANS_DB_FILE", "guardians_db.json")
    support_db: str = os.getenv("SUPPORT_DB_FILE", "support_store.json")

    @property
    def guardians_db_path(self) -> str:
        return os.path.join(self.data_dir, self.guardians_db)

    @property
    def support_db_path(self) -> str:
        return os.path.join(self.data_dir, self.support_db)
