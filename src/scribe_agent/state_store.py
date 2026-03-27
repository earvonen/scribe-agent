from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StateStore:
    """Tracks processed source .txt content (by SHA-256) so the same text is not turned into duplicate issues."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def _atomic_write(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self._path.parent),
            prefix=".scribe-state-",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, self._path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def load(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {}
        try:
            with self._path.open(encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("State file unreadable (%s), starting fresh", e)
            return {}

    def is_content_processed(self, content_sha256: str) -> bool:
        data = self.load()
        return content_sha256 in data.get("processed_txt_sha256", {})

    def mark_content_processed(self, content_sha256: str, meta: dict[str, Any]) -> None:
        data = self.load()
        inc = data.setdefault("processed_txt_sha256", {})
        inc[content_sha256] = meta
        self._atomic_write(data)
