"""Held-out discipline: a scaffold is frozen, hashed, and scored on a held-out split once."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def scaffold_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


class HeldoutLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.touch(exist_ok=True)

    def _rows(self) -> list[dict]:
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def claim(self, scaffold: str, split: str, allow_reuse: bool = False) -> None:
        """Raise if this split was already scored by a different scaffold (i.e. iterating on held-out)."""
        prior = [r for r in self._rows() if r["split"] == split]
        if prior and not allow_reuse and any(r["scaffold"] != scaffold for r in prior):
            raise PermissionError(f"{split} already scored by another scaffold; pass allow_reuse=True "
                                  "to override (logged; report the result as contaminated)")
        with self.path.open("a") as f:
            f.write(json.dumps({"scaffold": scaffold, "split": split, "reuse": bool(prior)}) + "\n")
