"""Static checks run BEFORE any GPU time is spent."""
from __future__ import annotations
import fnmatch
import hashlib
import json
import re
from pathlib import Path
from .schema import Task

# name -> (type, allowed range or allowed set)
CONFIG_SPEC: dict[str, tuple[type, object]] = {
    "micro_batch_size": (int, (1, 256)),
    "grad_accum_steps": (int, (1, 128)),
    "precision": (str, {"bf16", "fp16", "fp32"}),
    "compile_mode": (str, {"none", "default", "reduce-overhead", "max-autotune"}),
    "flash_attn": (bool, None),
    "fused_optimizer": (bool, None),
    "activation_checkpointing": (bool, None),
    "dataloader_workers": (int, (0, 32)),
    "pin_memory": (bool, None),
    "prefetch_factor": (int, (1, 16)),
    "parallelism": (str, {"ddp", "fsdp"}),
    "fsdp_sharding": (str, {"full", "grad_op", "no_shard"}),
    "ddp_bucket_mb": (int, (1, 512)),
}

# Free-form arm: patches may only touch these paths.
ALLOWED_PATCH_GLOBS = ["train/hooks/*.py"]


def validate_config(task: Task, cfg: dict) -> list[str]:
    errs: list[str] = []
    for k, v in cfg.items():
        if k not in CONFIG_SPEC:
            errs.append(f"unknown key: {k}")
            continue
        typ, rng = CONFIG_SPEC[k]
        if typ is int and isinstance(v, bool):  # bool is an int subclass; reject
            errs.append(f"{k}: expected int")
        elif not isinstance(v, typ):
            errs.append(f"{k}: expected {typ.__name__}")
        elif isinstance(rng, tuple) and not (rng[0] <= v <= rng[1]):
            errs.append(f"{k}: {v} outside {rng}")
        elif isinstance(rng, set) and v not in rng:
            errs.append(f"{k}: {v!r} not in {sorted(rng)}")
    # Tokens per optimizer step must match the task exactly (no cheating via less work).
    mb = cfg.get("micro_batch_size")
    ga = cfg.get("grad_accum_steps")
    if isinstance(mb, int) and isinstance(ga, int):
        tokens = mb * ga * task.n_gpus * task.seq_len
        if tokens != task.global_batch_tokens:
            errs.append(f"tokens/step {tokens} != task {task.global_batch_tokens}")
    else:
        errs.append("micro_batch_size and grad_accum_steps are required")
    if cfg.get("parallelism") == "ddp" and "fsdp_sharding" in cfg:
        errs.append("fsdp_sharding set but parallelism=ddp")
    return errs


_PATCH_FILE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


def patch_paths(patch: str) -> list[str]:
    return _PATCH_FILE.findall(patch)


def validate_patch(patch: str) -> list[str]:
    paths = patch_paths(patch)
    if not paths:
        return ["patch touches no files"]
    return [
        f"patch path not allowed: {p}"
        for p in paths
        if not any(fnmatch.fnmatch(p, g) for g in ALLOWED_PATCH_GLOBS)
    ]


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_manifest(root: Path, frozen: list[str]) -> dict[str, str]:
    return {rel: hash_file(root / rel) for rel in frozen}


def check_manifest(root: Path, manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    return [
        f"frozen file changed: {rel}"
        for rel, h in manifest.items()
        if not (root / rel).exists() or hash_file(root / rel) != h
    ]
