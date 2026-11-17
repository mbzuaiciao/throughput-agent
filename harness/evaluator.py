"""Evaluator-owned measurement. The agent never imports this module.

Threat model: the agent controls `config` and (free-form arm) files under train/hooks/.
It must not be able to (a) change what is measured, (b) change how much work is done,
(c) read held-out results beyond its own task's feedback, or (d) fake the clock.
"""
from __future__ import annotations
import json
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path
from .schema import Task, Submission, EvalResult, RunMeasurement, Verdict
from .validate import validate_config, validate_patch, check_manifest

TIMING_AGREEMENT = 0.05   # CUDA-event tps vs wall-clock tps must agree within 5%


class Evaluator:
    def __init__(self, pristine_repo: Path, manifest: Path, baselines: dict[str, dict]):
        """baselines[task_id] = {"median_tps": float, "loss_at_check": float}"""
        self.pristine = pristine_repo
        self.manifest = manifest
        self.baselines = baselines

    def evaluate(self, task: Task, sub: Submission, timeout_s: int = 1800) -> EvalResult:
        base = self.baselines[task.task_id]
        errs = validate_config(task, sub.config)
        if sub.patch:
            errs += validate_patch(sub.patch)
        if errs:
            return EvalResult(sub.submission_id, task.task_id, Verdict.REJECTED_SCHEMA, detail="; ".join(errs))

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "repo"
            shutil.copytree(self.pristine, work)   # clean copy per submission: immutability
            if sub.patch:
                p = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"],
                                   input=sub.patch, text=True, cwd=work, capture_output=True)
                if p.returncode != 0:
                    return EvalResult(sub.submission_id, task.task_id, Verdict.REJECTED_SCHEMA,
                                      detail=f"patch failed: {p.stderr[:300]}")
            frozen_errs = check_manifest(work, self.manifest)
            if frozen_errs:
                return EvalResult(sub.submission_id, task.task_id, Verdict.REJECTED_FROZEN,
                                  detail="; ".join(frozen_errs))

            cfg_path = work / "config.json"
            cfg_path.write_text(json.dumps(sub.config))
            ms: list[RunMeasurement] = []
            for seed in range(task.n_repeats):
                out = work / f"out_{seed}.json"
                cmd = ["torchrun", f"--nproc_per_node={task.n_gpus}", "bench/entry.py",
                       "--task", task.task_id, "--config", str(cfg_path),
                       "--seed", str(seed), "--out", str(out)]
                try:
                    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    return EvalResult(sub.submission_id, task.task_id, Verdict.TIMEOUT, ms)
                if r.returncode != 0 or not out.exists():
                    v = Verdict.OOM if "out of memory" in r.stderr.lower() else Verdict.CRASH
                    return EvalResult(sub.submission_id, task.task_id, v, ms, detail=r.stderr[-400:])
                o = json.loads(out.read_text())
                wall_tps = o["timed_tokens"] / max(o["wall_timed_s"], 1e-9)
                ms.append(RunMeasurement(seed, o["tokens_per_sec"], wall_tps,
                                         o["peak_mem_gb"], o["loss_at_check"]))
        return self._judge(task, sub, base, ms)

    def _judge(self, task: Task, sub: Submission, base: dict, ms: list[RunMeasurement]) -> EvalResult:
        for m in ms:
            if abs(m.tokens_per_sec - m.parent_wall_tokens_per_sec) / m.parent_wall_tokens_per_sec > TIMING_AGREEMENT:
                return EvalResult(sub.submission_id, task.task_id, Verdict.REJECTED_TIMING, ms,
                                  detail="CUDA-event and wall-clock throughput disagree")
        med_loss = statistics.median(m.loss_at_check for m in ms)
        if abs(med_loss - base["loss_at_check"]) / base["loss_at_check"] > task.loss_rel_tol:
            return EvalResult(sub.submission_id, task.task_id, Verdict.REJECTED_LOSS, ms,
                              detail=f"loss {med_loss:.4f} vs baseline {base['loss_at_check']:.4f}")
        med_tps = statistics.median(m.tokens_per_sec for m in ms)
        return EvalResult(sub.submission_id, task.task_id, Verdict.ACCEPTED, ms,
                          median_tps=med_tps, speedup_vs_baseline=med_tps / base["median_tps"])
