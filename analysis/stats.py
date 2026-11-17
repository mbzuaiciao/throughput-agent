"""Bootstrap statistics. With ~6 held-out tasks the CIs WILL be wide; report them anyway."""
from __future__ import annotations
import math
import random
import statistics


def bootstrap_ci(values: list[float], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0, stat=statistics.mean) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return stat(values), lo, hi


def geo_mean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def dev_heldout_gap(dev: list[float], held: list[float], n_boot: int = 10_000,
                    seed: int = 0) -> tuple[float, float, float]:
    """Gap in mean log-speedup (dev minus held-out) with bootstrap CI: the AFML-style overfitting statistic.
    Inputs are per-task best speedups (>0)."""
    rng = random.Random(seed)
    ld = [math.log(x) for x in dev]
    lh = [math.log(x) for x in held]
    diffs = sorted(
        statistics.mean(ld[rng.randrange(len(ld))] for _ in ld)
        - statistics.mean(lh[rng.randrange(len(lh))] for _ in lh)
        for _ in range(n_boot)
    )
    return (statistics.mean(ld) - statistics.mean(lh),
            diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1])
