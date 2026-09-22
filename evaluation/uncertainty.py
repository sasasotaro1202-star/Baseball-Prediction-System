"""Paired chronological block-bootstrap uncertainty for locked OOS evidence.

This module is evaluation-only. It resamples contiguous chronological blocks
with replacement and computes paired candidate-minus-baseline metric changes.
It never selects a candidate, modifies a model, or opens a holdout for tuning.

The paired design preserves the game-level dependence between baseline and
candidate predictions. It is intentionally deterministic through an explicit
seed and reports percentile intervals rather than a single point estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Sequence

import numpy as np

Array = np.ndarray


def _clip_probs(p: Array) -> Array:
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or len(p) == 0 or p.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty 2-D matrix with >=2 classes")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    sums = p.sum(axis=1, keepdims=True)
    if np.any(sums <= 0) or np.any(np.abs(sums - 1.0) > 1e-5):
        raise ValueError("probability rows must sum to 1")
    return p / sums


def _metrics(y: Array, p: Array) -> Mapping[str, float]:
    p = _clip_probs(p)
    y = np.asarray(y, dtype=int)
    if len(y) != len(p):
        raise ValueError("target/probability mismatch")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("target class outside probability columns")
    idx = np.arange(len(y))
    ll = -float(np.mean(np.log(np.maximum(p[idx, y], 1e-15))))
    one = np.zeros_like(p)
    one[idx, y] = 1.0
    brier = float(np.mean(np.sum((p - one) ** 2, axis=1)))
    accuracy = float(np.mean(np.argmax(p, axis=1) == y))
    return {"LogLoss": ll, "Brier": brier, "Accuracy": accuracy, "rows": int(len(y))}


def _block_starts(n: int, block_size: int) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if n < block_size:
        raise ValueError("rows must be >= block_size")
    return np.arange(0, n, block_size, dtype=int)


@dataclass(frozen=True)
class PairedBootstrapResult:
    status: str
    seed: int
    replications: int
    block_size: int
    block_scheme: str
    baseline: dict[str, float]
    candidate: dict[str, float]
    improvement: dict[str, float]
    improvement_ci95: dict[str, tuple[float, float]]
    p_improvement_positive: dict[str, float]


def paired_block_bootstrap(
    *,
    y: Array,
    baseline_proba: Array,
    candidate_proba: Array,
    block_size: int = 30,
    replications: int = 400,
    seed: int = 42,
    block_labels: Array | Sequence[object] | None = None,
) -> PairedBootstrapResult:
    """Estimate uncertainty of candidate-vs-baseline metric differences.

    Blocks are sampled with replacement over chronological contiguous chunks.
    The candidate and baseline use the identical resampled row indices.
    """
    y = np.asarray(y, dtype=int)
    b = _clip_probs(baseline_proba)
    c = _clip_probs(candidate_proba)
    if len(y) != len(b) or len(y) != len(c):
        raise ValueError("y and probability matrices must have equal length")
    if len(y) < max(60, block_size * 2):
        raise ValueError("insufficient holdout rows for paired block bootstrap")
    if replications < 100:
        raise ValueError("replications must be >= 100")

    baseline = dict(_metrics(y, b))
    candidate = dict(_metrics(y, c))
    improvement = {
        "LogLoss": float(baseline["LogLoss"] - candidate["LogLoss"]),
        "Brier": float(baseline["Brier"] - candidate["Brier"]),
        "Accuracy": float(candidate["Accuracy"] - baseline["Accuracy"]),
    }

    block_scheme = "fixed_contiguous"
    if block_labels is None:
        starts = _block_starts(len(y), block_size)
        blocks = [np.arange(s, min(s + block_size, len(y)), dtype=int) for s in starts]
    else:
        labels = np.asarray(block_labels, dtype=object)
        if labels.ndim != 1 or len(labels) != len(y):
            raise ValueError("block_labels must be one-dimensional and match y length")
        if any(value is None or str(value).strip() == "" or str(value).lower() == "nan" for value in labels):
            raise ValueError("block_labels must be complete")
        blocks = []
        start = 0
        while start < len(labels):
            end = start + 1
            while end < len(labels) and labels[end] == labels[start]:
                end += 1
            for sub_start in range(start, end, block_size):
                sub_end = min(end, sub_start + block_size)
                blocks.append(np.arange(sub_start, sub_end, dtype=int))
            start = end
        if len(blocks) < 2:
            raise ValueError("block_labels must yield at least two bootstrap blocks")
        block_scheme = "labeled_contiguous"
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in improvement}
    full_blocks = len(blocks)
    for _ in range(int(replications)):
        choices = rng.integers(0, full_blocks, size=full_blocks)
        idx = np.concatenate([blocks[i] for i in choices])
        yy = y[idx]
        bm = _metrics(yy, b[idx])
        cm = _metrics(yy, c[idx])
        samples["LogLoss"].append(bm["LogLoss"] - cm["LogLoss"])
        samples["Brier"].append(bm["Brier"] - cm["Brier"])
        samples["Accuracy"].append(cm["Accuracy"] - bm["Accuracy"])

    ci95 = {
        key: (
            float(np.quantile(np.asarray(values, dtype=float), 0.025)),
            float(np.quantile(np.asarray(values, dtype=float), 0.975)),
        )
        for key, values in samples.items()
    }
    positive = {
        key: float(np.mean(np.asarray(values, dtype=float) > 0))
        for key, values in samples.items()
    }
    return PairedBootstrapResult(
        status="EVALUATED",
        seed=int(seed),
        replications=int(replications),
        block_size=int(block_size),
        block_scheme=block_scheme,
        baseline=baseline,
        candidate=candidate,
        improvement=improvement,
        improvement_ci95=ci95,
        p_improvement_positive=positive,
    )


def to_dict(result: PairedBootstrapResult) -> dict[str, object]:
    return asdict(result)
