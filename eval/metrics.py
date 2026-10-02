"""Ranking metrics with binary relevance, and a paired bootstrap for comparing two systems."""

import math
import random


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(item in relevant for item in ranked[:k]) / k


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG; 0 when nothing is relevant."""
    dcg = sum(1 / math.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def recall_at_k(approximate: list, exact: list, k: int) -> float:
    """Share of the exact top k that the approximate top k also found."""
    truth = set(exact[:k])
    return len(truth & set(approximate[:k])) / len(truth) if truth else 1.0


def paired_bootstrap(a: list[float], b: list[float], resamples: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a - b over paired items, with its 95% percentile bootstrap interval."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    return sum(diffs) / n, means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
