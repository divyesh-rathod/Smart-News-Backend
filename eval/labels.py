"""
Relevance labels in eval/labels.jsonl, one JSON object per line, keyed by article link.

Links rather than database ids, because a re-scrape on another machine gives the same article a new id. The
file holds no article text: the repository is public and the text belongs to the Guardian.
"""

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

LABELS_PATH = Path(__file__).with_name("labels.jsonl")
POOL_DEPTH = 5
SAMPLE_SEED = "smart-news-eval"


def read_labels(path: Path = LABELS_PATH) -> dict[tuple[str, str], bool]:
    """(source link, candidate link) -> relevant. A later line for the same pair replaces an earlier one."""
    labels: dict[tuple[str, str], bool] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                labels[(record["source"], record["candidate"])] = record["relevant"]
    return labels


def append_label(source: str, candidate: str, relevant: bool, path: Path = LABELS_PATH) -> None:
    record = {
        "source": source,
        "candidate": candidate,
        "relevant": relevant,
        "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _sample_order(link: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SEED}:{link}".encode()).hexdigest()


def choose_sources(links: list[str], labels: dict[tuple[str, str], bool], n: int) -> list[str]:
    """
    n source articles to evaluate: already-labelled ones first, then a fixed pseudo-random order.

    Starting from the labelled ones keeps the sample stable when new articles are scraped.
    """
    available = set(links)
    labelled = {source for source, _ in labels} & available
    ordered = sorted(labelled, key=_sample_order) + sorted(available - labelled, key=_sample_order)
    return ordered[:n]


def pool(source: str, rankings: dict[str, list[str]], depth: int = POOL_DEPTH) -> list[str]:
    """Union of every system's top `depth`, shuffled per source so the labeller can't tell which system found what."""
    candidates = sorted({link for ranked in rankings.values() for link in ranked[:depth]})
    random.Random(source).shuffle(candidates)
    return candidates
