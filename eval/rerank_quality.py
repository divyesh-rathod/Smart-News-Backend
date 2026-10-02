"""
Score every system's top 5 against the hand labels:  python -m eval.rerank_quality

Uses the sources in eval/labels.jsonl that are still in the local corpus. Prints, per system: P@1, P@5,
nDCG@5, judged@5 (share of its top 5 that has a label; unlabelled counts as not relevant), the paired
bootstrap 95% interval of its P@5 minus production's, and its median ranking time.
"""

import statistics

from eval.corpus import load_corpus
from eval.labels import read_labels
from eval.metrics import ndcg_at_k, paired_bootstrap, precision_at_k
from eval.systems import DESCRIPTIONS, SYSTEMS, TOP_K, Systems

BASELINE = "rerank"


def main() -> None:
    labels = read_labels()
    corpus = load_corpus()
    sources = sorted({source for source, _ in labels} & corpus.row.keys())
    if not sources:
        raise SystemExit("No labelled sources in the local corpus. Run: python -m eval.label")
    relevant = {source: {c for (s, c), rel in labels.items() if s == source and rel} for source in sources}

    systems = Systems(corpus)
    scores: dict[str, dict[str, list[float]]] = {}
    for name in SYSTEMS:
        per_source = {"p1": [], "p5": [], "ndcg5": [], "judged5": []}
        for source in sources:
            ranked = systems.links(name, corpus.row[source])
            per_source["p1"].append(precision_at_k(ranked, relevant[source], 1))
            per_source["p5"].append(precision_at_k(ranked, relevant[source], TOP_K))
            per_source["ndcg5"].append(ndcg_at_k(ranked, relevant[source], TOP_K))
            per_source["judged5"].append(sum((source, link) in labels for link in ranked[:TOP_K]) / TOP_K)
        scores[name] = per_source

    no_relevant = sum(not relevant[source] for source in sources)
    print(f"{len(sources)} labelled sources, {len(labels)} labels; {no_relevant} sources have no relevant candidate.\n")
    print(f"| system | P@1 | P@5 | nDCG@5 | judged@5 | P@5 vs {BASELINE} (95% CI) | median ms |")
    print("|---|---|---|---|---|---|---|")
    for name in SYSTEMS:
        s = scores[name]
        mean = {metric: statistics.mean(values) for metric, values in s.items()}
        if name == BASELINE:
            versus = "baseline"
        else:
            diff, low, high = paired_bootstrap(s["p5"], scores[BASELINE]["p5"])
            versus = f"{diff:+.3f} ({low:+.3f}, {high:+.3f})"
        ms = statistics.median(systems.seconds[name]) * 1000
        print(f"| {name} | {mean['p1']:.3f} | {mean['p5']:.3f} | {mean['ndcg5']:.3f} | {mean['judged5']:.2f} | {versus} | {ms:.1f} |")

    print()
    for name in SYSTEMS:
        print(f"- {name}: {DESCRIPTIONS[name]}")
    stage1, rr = statistics.mean(scores["stage1"]["p5"]), statistics.mean(scores[BASELINE]["p5"])
    if stage1:
        print(f"\nRerank lift over stage 1 on P@5: {rr / stage1 - 1:+.1%} ({stage1:.3f} -> {rr:.3f})")


if __name__ == "__main__":
    main()
