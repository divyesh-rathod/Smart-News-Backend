"""
Label candidate recommendations by hand:  python -m eval.label [--sources 30]

For each sampled source article it shows the union of every system's top 5, shuffled and without system
names, and asks whether each candidate is relevant. Every answer is appended to eval/labels.jsonl at once,
so you can quit at any point and resume later; pairs that already have a label are skipped.
"""

import argparse
import textwrap

from eval.corpus import load_corpus
from eval.labels import append_label, choose_sources, pool, read_labels
from eval.systems import SYSTEMS, Systems

GUIDE = """
Would someone who just liked the SOURCE article plausibly want this CANDIDATE recommended next?

  y  relevant: the same story or event, or the same specific topic (both about UK interest rates,
     both about the same football club or tournament, both about one country's election)
  n  not relevant: only the same broad section (both "sport", both "business") or unrelated
  s  skip (unsure; asked again next time)    b  back to the previous candidate    q  save and quit
"""


def show(doc, heading: str) -> None:
    body = doc.text if len(doc.text) <= 450 else doc.text[:450] + "..."
    print(f"\n{heading}: {doc.title}")
    print(textwrap.fill(body, width=100, initial_indent="    ", subsequent_indent="    "))
    print(f"    {doc.link}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--sources", type=int, default=30, help="how many source articles to label (default 30)")
    args = parser.parse_args()

    print("Loading the corpus and models...")
    corpus = load_corpus()
    systems = Systems(corpus)
    labels = read_labels()
    sources = choose_sources([doc.link for doc in corpus.docs], labels, args.sources)
    print(GUIDE)

    labelled_now = 0
    for number, source in enumerate(sources, 1):
        source_row = corpus.row[source]
        candidates = pool(source, {name: systems.links(name, source_row) for name in SYSTEMS})
        todo = [link for link in candidates if (source, link) not in labels]
        if not todo:
            continue
        print("\n" + "=" * 100)
        show(corpus.docs[source_row], f"SOURCE {number}/{len(sources)}")
        i = 0
        while i < len(todo):
            show(corpus.docs[corpus.row[todo[i]]], f"  candidate {i + 1}/{len(todo)}")
            answer = input("  relevant? [y/n/s/b/q] > ").strip().lower()
            if answer == "q":
                print(f"\nSaved. {labelled_now} labels this session, {len(labels)} in total.")
                return
            if answer == "b":
                i = max(0, i - 1)
            elif answer == "s":
                i += 1
            elif answer in ("y", "n"):
                append_label(source, todo[i], answer == "y")
                labels[(source, todo[i])] = answer == "y"
                labelled_now += 1
                i += 1
            else:
                print("  Type y, n, s, b or q.")

    print(f"\nDone: every candidate for {len(sources)} sources is labelled ({len(labels)} labels in total).")


if __name__ == "__main__":
    main()
