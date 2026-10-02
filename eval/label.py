"""
Label candidate recommendations by hand:  python -m eval.label [--sources 30]

For each sampled source article it shows the union of every system's top 5, shuffled and without system
names, and asks whether each candidate is relevant. Every answer is appended to eval/labels.jsonl at once,
so you can quit at any point and resume later; pairs that already have a label are skipped.
"""

import argparse
import textwrap
from concurrent.futures import ThreadPoolExecutor

from transformers.utils import logging as transformers_logging

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

    transformers_logging.disable_progress_bar()
    print("Loading the corpus and models (about 15 s)...")
    corpus = load_corpus()
    systems = Systems(corpus)
    labels = read_labels()
    sources = choose_sources([doc.link for doc in corpus.docs], labels, args.sources)

    def candidates_for(source: str) -> list[str]:
        return pool(source, {name: systems.links(name, corpus.row[source]) for name in SYSTEMS})

    labelled_now: set[tuple[str, str]] = set()
    # Ranking one source with all 9 systems takes ~4 s, so the next source's pool is built while you label.
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        upcoming = prefetch.submit(candidates_for, sources[0])
        upcoming.result()  # the first load also builds the int8 model and re-embeds the corpus once
        print(GUIDE)
        for number, source in enumerate(sources, 1):
            candidates = upcoming.result()
            if number < len(sources):
                upcoming = prefetch.submit(candidates_for, sources[number])
            todo = [link for link in candidates if (source, link) not in labels]
            if not todo:
                continue
            print("\n" + "=" * 100)
            show(corpus.docs[corpus.row[source]], f"SOURCE {number}/{len(sources)}")
            i = 0
            while i < len(todo):
                show(corpus.docs[corpus.row[todo[i]]], f"  candidate {i + 1}/{len(todo)}")
                answer = input("  relevant? [y/n/s/b/q] > ").strip().lower()
                if answer == "q":
                    print(f"\nSaved. {len(labelled_now)} labels this session, {len(labels)} in total.")
                    return
                if answer == "b":
                    i = max(0, i - 1)
                elif answer == "s":
                    i += 1
                elif answer in ("y", "n"):
                    append_label(source, todo[i], answer == "y")
                    labels[(source, todo[i])] = answer == "y"
                    labelled_now.add((source, todo[i]))
                    i += 1
                else:
                    print("  Type y, n, s, b or q.")

    print(f"\nDone: every candidate for {len(sources)} sources is labelled ({len(labels)} labels in total).")


if __name__ == "__main__":
    main()
