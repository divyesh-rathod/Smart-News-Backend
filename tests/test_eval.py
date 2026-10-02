import json
import math

import pytest

from eval import labels as label_store
from eval.metrics import ndcg_at_k, paired_bootstrap, precision_at_k, recall_at_k


def test_precision_counts_relevant_items_in_the_top_k():
    assert precision_at_k(["a", "b", "c", "d", "e", "f"], {"a", "c", "f"}, 5) == pytest.approx(0.4)


def test_ndcg_rewards_relevant_items_ranked_higher():
    relevant = {"a", "b"}

    assert ndcg_at_k(["a", "b", "x", "y", "z"], relevant, 5) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y", "z", "w", "v"], relevant, 5) == 0.0
    late = (1 / math.log2(5) + 1 / math.log2(6)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "y", "z", "a", "b"], relevant, 5) == pytest.approx(late)


def test_ndcg_is_zero_when_nothing_is_relevant():
    assert ndcg_at_k(["a", "b"], set(), 5) == 0.0


def test_recall_is_the_share_of_the_exact_top_k_found():
    assert recall_at_k([1, 2, 9], [1, 2, 3, 4], 4) == pytest.approx(0.5)
    assert recall_at_k([], [], 50) == 1.0


def test_paired_bootstrap_of_a_constant_difference_is_that_difference():
    mean, low, high = paired_bootstrap([0.6, 0.8, 0.4], [0.4, 0.6, 0.2], resamples=200)

    assert (mean, low, high) == pytest.approx((0.2, 0.2, 0.2))


def test_paired_bootstrap_interval_contains_the_mean():
    mean, low, high = paired_bootstrap([1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 0, 1], resamples=2000)

    assert low <= mean <= high
    assert mean == pytest.approx(1 / 3)


def test_labels_round_trip_and_a_later_answer_replaces_an_earlier_one(tmp_path):
    path = tmp_path / "labels.jsonl"
    label_store.append_label("src", "a", True, path)
    label_store.append_label("src", "b", False, path)
    label_store.append_label("src", "a", False, path)

    assert label_store.read_labels(path) == {("src", "a"): False, ("src", "b"): False}
    assert set(json.loads(path.read_text().splitlines()[0])) == {"source", "candidate", "relevant", "labelled_at"}


def test_missing_labels_file_means_no_labels(tmp_path):
    assert label_store.read_labels(tmp_path / "absent.jsonl") == {}


def test_source_sample_keeps_labelled_sources_when_new_articles_arrive():
    links = [f"https://example.com/{i}" for i in range(50)]
    first = label_store.choose_sources(links, {}, 5)
    labels = {(source, "candidate"): True for source in first}

    later = label_store.choose_sources(links + [f"https://example.com/new-{i}" for i in range(500)], labels, 5)

    assert sorted(later) == sorted(first)
    assert label_store.choose_sources(links, {}, 5) == first


def test_pool_is_the_union_of_each_top_5_in_a_fixed_shuffled_order():
    rankings = {"a": ["1", "2", "3", "4", "5", "6"], "b": ["5", "7", "1", "8", "9", "10"]}

    candidates = label_store.pool("source", rankings)

    assert sorted(candidates) == ["1", "2", "3", "4", "5", "7", "8", "9"]
    assert candidates == label_store.pool("source", rankings)
