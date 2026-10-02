import asyncio
import random

import pytest
import torch

from app.ml_models.rerank import get_model, rerank_top_k, score_pairs


def make_candidate(article_id: str, text: str | None) -> dict:
    return {
        "article_id": article_id,
        "cleaned_text": text,
        "category_1": "Business",
        "category_2": None,
        "title": f"Title {article_id}",
        "link": f"https://example.com/{article_id}",
        "distance": 0.5,
    }


class FakeEncoding(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """Records each batch; the 'encoding' is just the candidate text lengths."""

    def __init__(self):
        self.batches = []

    def __call__(self, pairs, **kwargs):
        self.batches.append(pairs)
        return FakeEncoding(lengths=torch.tensor([[float(len(text))] for _, text in pairs]))


class FakeModel:
    """Scores a pair by the length of its candidate text."""

    device = "cpu"

    def __call__(self, lengths):
        return type("Output", (), {"logits": lengths})()


def test_score_pairs_keeps_input_order_across_length_sorted_batches():
    lengths = list(range(1, 41))
    random.Random(0).shuffle(lengths)
    pairs = [["query", "x" * n] for n in lengths]
    tokenizer = FakeTokenizer()

    scores = score_pairs(tokenizer, FakeModel(), pairs, batch_size=16)

    assert scores == [float(n) for n in lengths]
    assert [len(batch) for batch in tokenizer.batches] == [16, 16, 8]
    batch_lengths = [[len(text) for _, text in batch] for batch in tokenizer.batches]
    assert batch_lengths == [list(range(1, 17)), list(range(17, 33)), list(range(33, 41))]


@pytest.mark.model
def test_length_sorted_batches_score_the_same_as_one_batch():
    tokenizer, model = get_model()
    texts = ["rates rose", "a recipe for lemon cake " * 30, "the striker signed", "inflation " * 120] * 5
    pairs = [["the central bank raised interest rates", text] for text in texts]

    batched = score_pairs(tokenizer, model, pairs, batch_size=4)
    single = score_pairs(tokenizer, model, pairs, batch_size=len(pairs))

    assert batched == pytest.approx(single, abs=1e-4)


@pytest.mark.model
def test_rerank_ranks_the_matching_article_first_and_keeps_metadata():
    query = "the central bank raised interest rates to fight inflation"
    candidates = [
        make_candidate("cake", "a recipe for lemon cake with vanilla icing"),
        make_candidate("rates", "the central bank raised interest rates again to tackle inflation"),
        make_candidate("football", "the football club signed a new striker on a free transfer"),
    ]

    result = asyncio.run(rerank_top_k(query, candidates, top_n=2))

    assert len(result) == 2
    assert result[0]["article_id"] == "rates"
    assert result[0]["score"] >= result[1]["score"]
    assert result[0]["title"] == "Title rates"
    assert result[0]["link"] == "https://example.com/rates"
    assert set(result[0]) == {
        "article_id", "cleaned_text", "category_1", "category_2", "title", "link", "score",
    }


@pytest.mark.model
def test_rerank_treats_missing_text_as_empty_string():
    candidates = [make_candidate("empty", None), make_candidate("full", "interest rates rose")]

    result = asyncio.run(rerank_top_k("interest rates", candidates, top_n=5))

    assert [r["article_id"] for r in result] == ["full", "empty"]
    assert result[1]["cleaned_text"] is None
