import asyncio

import pytest

from app.ml_models.rerank import rerank_top_k

pytestmark = pytest.mark.model


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


def test_rerank_treats_missing_text_as_empty_string():
    candidates = [make_candidate("empty", None), make_candidate("full", "interest rates rose")]

    result = asyncio.run(rerank_top_k("interest rates", candidates, top_n=5))

    assert [r["article_id"] for r in result] == ["full", "empty"]
    assert result[1]["cleaned_text"] is None
