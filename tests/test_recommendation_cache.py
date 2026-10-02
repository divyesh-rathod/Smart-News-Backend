import asyncio
import subprocess
import uuid

import pytest
from cachetools import TTLCache

from app.controller import scripts_controller
from app.services import news_services, recommendation_cache

RECOMMENDATIONS = ([{"article_id": "top"}], [{"article_id": "top"}, {"article_id": "other"}])


@pytest.fixture
def retrieve_calls(monkeypatch):
    """Replace retrieve.main as seen by the service; each call pops the next result (or raises it)."""
    calls, results = [], []

    async def main(article_id):
        calls.append(article_id)
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(news_services, "main", main)
    return calls, results


def recommend(article_id):
    return asyncio.run(news_services.recommend_similar_articles(article_id))


def test_recommendations_are_computed_once_per_article(retrieve_calls):
    calls, results = retrieve_calls
    results += [RECOMMENDATIONS, RECOMMENDATIONS]
    article, other_article = uuid.uuid4(), uuid.uuid4()

    assert recommend(article) == RECOMMENDATIONS
    assert recommend(article) == RECOMMENDATIONS
    recommend(other_article)

    assert calls == [str(article), str(other_article)]


def test_empty_recommendations_are_not_cached(retrieve_calls):
    calls, results = retrieve_calls
    results += [([], []), RECOMMENDATIONS]
    article = uuid.uuid4()

    assert recommend(article) == ([], [])
    assert recommend(article) == RECOMMENDATIONS
    assert len(calls) == 2


def test_failures_are_not_cached(retrieve_calls):
    calls, results = retrieve_calls
    results += [RuntimeError("model crashed"), RECOMMENDATIONS]
    article = uuid.uuid4()

    assert recommend(article) == ([], [])
    assert recommend(article) == RECOMMENDATIONS
    assert len(calls) == 2


def test_cached_recommendations_expire_after_the_ttl(retrieve_calls, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(recommendation_cache, "_cache", TTLCache(maxsize=8, ttl=900, timer=lambda: now[0]))
    calls, results = retrieve_calls
    results += [RECOMMENDATIONS, RECOMMENDATIONS]
    article = uuid.uuid4()

    recommend(article)
    now[0] = 899
    recommend(article)
    assert len(calls) == 1

    now[0] = 901
    recommend(article)
    assert len(calls) == 2


@pytest.mark.parametrize("failing_step", [None, "app.ml_models.generate_embeddings"])
def test_a_pipeline_run_clears_the_cache_even_if_a_step_fails(monkeypatch, failing_step):
    def run_subprocess(module_path):
        if module_path == failing_step:
            raise subprocess.CalledProcessError(1, module_path)

    monkeypatch.setattr(scripts_controller, "run_subprocess", run_subprocess)
    recommendation_cache.put("article", RECOMMENDATIONS)

    if failing_step:
        with pytest.raises(subprocess.CalledProcessError):
            scripts_controller.start_full_pipeline_subprocesses()
    else:
        scripts_controller.start_full_pipeline_subprocesses()

    assert recommendation_cache.get("article") is None
