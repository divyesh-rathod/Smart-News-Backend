"""
Time the like path on the local corpus:  python -m eval.bench_latency [--sources 40] [--concurrency 8]

Reports cross-encoder load time and peak process memory, then per like: stage 1 + DB vs rerank time
(median, p95), the longest event-loop stall, throughput under concurrent likes, a cache miss vs hit, and
SBERT embedding throughput (ingest).
"""

import argparse
import asyncio
import resource
import statistics
import sys
import time

from sqlalchemy import select

from app.config import settings
from app.db.models import ProcessedArticle
from app.db.session import AsyncSessionLocal
from app.ml_models import rerank, retrieve
from app.services import news_services, recommendation_cache
from app.utils import sbert_helper


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, KiB on Linux


def summary(values_ms: list[float]) -> str:
    return f"median {statistics.median(values_ms):6.1f} ms   p95 {percentile(values_ms, 95):6.1f} ms   max {max(values_ms):6.1f} ms"


async def record_stalls(stop: asyncio.Event, stalls_ms: list[float], tick: float = 0.005) -> None:
    while not stop.is_set():
        start = time.perf_counter()
        await asyncio.sleep(tick)
        stalls_ms.append((time.perf_counter() - start - tick) * 1000)


async def run(sources: int, concurrency: int) -> None:
    start = time.perf_counter()
    rerank.get_model()
    print(f"cross-encoder load: {time.perf_counter() - start:.1f} s on {settings.MODEL_DEVICE}, "
          f"{rerank.torch.get_num_threads()} torch threads, peak memory {peak_rss_mb():.0f} MB")

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ProcessedArticle.article_id, ProcessedArticle.category_2)
            .where(ProcessedArticle.embedding.is_not(None))
            .order_by(ProcessedArticle.article_id)
        )
        rows = result.all()
    ids = [str(article_id) for article_id, _ in rows[:sources]]
    await retrieve.main(ids[0])  # warm-up: first-call allocations and the DB connection

    rerank_ms: list[float] = []
    original_rerank = retrieve.rerank_top_k

    async def timed_rerank(*args, **kwargs):
        t = time.perf_counter()
        try:
            return await original_rerank(*args, **kwargs)
        finally:
            rerank_ms.append((time.perf_counter() - t) * 1000)

    retrieve.rerank_top_k = timed_rerank
    try:
        total_ms, stalls_ms, stop = [], [], asyncio.Event()
        watcher = asyncio.create_task(record_stalls(stop, stalls_ms))
        for article_id in ids:
            t = time.perf_counter()
            await retrieve.main(article_id)
            total_ms.append((time.perf_counter() - t) * 1000)
        stop.set()
        await watcher
    finally:
        retrieve.rerank_top_k = original_rerank

    print(f"\n{len(ids)} likes one at a time")
    print(f"  total            {summary(total_ms)}")
    print(f"  stage 1 + DB     {summary([t - r for t, r in zip(total_ms, rerank_ms)])}")
    print(f"  rerank           {summary(rerank_ms)}")
    print(f"  event-loop stall median {statistics.median(stalls_ms):.1f} ms, max {max(stalls_ms):.1f} ms")

    semaphore, concurrent_ms = asyncio.Semaphore(concurrency), []

    async def one(article_id: str) -> None:
        async with semaphore:
            t = time.perf_counter()
            await retrieve.main(article_id)
            concurrent_ms.append((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    await asyncio.gather(*(one(article_id) for article_id in ids))
    wall = time.perf_counter() - t
    print(f"\n{len(ids)} likes, {concurrency} at a time: {len(ids) / wall:.1f} likes/s, per like {summary(concurrent_ms)}")

    recommendation_cache.clear()
    timings = []
    for _ in range(2):
        t = time.perf_counter()
        await news_services.recommend_similar_articles(ids[0])
        timings.append((time.perf_counter() - t) * 1000)
    print(f"\nrecommendations for one article: cache miss {timings[0]:.0f} ms, cache hit {timings[1]:.2f} ms")

    texts = [text for _, text in rows[:256] if text]
    t = time.perf_counter()
    model = sbert_helper.get_model()
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    model.encode(texts, normalize_embeddings=True, batch_size=64)
    print(f"\nSBERT: load {load_s:.1f} s, {len(texts) / (time.perf_counter() - t):.0f} embeddings/s "
          f"(batch 64, {len(texts)} real article inputs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--sources", type=int, default=40, help="likes to time (default 40)")
    parser.add_argument("--concurrency", type=int, default=8, help="likes in flight at once (default 8)")
    args = parser.parse_args()
    asyncio.run(run(args.sources, args.concurrency))


if __name__ == "__main__":
    main()
