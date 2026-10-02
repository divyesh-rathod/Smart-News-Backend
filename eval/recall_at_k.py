"""
HNSW against an exact scan:  python -m eval.recall_at_k [--sources 200] [--scale-dsn DSN [--sizes 10000,50000,100000]]

Part 1 runs the app's stage-1 query on the local corpus with the HNSW index forced, for several
hnsw.ef_search values, against an exact scan: rows returned, recall@50 and median query time.

Part 2 needs an empty scratch database migrated with `alembic upgrade head`. It fills it with synthetic
articles (a real embedding plus Gaussian noise, so the data stays clustered like real news) at growing
sizes. At each size it rebuilds the index and reports the build time, whether Postgres picks the index
without being forced, and exact vs HNSW query time and recall@50 at the app's ef_search.
"""

import argparse
import asyncio
import statistics
import time
import uuid
from datetime import datetime, timezone

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from sqlalchemy.dialects.postgresql import asyncpg as asyncpg_dialect
from sqlalchemy.schema import CreateIndex

from app.config import settings
from app.db.models import ProcessedArticle
from app.ml_models.retrieve import HNSW_EF_SEARCH, STAGE1_LIMIT, build_stage1_query
from eval.corpus import load_corpus
from eval.metrics import recall_at_k

INDEX_NAME = "ix_processed_articles_embedding_hnsw"
EF_SEARCH_VALUES = (40, 64, 100, 200)


def compile_stage1(source_id: uuid.UUID, embedding) -> tuple[str, list]:
    compiled = build_stage1_query(source_id, embedding).compile(dialect=asyncpg_dialect.dialect())
    return str(compiled), [compiled.params[name] for name in compiled.positiontup]


async def run_queries(conn: asyncpg.Connection, queries: list[tuple[str, list]]) -> tuple[list[list], list[float]]:
    results, times_ms = [], []
    for sql, params in queries:
        start = time.perf_counter()
        rows = await conn.fetch(sql, *params)
        times_ms.append((time.perf_counter() - start) * 1000)
        results.append([row["article_id"] for row in rows])
    return results, times_ms


async def exact_and_hnsw(conn: asyncpg.Connection, queries, ef_values) -> list[tuple[str, float, float, float, float]]:
    """(label, mean rows returned, mean recall@50, min recall@50, median ms) for the exact scan and each ef_search."""
    async with conn.transaction():
        await conn.execute("SET LOCAL enable_indexscan = off")
        exact, exact_ms = await run_queries(conn, queries)
    rows = [("exact scan", statistics.mean(map(len, exact)), 1.0, 1.0, statistics.median(exact_ms))]
    for ef in ef_values:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL enable_seqscan = off; SET LOCAL enable_sort = off; SET LOCAL hnsw.ef_search = {ef}")
            approx, approx_ms = await run_queries(conn, queries)
        recalls = [recall_at_k(a, e, STAGE1_LIMIT) for a, e in zip(approx, exact)]
        rows.append((f"hnsw ef_search={ef}", statistics.mean(map(len, approx)), statistics.mean(recalls), min(recalls), statistics.median(approx_ms)))
    return rows


def print_table(rows) -> None:
    print("| search | rows returned | recall@50 mean | recall@50 min | median ms |")
    print("|---|---|---|---|---|")
    for label, n, mean, low, ms in rows:
        print(f"| {label} | {n:.1f} | {mean:.3f} | {low:.2f} | {ms:.2f} |")


async def live_corpus(sources: int) -> None:
    conn = await asyncpg.connect(settings.DATABASE_URL.replace("+asyncpg", ""))
    await register_vector(conn)
    try:
        rows = await conn.fetch(
            "SELECT article_id, embedding FROM processed_articles WHERE embedding IS NOT NULL ORDER BY article_id LIMIT $1",
            sources,
        )
        total = await conn.fetchval("SELECT count(*) FROM processed_articles WHERE embedding IS NOT NULL")
        print(f"Live corpus: {total} embedded articles, {len(rows)} source articles\n")
        print_table(await exact_and_hnsw(conn, [compile_stage1(r["article_id"], r["embedding"]) for r in rows], EF_SEARCH_VALUES))
    finally:
        await conn.close()


async def scale(dsn: str, embeddings: np.ndarray, sizes: list[int], queries: int = 100, noise: float = 0.03, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    query_vectors = embeddings[rng.choice(len(embeddings), size=queries, replace=False)]
    stage1 = [compile_stage1(uuid.uuid4(), vector) for vector in query_vectors]
    index = next(i for i in ProcessedArticle.__table__.indexes if i.name == INDEX_NAME)
    create_index = str(CreateIndex(index).compile(dialect=asyncpg_dialect.dialect()))

    conn = await asyncpg.connect(dsn)
    await register_vector(conn)
    try:
        if await conn.fetchval("SELECT count(*) FROM articles"):
            raise SystemExit(f"{dsn} must be an empty database migrated with alembic upgrade head")
        # The default 64 MB can't hold the graph past ~30k rows, which makes the build far slower.
        await conn.execute("SET maintenance_work_mem = '1GB'")
        print(f"\nSynthetic scale test: real embeddings + N(0, {noise}) noise, {queries} real articles as queries\n")
        print("| rows | index build s | planner picks HNSW unforced | exact median ms | HNSW median ms "
              f"(ef_search={HNSW_EF_SEARCH}) | HNSW recall@50 mean / min |")
        print("|---|---|---|---|---|---|")
        current, now = 0, datetime.now(timezone.utc)
        for size in sizes:
            n = size - current
            vectors = embeddings[rng.integers(0, len(embeddings), n)] + rng.normal(0, noise, (n, embeddings.shape[1]))
            vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)
            ids = [uuid.uuid4() for _ in range(n)]
            await conn.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
            await conn.copy_records_to_table(
                "articles",
                records=[(ids[i], f"synthetic {current + i}", f"synthetic://{current + i}", now, True) for i in range(n)],
                columns=["id", "title", "link", "pub_date", "processed"],
            )
            await conn.copy_records_to_table(
                "processed_articles",
                records=[(ids[i], "", vectors[i]) for i in range(n)],
                columns=["article_id", "cleaned_text", "embedding"],
            )
            current = size
            start = time.perf_counter()
            await conn.execute(create_index)
            build_s = time.perf_counter() - start
            await conn.execute("ANALYZE articles; ANALYZE processed_articles")

            sql, params = stage1[0]
            async with conn.transaction():
                await conn.execute(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}")
                plan = "\n".join(r[0] for r in await conn.fetch(f"EXPLAIN {sql}", *params))
            (_, _, _, _, exact_ms), (_, _, mean, low, hnsw_ms) = await exact_and_hnsw(conn, stage1, [HNSW_EF_SEARCH])
            print(f"| {size:,} | {build_s:.1f} | {'yes' if INDEX_NAME in plan else 'no'} | {exact_ms:.2f} | {hnsw_ms:.2f} | {mean:.3f} / {low:.2f} |")
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--sources", type=int, default=200, help="source articles for part 1 (default 200)")
    parser.add_argument("--scale-dsn", help="empty migrated scratch database for part 2, e.g. postgresql://localhost/smart_news_bench")
    parser.add_argument("--sizes", default="10000,50000,100000", help="row counts for part 2 (default 10000,50000,100000)")
    args = parser.parse_args()
    asyncio.run(live_corpus(args.sources))
    if args.scale_dsn:
        embeddings = load_corpus().embeddings
        asyncio.run(scale(args.scale_dsn, embeddings, [int(size) for size in args.sizes.split(",")]))


if __name__ == "__main__":
    main()
