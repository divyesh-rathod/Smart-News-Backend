# Smart News — backend

A FastAPI service that collects news from The Guardian's RSS feeds, embeds every article, and, when a reader likes an article, recommends five similar ones with a two-stage ranker: vector search in PostgreSQL (pgvector), then a cross-encoder rerank.

Frontend: [smart-news-frontend](https://github.com/divyesh-rathod/smart-news-frontend).

## How it works

### Ingest

Three steps, run in order from the CLI or through `POST /api/V1/scripts/run_pipeline`:

1. **Scrape** (`app/scrapping/scraper.py`): fetches 47 Guardian RSS feeds, 8 at a time, with a 10 s timeout and redirects followed. New articles are inserted with `ON CONFLICT (link) DO NOTHING`; items without a valid `pubDate` are skipped.
2. **Preprocess** (`app/preprocessing/preprocess.py`): strips HTML from each description and lowercases it (`cleaned_text`), and builds the embedding input `category_2` = description followed by the categories.
3. **Embed** (`app/ml_models/generate_embeddings.py`, `app/utils/sbert_helper.py`): encodes `category_2` with Sentence-BERT `all-MiniLM-L6-v2` (384 dimensions, L2-normalized) in batches of 100. Articles without text are skipped.

### Recommending on a like

`PUT /api/V1/news/like/{article_id}` with `{"liked": true}` saves the like (an idempotent upsert), then:

1. **Stage 1, retrieval** (`app/ml_models/retrieve.py`): the 50 nearest embedded articles by cosine distance (`<=>`). An HNSW index (`vector_cosine_ops`, `m=16`, `ef_construction=64`) serves the query once the table is large enough for Postgres to prefer it; `hnsw.ef_search` is set to 100 per query, because pgvector's default of 40 would cap an index scan below 50 rows.
2. **Stage 2, rerank** (`app/ml_models/rerank.py`): cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (liked article, candidate) pair of descriptions, up to 512 tokens, in batches of 16 sorted by length. Inference runs on one dedicated thread, so concurrent likes queue instead of competing for the CPU. The top 5 are returned.
3. **Cache** (`app/services/recommendation_cache.py`): results are cached per article in memory (256 entries, 15-minute TTL). A pipeline run through the API clears the cache when it finishes.

Recommendations are best effort: if the article isn't embedded yet, has no embedded neighbours, or ranking fails (logged), the like is still saved and the lists come back empty.

The cross-encoder is loaded at startup (FastAPI lifespan); SBERT is only loaded by the ingest step.

## Measured performance

Measured on an Apple Silicon Mac (10 cores, CPU only, 8 torch threads) with a corpus of 1,123 articles, unless noted. Reproduce with the scripts in `eval/`.

| What | Result |
|---|---|
| Like with recommendations, median / p95 | 539 ms / 769 ms (cross-encoder 526 ms, stage 1 + DB 13 ms) |
| Like whose recommendations are cached | 2.9 ms over HTTP |
| First like after startup | 608 ms (the model loads during startup, 1.5–2 s) |
| 8 concurrent likes | 1.9 likes/s |
| Stage-1 vector search, 1,123 articles | 0.7 ms with HNSW, 1.9 ms exact scan |
| Stage-1 vector search, 100,000 synthetic articles | 0.9 ms with HNSW, 30 ms exact scan; HNSW recall@50 0.998 |
| Size at which Postgres starts using the index | between 3,000 and 5,000 articles |
| Peak memory of the API process | 442 MB |
| SBERT embedding (ingest) | 228 articles/s |
| Scraping all 47 feeds | 2.9 s (~1,300 items, network-bound) |
| Rerank quality vs stage 1 alone (P@5) | not measured yet: needs the hand labels in `eval/labels.jsonl` |

The like latency is almost all cross-encoder inference on CPU; a GPU (`MODEL_DEVICE=cuda` or `mps`) or a smaller candidate set would cut it.

## API

All routes are under `/api/V1`. News and user routes need `Authorization: Bearer <token>` from signup or login.

| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/auth/signup` | `{name, email, phone_number, password, profile_picture?}` | returns `{user, access_token}` |
| POST | `/auth/login` | `{email, password}` | returns `{user, access_token}` |
| PUT | `/users/update` | `{name?, phone_number?, profile_picture?}` | |
| GET | `/news/unseen-articles?limit=20&cursor=...` | | unread articles, newest first, each with `liked`; pass the previous page's `next_cursor` for the next page (`null` on the last) |
| POST | `/news/mark-as-read/{article_id}` | | idempotent |
| PUT | `/news/like/{article_id}` | `{liked: true\|false}` | idempotent; returns `{message, liked, top5, similar}` |
| POST | `/news/set-date?last_read_date=...` | | defaults to now (UTC) |
| POST | `/scripts/run_pipeline` | | needs header `X-Admin-Token: <ADMIN_API_KEY>`; 503 if no key is configured |

Interactive docs: `http://localhost:8000/docs`. Call `/auth/login` there, then paste the returned `access_token` into Authorize (HTTPBearer); for the pipeline endpoint, also fill in `X-Admin-Token`. From the command line:

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/V1/auth/login -H 'Content-Type: application/json' \
  -d '{"email": "you@example.com", "password": "..."}' | python -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
curl -s -X PUT localhost:8000/api/V1/news/like/<article_id> -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"liked": true}'
```

## Setup

Requirements: Python 3.12, PostgreSQL with the pgvector extension (tested with PostgreSQL 17 and pgvector 0.8).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
# On Linux, add --extra-index-url https://download.pytorch.org/whl/cpu to get the CPU build of torch.
```

Create the database, and the extension inside it. pgvector isn't a trusted extension, so a superuser has to create it (or run the migrations):

```bash
psql -d postgres -c "CREATE ROLE smart_news LOGIN PASSWORD 'smart_news';"
createdb -O smart_news smart_news
psql -d smart_news -c "CREATE EXTENSION IF NOT EXISTS vector;"   # as a superuser
```

Configure, migrate, ingest and run:

```bash
cp .env.example .env   # then set DATABASE_URL, SECRET_KEY (32+ characters) and, to enable the pipeline endpoint, ADMIN_API_KEY
alembic upgrade head
python -m app.scrapping.scraper
python -m app.preprocessing.preprocess
python -m app.ml_models.generate_embeddings
uvicorn app.main:app --port 8000
```

The first run downloads both models from Hugging Face (about 90 MB each).

## Configuration

Read from `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | required | `postgresql+asyncpg://user:password@host/db` |
| `SECRET_KEY` | none usable | JWT signing key. The API refuses to start with a placeholder or a key shorter than 32 characters |
| `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_DAYS` | `HS256`, `15` | JWT settings |
| `ADMIN_API_KEY` | unset | shared secret for `/scripts/run_pipeline`; unset disables it |
| `DEBUG` | `False` | echo SQL statements |
| `LOG_LEVEL` | `INFO` | app log level under uvicorn |
| `MODEL_DEVICE` | `cpu` | torch device for the cross-encoder (`cpu`, `cuda`, `mps`) |
| `TORCH_NUM_THREADS` | torch's default | CPU threads for inference; lower it when several workers share a machine |
| `RECOMMENDATION_CACHE_SIZE`, `RECOMMENDATION_CACHE_TTL_SECONDS` | `256`, `900` | per-process recommendation cache |

## Tests and CI

```bash
ruff check .
pytest -m "not model"   # what CI runs; DB tests are skipped without TEST_DATABASE_URL
TEST_DATABASE_URL=postgresql+asyncpg://user@localhost/smart_news_test pytest   # everything
```

- Tests that use the `db` fixture need a migrated, disposable database at `TEST_DATABASE_URL`; they empty every table. Create it with `DATABASE_URL=<that url> alembic upgrade head`.
- Tests marked `model` load the real models (downloads on first run), so CI skips them.
- 72 tests in total (68 in CI). They cover retrieval (cosine ordering, NULL embeddings, the HNSW `ef_search` limit), likes (idempotency, concurrent first likes, ranking failures), feed paging (no gaps or repeats, reloads), the cache, the scraper, the embedding loop, auth (tokens, the pipeline secret, `SECRET_KEY` checks), and startup.
- GitHub Actions runs ruff, `alembic upgrade head` and `alembic check` against a `pgvector/pgvector:0.8.7-pg17` service container, then `pytest -m "not model"`.

## Evaluation

| Command | What it measures |
|---|---|
| `python -m eval.bench_latency` | model load, memory, per-stage like latency, concurrency, cache hit, SBERT rate |
| `python -m eval.recall_at_k [--scale-dsn DSN]` | HNSW vs exact recall@50 by `ef_search`; with a scratch database, index build time, planner choice and latency up to 100k rows |
| `python -m eval.label` | interactive labelling of each system's top 5 (pooled, shuffled, anonymous) into `eval/labels.jsonl` |
| `python -m eval.rerank_quality` | P@1, P@5, nDCG@5 per system on the labelled sources, with paired bootstrap intervals |

`eval/systems.py` reproduces the production stage 1 and rerank, plus variants that each change one thing: categories first in the embedding input, the title as the rerank query, a capped query, boilerplate removed, a 256-token limit, 20 candidates instead of 50, and int8 quantization.

## Project layout

```
app/
  api/V1/         routes (auth, users, news, scripts)
  controller/     HTTP error mapping
  services/       likes, feed, recommendation cache
  ml_models/      retrieve.py (stage 1), rerank.py (stage 2), generate_embeddings.py
  scrapping/      RSS scraper
  preprocessing/  text cleaning
  db/             SQLAlchemy models and async session
migrations/       Alembic revisions
eval/             benchmarks and the rerank evaluation
tests/            pytest suite
```

## Known limitations

- **Recommendations aren't personalised beyond the liked article.** They're cached per article, not per user, and may include articles the user has already read.
- **Truncation.** About 10% of (query, candidate) pairs exceed the cross-encoder's 512 tokens, and SBERT's 256-token limit cuts 17% of embedding inputs, dropping the categories at their end. The variants in `eval/` exist to measure fixes before changing rankings.
- **Auth is minimal.** The pipeline endpoint uses one shared secret, there's no token refresh, and passlib is unmaintained.
- **The cache is per process.** Several uvicorn workers each keep their own, and a pipeline run clears only the cache of the worker that ran it; after a CLI ingest, recommendations can be up to 15 minutes stale.

## License

MIT, see [LICENSE](LICENSE).
