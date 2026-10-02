import threading

from cachetools import TTLCache

from app.config import settings

# (top5, similar) per article id. TTLCache isn't thread-safe, and clear() runs on the background-task
# thread that finishes a pipeline run, so every access takes the lock.
_cache: TTLCache[str, tuple[list[dict], list[dict]]] = TTLCache(
    maxsize=settings.RECOMMENDATION_CACHE_SIZE,
    ttl=settings.RECOMMENDATION_CACHE_TTL_SECONDS,
)
_lock = threading.Lock()


def get(article_id: str) -> tuple[list[dict], list[dict]] | None:
    with _lock:
        return _cache.get(article_id)


def put(article_id: str, recommendations: tuple[list[dict], list[dict]]) -> None:
    with _lock:
        _cache[article_id] = recommendations


def clear() -> None:
    with _lock:
        _cache.clear()
