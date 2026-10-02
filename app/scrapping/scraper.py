# app/scrapping/scraper.py

import asyncio
import logging

import httpx
from bs4 import BeautifulSoup
from sqlalchemy.dialects.postgresql import insert
from email.utils import parsedate_to_datetime
from app.db.session import AsyncSessionLocal
from app.db.models.article import Article

logger = logging.getLogger(__name__)

FETCH_CONCURRENCY = 8  # feeds requested at once, to stay polite to the Guardian
FETCH_TIMEOUT_SECONDS = 10

RSS_ENDPOINTS = [
    "international", "football", "politics", "global-development",
    "world/ukraine", "world/asia", "us-news", "australia-news",
    "technology", "business","sport/tennis","sport/formulaone","uk/lifeandstyle",
    "fashion","food","tone/recipes","lifeandstyle/relationships",
    "lifeandstyle/health-and-wellbeing","lifeandstyle/women","lifeandstyle/men",
    "uk/travel","travel/usa","travel/europe","science","books","uk/film","games",
    "music/classical-music-and-opera","sport/cricket","uk/environment",
    "environment/climate-crisis","environment/wildlife","environment/energy",
    "environment/pollution","tone/obituaries","uk/business",
    "business/economics","business/banking","uk/money","money/savings",
    "money/property","money/work-and-careers","money/debt","business/stock-markets",
    "business/series/project-syndicate-economists","uk/business-to-business",
    "business/retail"
]

async def fetch_rss_feed(client: httpx.AsyncClient, url: str) -> BeautifulSoup:
    resp = await client.get(url)
    if resp.status_code == 200:
        return BeautifulSoup(resp.content, "lxml-xml")
    else:
        raise Exception(f"Failed to fetch RSS feed: {resp.status_code} for URL: {url}")


async def fetch_all_feeds(endpoints: list[str], client: httpx.AsyncClient) -> list[dict]:
    """Fetch and parse every feed, FETCH_CONCURRENCY at a time. A feed that fails is logged and skipped."""
    semaphore = asyncio.Semaphore(FETCH_CONCURRENCY)

    async def fetch(endpoint: str) -> list[dict]:
        url = f"https://www.theguardian.com/{endpoint}/rss"
        async with semaphore:
            try:
                soup = await fetch_rss_feed(client, url)
            except Exception as e:
                logger.warning("Error fetching feed from %s: %s", url, e)
                return []
        return parse_rss_items(soup)

    feeds = await asyncio.gather(*(fetch(endpoint) for endpoint in endpoints))
    return [article for feed in feeds for article in feed]

def parse_rss_items(soup: BeautifulSoup) -> list[dict]:
    items = soup.find_all("item")
    articles = []
    for item in items:
        title = item.find("title").text if item.find("title") else "No Title"
        link = item.find("link").text if item.find("link") else "No Link"
        raw_pub = item.find("pubDate").text if item.find("pubDate") else None
        if raw_pub:
            try:
                pub_date = parsedate_to_datetime(raw_pub)
            except (TypeError, ValueError):
                # fallback if parsing fails
                pub_date = None
        else:
            pub_date = None
        description = item.find("description").text if item.find("description") else "No Description"
        categories = [cat.text.strip() for cat in item.find_all("category")]
        articles.append({
            "title": title,
            "link": link,
            "pub_date": pub_date,
            "description": description,
            "categories": categories,
        })
    return articles

async def store_articles_in_db(articles: list[dict]) -> int:
    """Insert the articles whose link isn't stored yet. Returns how many were inserted."""
    # pub_date is NOT NULL: one undated item used to fail the whole batch.
    dated = [art for art in articles if art["pub_date"] is not None]
    if len(dated) < len(articles):
        logger.warning("Skipping %d items without a valid pubDate", len(articles) - len(dated))
    if not dated:
        return 0

    rows = [
        {
            "title": art["title"],
            "link": art["link"],
            "pub_date": art["pub_date"],
            "description": art["description"],
            "categories": art.get("categories", []),
        }
        for art in dated
    ]
    # The unique link constraint skips both stored links and links repeated across feeds in this batch.
    stmt = insert(Article).on_conflict_do_nothing(index_elements=[Article.link]).returning(Article.id)
    async with AsyncSessionLocal() as session:
        result = await session.execute(stmt, rows)
        inserted = len(result.all())
        await session.commit()

    logger.info("Stored %d new articles, skipped %d duplicates", inserted, len(dated) - inserted)
    return inserted

async def main():
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS) as client:
        all_articles = await fetch_all_feeds(RSS_ENDPOINTS, client)

    await store_articles_in_db(all_articles)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
