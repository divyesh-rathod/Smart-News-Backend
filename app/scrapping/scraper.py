# app/scrapping/scraper.py

import asyncio
import logging

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select
from email.utils import parsedate_to_datetime
from app.db.session import AsyncSessionLocal
from app.db.models.article import Article

logger = logging.getLogger(__name__)

RSS_ENDPOINTS = [
    "international", "football", "politics", "global-development",
    "world/ukraine", "world/asia", "us-news", "australia-news",
    "technology", "business","sport/tennis","sport/formulaone","uk/lifeandstyle",
    "fashion","food","tone/recipes","lifeandstyle/relationships",
    "lifeandstyle/health-and-wellbeing","lifeandstyle/women","lifeandstyle/men",
    "uk/travel","travel/usa","travel/europe","science","books","uk/film","games",
    "music/classical-music-and-opera","sport/cricket","uk/environment",
    "environment/climate-crisis","environment/wildlife","environment/energy",
    "environment/pollution","global-development","tone/obituaries","uk/business",
    "business/economics","business/banking","uk/money","money/savings",
    "money/property","money/work-and-careers","money/debt","business/stock-markets",
    "business/series/project-syndicate-economists","uk/business-to-business",
    "business/retail"
]

async def fetch_rss_feed(url: str) -> BeautifulSoup:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            return BeautifulSoup(resp.content, "lxml-xml")
        else:
            raise Exception(f"Failed to fetch RSS feed: {resp.status_code} for URL: {url}")

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

async def store_articles_in_db(articles: list[dict]):
    async with AsyncSessionLocal() as session: 
        try:
            # 1) Fetch all existing links from DB
            result = await session.execute(select(Article.link))
            existing_links = {row[0] for row in result.all()}

            skipped = 0
            for art in articles:
                link = art["link"]
                if link in existing_links:
                    skipped += 1
                    continue

                new_article = Article(
                    title=art["title"],
                    link=link,
                    pub_date=art["pub_date"],
                    description=art["description"],
                    categories=art.get("categories", []),
                )
                session.add(new_article)
                existing_links.add(link)

            await session.commit()
            logger.info("Stored %d new articles, skipped %d duplicates", len(articles) - skipped, skipped)
        except Exception:
            await session.rollback()
            logger.exception("Error storing articles")

async def main():
    all_articles: list[dict] = []
    for endpoint in RSS_ENDPOINTS:
        url = f"https://www.theguardian.com/{endpoint}/rss"
        try:
            soup = await fetch_rss_feed(url)
        except Exception as e:
            logger.warning("Error fetching feed from %s: %s", url, e)
            continue
        all_articles.extend(parse_rss_items(soup))

    await store_articles_in_db(all_articles)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
