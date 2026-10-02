import asyncio
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select

from app.db.models import Article
from app.db.session import AsyncSessionLocal
from app.scrapping import scraper
from app.scrapping.scraper import RSS_ENDPOINTS, parse_rss_items

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss><channel>
  <item>
    <title>Rates rise again</title>
    <link>https://www.theguardian.com/business/a</link>
    <pubDate>Thu, 01 Oct 2026 10:00:00 GMT</pubDate>
    <description>&lt;p&gt;The central bank acted.&lt;/p&gt;</description>
    <category>Business</category>
    <category>Economics</category>
  </item>
  <item>
    <link>https://www.theguardian.com/business/b</link>
    <pubDate>not a date</pubDate>
  </item>
</channel></rss>"""


def test_each_feed_is_fetched_once():
    duplicates = sorted({e for e in RSS_ENDPOINTS if RSS_ENDPOINTS.count(e) > 1})

    assert duplicates == []


def test_parse_rss_items_reads_each_field():
    first = parse_rss_items(BeautifulSoup(RSS, "lxml-xml"))[0]

    assert first["title"] == "Rates rise again"
    assert first["link"] == "https://www.theguardian.com/business/a"
    assert first["pub_date"] == datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
    assert first["description"] == "<p>The central bank acted.</p>"
    assert first["categories"] == ["Business", "Economics"]


def test_parse_rss_items_falls_back_when_fields_are_missing_or_invalid():
    second = parse_rss_items(BeautifulSoup(RSS, "lxml-xml"))[1]

    assert second["title"] == "No Title"
    assert second["pub_date"] is None
    assert second["description"] == "No Description"
    assert second["categories"] == []


def feed_xml(endpoint: str) -> str:
    return f"""<?xml version="1.0"?><rss><channel><item>
      <title>{endpoint}</title><link>https://www.theguardian.com/{endpoint}/1</link>
      <pubDate>Thu, 01 Oct 2026 10:00:00 GMT</pubDate><description>text</description>
    </item></channel></rss>"""


def test_feeds_are_fetched_concurrently_up_to_the_cap_and_failed_feeds_are_skipped():
    endpoints = [f"feed-{i}" for i in range(20)] + ["broken", "timing-out"]
    in_flight, peak = 0, 0

    async def handler(request):
        nonlocal in_flight, peak
        endpoint = request.url.path.strip("/").removesuffix("/rss")
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.01)
            if endpoint == "broken":
                return httpx.Response(500)
            if endpoint == "timing-out":
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(200, text=feed_xml(endpoint))
        finally:
            in_flight -= 1

    async def fetch():
        async with scraper.create_client(httpx.MockTransport(handler)) as client:
            return await scraper.fetch_all_feeds(endpoints, client)

    articles = asyncio.run(fetch())

    assert sorted(a["title"] for a in articles) == sorted(f"feed-{i}" for i in range(20))
    assert peak == scraper.FETCH_CONCURRENCY


def test_a_feed_that_redirects_to_an_edition_url_is_followed():
    def handler(request):
        if request.url.path == "/business/rss":
            return httpx.Response(302, headers={"Location": "https://www.theguardian.com/uk/business/rss"})
        if request.url.path == "/uk/business/rss":
            return httpx.Response(200, text=feed_xml("uk/business"))
        return httpx.Response(404)

    async def fetch():
        async with scraper.create_client(httpx.MockTransport(handler)) as client:
            return await scraper.fetch_all_feeds(["business"], client)

    assert [a["title"] for a in asyncio.run(fetch())] == ["uk/business"]


def item(slug: str, pub_date: datetime | None = datetime(2026, 10, 1, tzinfo=timezone.utc)) -> dict:
    return {
        "title": f"Title {slug}",
        "link": f"https://www.theguardian.com/{slug}",
        "pub_date": pub_date,
        "description": "text",
        "categories": ["News"],
    }


def stored_slugs() -> list[str]:
    async def get():
        async with AsyncSessionLocal() as session:
            return (await session.execute(select(Article.link))).scalars().all()

    return sorted(link.removeprefix("https://www.theguardian.com/") for link in asyncio.run(get()))


def test_storing_skips_links_already_stored_or_repeated_in_the_batch(db):
    asyncio.run(scraper.store_articles_in_db([item("a")]))

    inserted = asyncio.run(scraper.store_articles_in_db([item("a"), item("b"), item("b"), item("c")]))

    assert inserted == 2
    assert stored_slugs() == ["a", "b", "c"]


def test_an_item_without_a_publication_date_is_skipped_instead_of_failing_the_batch(db):
    inserted = asyncio.run(scraper.store_articles_in_db([item("dated"), item("undated", pub_date=None)]))

    assert inserted == 1
    assert stored_slugs() == ["dated"]
