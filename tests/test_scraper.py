from datetime import datetime, timezone

from bs4 import BeautifulSoup

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
