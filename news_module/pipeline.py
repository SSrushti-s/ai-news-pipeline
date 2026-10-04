import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

import aiohttp
from bs4 import BeautifulSoup
import feedparser
from dateutil import parser as dateparser

from .config import MAX_ARTICLES_PER_SOURCE, RECENCY_HOURS
from .dedupe import DedupeStore
from .http_client import AsyncHttpClient
from .logo_resolver import LogoResolver
from .news_model import Article, Publisher
from .source_health import SourceHealthStore
from .sources import SOURCES
from .util import slugify

logger = logging.getLogger("ai_news_crawler.pipeline")

# Some feeds (notably hnrss) ship link/vote metadata in the <description>
# instead of an actual summary, e.g.:
#   "Comments URL: https://... \n Points: 1 \n # Comments: 0"
# If a cleaned summary looks like it's mostly made of these fields, it's
# not usable as a `dek` and we fall back instead of surfacing garbage.
_META_FIELD_PATTERN = re.compile(r"(?i)\b(article url|comments url|points|#\s*comments)\s*:")


def _parse_published(entry) -> datetime | None:
    for field_name in ("published", "updated", "pubDate"):
        value = entry.get(field_name)
        if value:
            try:
                dt = dateparser.parse(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except (ValueError, TypeError):
                continue
    return None


def _parse_feed_text(feed_text: str):
    return feedparser.parse(feed_text)


def _clean_dek(raw_summary: str, title: str) -> str:
    """Strip HTML and reject feed-metadata masquerading as a summary."""
    text = ""
    if raw_summary:
        soup = BeautifulSoup(raw_summary, "html.parser")
        text = soup.get_text(separator=" ").strip()

    if not text or len(_META_FIELD_PATTERN.findall(text)) >= 2:
        # Empty string is a sentinel: "no usable dek from the feed itself".
        # NewsLLMEnricher fills this in with a plain-language rephrase of
        # the title. If enrichment also fails, it falls back to static text
        # at that point -- so this function never needs to know about LLMs.
        return ""

    return text[:280]


async def _fetch_source(
    client: AsyncHttpClient,
    source: dict,
    dedupe: DedupeStore,
    cutoff: datetime,
    health: SourceHealthStore,
    logo_session: aiohttp.ClientSession,
    logo_resolver: LogoResolver,
) -> list[Article]:
    name = source["name"]
    feed_url = source["feed_url"]
    default_category = source.get("category", "Uncategorized")
    publisher_data = dict(source["publisher"])

    # Resolve publisher logo/favicon once per source. LogoResolver caches by
    # domain, so duplicate publishers across feeds do not repeatedly scrape
    # the same website.
    try:
        logo_url, favicon_url = await logo_resolver.resolve_publisher_assets(
            logo_session,
            publisher_data.get("website", ""),
            publisher_data.get("domain", ""),
        )
        publisher_data["logoUrl"] = logo_url
        publisher_data["faviconUrl"] = favicon_url
    except Exception as e:
        logger.warning(
            "Logo resolution failed for publisher %s: %s",
            publisher_data.get("name", name),
            e,
        )

    publisher = Publisher(**publisher_data)

    if health.should_skip(name):
        logger.info("Skipping %s: in backoff after repeated failures", name)
        health.decrement_skip_counters()
        return []

    logger.info("Fetching %s", name)
    text = await client.get_text(feed_url)
    if text is None:
        health.record_failure(name)
        return []

    parsed = await asyncio.to_thread(_parse_feed_text, text)
    if parsed.bozo and not parsed.entries:
        health.record_failure(name)
        return []

    health.record_success(name)

    rows: list[Article] = []
    skipped_stale = 0
    skipped_seen = 0

    for entry in parsed.entries:
        published_at = _parse_published(entry)
        if published_at is None or published_at < cutoff:
            skipped_stale += 1
            continue

        url = (entry.get("link") or "").strip()
        if not url or not dedupe.is_new(url):
            skipped_seen += 1
            continue

        title = (entry.get("title") or "").strip()
        raw_summary = (entry.get("summary") or entry.get("description") or "").strip()
        dek = _clean_dek(raw_summary, title)
        tags = [t.term for t in entry.get("tags", [])] if entry.get("tags") else []

        rows.append(
            Article(
                slug=slugify(title),
                title=title,
                dek=dek,
                aiSummary="",
                articleUrl=url,
                category=default_category,
                filterTags=tags,
                publishedAt=published_at.isoformat(),
                publisher=publisher,
                topics=[],
            )
        )
        dedupe.mark_seen(url)

    logger.info(
        "%s: %d entries in feed -> %d new, %d too old, %d already seen in a prior run",
        name, len(parsed.entries), len(rows), skipped_stale, skipped_seen,
    )
    return rows


async def run_news(
    client: AsyncHttpClient, target_count: int, dedupe: DedupeStore, hours: int = RECENCY_HOURS
) -> list[Article]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    health = SourceHealthStore()
    logo_resolver = LogoResolver()

    # A single shared session is used for all logo lookups. The resolver also
    # keeps a persistent domain -> logo/favicon cache on disk.
    timeout = aiohttp.ClientTimeout(total=8)
    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as logo_session:
        try:
            results = await asyncio.gather(
                *(
                    _fetch_source(
                        client,
                        source,
                        dedupe,
                        cutoff,
                        health,
                        logo_session,
                        logo_resolver,
                    )
                    for source in SOURCES
                )
            )
        finally:
            health.close()

    # `results` is a list of per-source article lists, aligned with SOURCES.
    # Cap each source's contribution *before* the global sort/truncate below.
    # Without this, a plain "pool everything, sort by publishedAt, take top
    # N" naturally gets crowded out by whichever handful of sources publish
    # most often inside the recency window (arXiv, Hacker News, etc.) -- a
    # blog that posts twice a week can never out-rank a feed that posts
    # every few minutes, even though you added it specifically for variety.
    all_rows: list[Article] = []
    for source_rows in results:
        source_rows.sort(key=lambda r: r.publishedAt, reverse=True)
        if MAX_ARTICLES_PER_SOURCE is not None:
            source_rows = source_rows[:MAX_ARTICLES_PER_SOURCE]
        all_rows.extend(source_rows)

    all_rows.sort(key=lambda r: r.publishedAt, reverse=True)
    return all_rows[:target_count]