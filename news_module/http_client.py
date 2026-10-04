import asyncio
import logging
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import aiohttp

from .config import (
    MAX_CONCURRENT_PER_HOST,
    MAX_CONCURRENT_REQUESTS,
    REQUEST_TIMEOUT,
    ROBOTS_CACHE_TTL_SECONDS,
    USER_AGENT,
)

logger = logging.getLogger("ai_news_crawler.http")


class AsyncHttpClient:
    def __init__(self):
        self._session: aiohttp.ClientSession | None = None
        self._global_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._host_semaphores: dict[str, asyncio.Semaphore] = {}
        self._robots_cache: dict[str, tuple[RobotFileParser | None, float]] = {}

    async def __aenter__(self) -> "AsyncHttpClient":
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_PER_HOST)
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": USER_AGENT}, timeout=timeout, connector=connector
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    def _host_semaphore(self, host: str) -> asyncio.Semaphore:
        if host not in self._host_semaphores:
            self._host_semaphores[host] = asyncio.Semaphore(MAX_CONCURRENT_PER_HOST)
        return self._host_semaphores[host]

    async def _get_robots(self, host: str, scheme: str) -> RobotFileParser | None:
        """Fetch and cache robots.txt for a host. Returns None if it can't
        be fetched/parsed, in which case callers should treat the URL as
        allowed (absence of robots.txt doesn't mean disallowed)."""
        cached = self._robots_cache.get(host)
        if cached and (time.monotonic() - cached[1]) < ROBOTS_CACHE_TTL_SECONDS:
            return cached[0]

        robots_url = f"{scheme}://{host}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            assert self._session is not None
            async with self._session.get(robots_url) as resp:
                if resp.status >= 400:
                    self._robots_cache[host] = (None, time.monotonic())
                    return None
                body = await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            self._robots_cache[host] = (None, time.monotonic())
            return None

        try:
            parser.parse(body.splitlines())
        except Exception:
            self._robots_cache[host] = (None, time.monotonic())
            return None

        self._robots_cache[host] = (parser, time.monotonic())
        return parser

    async def is_allowed(self, url: str) -> bool:
        """Check robots.txt before fetching a feed. Fails open (returns
        True) if robots.txt is missing or unparseable, per standard
        robots.txt semantics."""
        parsed = urlparse(url)
        if not parsed.netloc:
            return True
        robots = await self._get_robots(parsed.netloc, parsed.scheme or "https")
        if robots is None:
            return True
        return robots.can_fetch(USER_AGENT, url)

    async def get_text(self, url: str) -> str | None:
        assert self._session is not None, "AsyncHttpClient must be used as `async with`"

        if not await self.is_allowed(url):
            logger.info("Skipping %s: disallowed by robots.txt", url)
            return None

        host = urlparse(url).netloc
        async with self._global_semaphore, self._host_semaphore(host):
            try:
                async with self._session.get(url) as resp:
                    resp.raise_for_status()
                    return await resp.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("Request failed for %s: %s", url, exc)
                return None