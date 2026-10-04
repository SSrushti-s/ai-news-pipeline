import os
import json
import logging
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import aiohttp

logger = logging.getLogger("ai_news_crawler.logo_resolver")

LOGO_CACHE_FILE = "./output/publisher_logos.json"

class LogoResolver:
    def __init__(self, cache_file: str = LOGO_CACHE_FILE):
        self.cache_file = cache_file
        self._cache: dict[str, dict[str, str]] = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, str]]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to persist logo cache: {e}")

    async def resolve_publisher_assets(
        self, session: aiohttp.ClientSession, website: str, domain: str
    ) -> tuple[str, str]:
        """
        Returns (logoUrl, faviconUrl).
        Inspects local cache, HTML icon/og tags, /favicon.ico, Clearbit, and Google CDN.
        """
        clean_domain = domain.lower().replace("www.", "").strip()

        # 1. Check local cache first
        if clean_domain in self._cache:
            entry = self._cache[clean_domain]
            return entry.get("logoUrl", ""), entry.get("faviconUrl", "")

        logo_url = ""
        favicon_url = ""

        # Default fallbacks
        google_favicon = f"https://www.google.com/s2/favicons?domain={clean_domain}&sz=128"
        clearbit_logo = f"https://logo.clearbit.com/{clean_domain}"

        target_url = website if website.startswith("http") else f"https://{clean_domain}"

        # 2. Try Clearbit Brand Logo CDN
        try:
            async with session.head(clearbit_logo, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    logo_url = clearbit_logo
        except Exception:
            pass

        # 3. Scrape Publisher Homepage for HTML icon tags
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            async with session.get(target_url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")

                    # Look for Apple Touch Icon (usually 180x180 high-res PNG logo)
                    apple_icon = soup.find(
                        "link",
                        rel=lambda r: (
                            any("apple-touch-icon" in str(value).lower() for value in r)
                            if isinstance(r, (list, tuple))
                            else "apple-touch-icon" in str(r).lower()
                        ),
                    )
                    if apple_icon and apple_icon.get("href"):
                        resolved = urljoin(target_url, apple_icon["href"])
                        if not logo_url:
                            logo_url = resolved
                        favicon_url = resolved

                    # Look for standard rel="icon" or rel="shortcut icon"
                    icon_tag = soup.find(
                        "link",
                        rel=lambda r: (
                            any("icon" in str(value).lower() for value in r)
                            if isinstance(r, (list, tuple))
                            else "icon" in str(r).lower()
                        ),
                    )
                    if icon_tag and icon_tag.get("href"):
                        resolved_icon = urljoin(target_url, icon_tag["href"])
                        favicon_url = resolved_icon
                        if not logo_url:
                            logo_url = resolved_icon

                    # Look for og:image if logo still not found
                    if not logo_url:
                        og_image = soup.find("meta", property="og:image")
                        if og_image and og_image.get("content"):
                            logo_url = urljoin(target_url, og_image["content"])
        except Exception as e:
            logger.debug(f"HTML logo scraping failed for {clean_domain}: {e}")

        # 4. Fallback assignment
        if not favicon_url:
            favicon_url = google_favicon
        if not logo_url:
            logo_url = favicon_url or google_favicon

        # 5. Persist to cache
        self._cache[clean_domain] = {"logoUrl": logo_url, "faviconUrl": favicon_url}
        self._save_cache()

        return logo_url, favicon_url