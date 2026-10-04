"""
Source list, loaded from sources.yaml.

Kept as a .py module (rather than having callers read the YAML directly)
so the rest of the codebase doesn't care where the list comes from, and so
we can validate each entry once, at load time, instead of failing deep
inside the pipeline on a malformed row.

At 500+ sources this file is a thin loader, not a place to hand-edit
entries -- edit sources.yaml instead.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("ai_news_crawler.sources")

SOURCES_FILE = Path(__file__).parent.parent / "sources.yaml"

_REQUIRED_SOURCE_KEYS = ("name", "feed_url")
_REQUIRED_PUBLISHER_KEYS = ("name", "domain", "website")


def _validate(entry: dict[str, Any], index: int) -> bool:
    for key in _REQUIRED_SOURCE_KEYS:
        if not entry.get(key):
            logger.warning("sources.yaml entry #%d missing required '%s', skipping", index, key)
            return False
    publisher = entry.get("publisher")
    if not isinstance(publisher, dict):
        logger.warning("sources.yaml entry #%d ('%s') missing publisher block, skipping", index, entry.get("name"))
        return False
    for key in _REQUIRED_PUBLISHER_KEYS:
        if not publisher.get(key):
            logger.warning(
                "sources.yaml entry #%d ('%s') missing publisher.%s, skipping",
                index, entry.get("name"), key,
            )
            return False
    return True


def load_sources(path: Path | str = SOURCES_FILE) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"sources file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []

    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a top-level YAML list of source entries")

    sources = [entry for i, entry in enumerate(raw) if _validate(entry, i)]
    logger.info("Loaded %d/%d sources from %s", len(sources), len(raw), path)
    return sources


# Loaded once at import time so existing callers (`from .sources import SOURCES`)
# keep working unchanged. Call load_sources() directly if you need to reload
# after editing the YAML without restarting the process.
SOURCES: list[dict] = load_sources()