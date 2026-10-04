import os
import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from news_module.config import STORAGE, RECENCY_HOURS
from news_module.dedupe import DedupeStore
from news_module.http_client import AsyncHttpClient
from news_module.news_enricher import NewsLLMEnricher
from news_module.pipeline import run_news
from news_module.db import ArticleDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ai_news_crawler.main")

DEFAULT_TARGET_COUNT = 40
INGEST_TO_DB = os.getenv("INGEST_TO_DB", "false").lower() == "true"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch, dedupe, and LLM-enrich AI news.")
    parser.add_argument("--news", type=int, default=DEFAULT_TARGET_COUNT, dest="target_count")
    parser.add_argument("--hours", type=int, default=RECENCY_HOURS, dest="hours")
    parser.add_argument("--reset-dedupe", action="store_true")
    return parser.parse_args()

async def main() -> None:
    args = parse_args()

    db = None
    if INGEST_TO_DB:
        db = ArticleDatabase()
        await db.connect()

    dedupe = DedupeStore()
    try:
        if args.reset_dedupe:
            logger.info("Clearing dedupe log (--reset-dedupe was passed)")
            dedupe.reset()

        async with AsyncHttpClient() as client:
            articles = await run_news(client, args.target_count, dedupe, hours=args.hours)

        if not articles:
            logger.info("No new articles matched.")
            return

        logger.info("Fetched %d new articles, enriching with LLM...", len(articles))
        enricher = NewsLLMEnricher()
        enriched = await enricher.enrich_batch(articles)

        # 1. Save local JSON copy (always available to inspect)
        output = {"news": [a.to_dict() for a in enriched]}
        out_path = Path(STORAGE.output_dir) / f"news_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("✅ Saved local output to %s", out_path)

        # 2. Ingest to Neon DB if enabled
        if INGEST_TO_DB and db:
            logger.info("Ingesting articles into Neon DB...")
            upserted = await db.upsert_articles(enriched)
            logger.info("🚀 Successfully ingested %d articles into Neon DB.", upserted)

    finally:
        dedupe.close()
        if db:
            await db.close()

if __name__ == "__main__":
    asyncio.run(main())