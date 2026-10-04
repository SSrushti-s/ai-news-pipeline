import os
import json
import logging
import asyncpg
from typing import List
from .news_model import Article

logger = logging.getLogger("ai_news_crawler.db")

class ArticleDatabase:
    def __init__(self):
        self.pool: asyncpg.Pool | None = None
        self.dsn = os.getenv("DATABASE_URL", "")

    async def connect(self):
        if not self.dsn:
            logger.warning("DATABASE_URL not set. Neon ingestion skipped.")
            return

        clean_dsn = self.dsn.split("?")[0]
        self.pool = await asyncpg.create_pool(
            dsn=clean_dsn,
            ssl="require",
            min_size=1,
            max_size=5
        )
        await self._init_table()
        logger.info("Connected to Neon DB successfully.")

    async def _init_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS "Article" (
            "id" TEXT PRIMARY KEY,
            "slug" TEXT NOT NULL,
            "title" TEXT NOT NULL,
            "dek" TEXT,
            "aiSummary" TEXT,
            "articleUrl" TEXT UNIQUE NOT NULL,
            "category" TEXT NOT NULL,
            "publishedAt" TEXT NOT NULL,
            "publisher" JSONB NOT NULL,
            "filterTags" JSONB NOT NULL,
            "topics" JSONB NOT NULL,
            "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query)

    async def upsert_articles(self, articles: List[Article]) -> int:
        if not self.pool or not articles:
            return 0

        query = """
        INSERT INTO "Article" (
            "id", "slug", "title", "dek", "aiSummary", "articleUrl",
            "category", "publishedAt", "publisher", "filterTags", "topics", "updatedAt"
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, NOW()
        )
        ON CONFLICT ("articleUrl") DO UPDATE SET
            "dek" = EXCLUDED."dek",
            "aiSummary" = EXCLUDED."aiSummary",
            "filterTags" = EXCLUDED."filterTags",
            "topics" = EXCLUDED."topics",
            "updatedAt" = NOW();
        """

        upserted = 0
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for a in articles:
                    import uuid
                    article_id = str(uuid.uuid4())
                    pub_json = json.dumps(a.publisher.to_dict())
                    tags_json = json.dumps(a.filterTags)
                    topics_json = json.dumps([t.to_dict() for t in a.topics])

                    await conn.execute(
                        query,
                        article_id, a.slug, a.title, a.dek, a.aiSummary, a.articleUrl,
                        a.category, a.publishedAt, pub_json, tags_json, topics_json
                    )
                    upserted += 1

        logger.info(f"Upserted {upserted} articles to Neon DB.")
        return upserted

    async def close(self):
        if self.pool:
            await self.pool.close()