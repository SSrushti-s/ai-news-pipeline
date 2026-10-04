import asyncio
import json
import logging
from typing import List

from .orchestrator import LLMOrchestrator
from .news_model import Article, Topic

logger = logging.getLogger("ai_news_crawler.llm_enricher")

SYSTEM_PROMPT = """You are a precise information extraction system. Given a news
article's title and text snippet, extract ONLY the following as JSON, with no preamble,
no markdown fences, and no commentary:

{
  "dek": "<A single plain-language sentence that just rephrases the TITLE in simpler words. It is a reworded headline, not a summary of the article body, and must not overlap in wording with aiSummary.>",
  "aiSummary": "<A concise 2-3 sentence factual summary of the article, in your own words>",
  "filterTags": ["<specific keyword tags, tech stacks, or companies mentioned, e.g., 'OpenAI', 'Python'>"],
  "topics": ["<2-4 core categorized high-level macro domains, e.g., 'AI', 'Security'>"]
}

Rules:
- dek only rewords the title in plainer language -- do not pull in facts from the description that aren't in the title.
- aiSummary must be grounded only in the provided text -- do not add outside knowledge.
- filterTags should be key technical terms, frameworks, or company names mentioned in the text.
- topics must be clean, high-level industry tracks (e.g., 'Hardware', 'Open Source').
- Output ONLY the JSON object, nothing else. Do not wrap in ```json markers.
"""


class NewsLLMEnricher:
    def __init__(self, orchestrator: LLMOrchestrator | None = None):
        self.orchestrator = orchestrator or LLMOrchestrator()

    async def enrich_article(self, article: Article) -> Article:
        text_context = f"Title: {article.title}\nDescription: {article.dek}"
        if not text_context.strip():
            return article

        result = await self.orchestrator.extract(SYSTEM_PROMPT, text_context)

        # If the result is a string (due to orchestrator bugs), try to normalize it into a dict
        if isinstance(result, str):
            try:
                cleaned = result.strip().removeprefix("```json").removesuffix("```").strip()
                result = json.loads(cleaned)
            except Exception:
                result = None

        if result is None or not isinstance(result, dict):
            logger.warning("LLM validation failed for %s, using fallbacks.", article.articleUrl)
            if not article.dek:
                article.dek = f"Latest updates on: {article.title}"
            return article

        # dek: only fill in if the feed didn't give us a usable one (see
        # pipeline._clean_dek, which leaves "" as a sentinel for that case).
        # A real feed-provided dek is left untouched.
        if not article.dek:
            raw_dek = result.get("dek")
            article.dek = str(raw_dek).strip() if raw_dek else f"Latest updates on: {article.title}"

        # Robust fallbacks for variable extraction mapping
        ai_summary = result.get("aiSummary") or result.get("summary") or ""
        article.aiSummary = str(ai_summary).strip()

        # Handle tag mapping arrays cleanly
        raw_tags = result.get("filterTags") or result.get("entities") or []
        article.filterTags = [str(t).strip() for t in raw_tags if t] if isinstance(raw_tags, list) else []

        # Topics are just {"name": ...} now -- no synthetic ids
        raw_topics = result.get("topics") or []
        parsed_topics: list[Topic] = []
        if isinstance(raw_topics, list):
            for t in raw_topics:
                if isinstance(t, dict) and t.get("name"):
                    parsed_topics.append(Topic(name=str(t["name"]).strip()))
                elif isinstance(t, str) and t.strip():
                    parsed_topics.append(Topic(name=t.strip()))
        article.topics = parsed_topics

        # The LLM's primary topic is a much better `category` than the
        # static per-source default set in sources.py, so promote it here.
        # If enrichment produced no usable topics, keep the source default.
        if parsed_topics:
            article.category = parsed_topics[0].name

        return article

    async def enrich_batch(self, articles: List[Article]) -> List[Article]:
        enriched_articles: List[Article] = []
        for index, article in enumerate(articles):
            if index > 0:
                # 1.5 second pacing gap protects free Groq/Gemini tiers from throttling
                await asyncio.sleep(1.5)

            logger.info("Processing article %d/%d: %s...", index + 1, len(articles), article.title[:40])
            enriched = await self.enrich_article(article)
            enriched_articles.append(enriched)

        return enriched_articles