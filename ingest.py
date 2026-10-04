# ingest.py
import json
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables before anything else
load_dotenv()

from news_module.config import STORAGE  
from news_module.news_model import Article   
from news_module.db import ArticleDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("graphone.ingest_news")

# A lightweight class mimic to satisfy `db.py`'s requirement for a `.to_dict()` method
class DictToClassWrapper:
    def __init__(self, data_dict):
        self.__dict__.update(data_dict)
        self._raw_data = data_dict

    def to_dict(self):
        return self._raw_data

async def push_local_news_to_neon():
    primary_file = Path(STORAGE.output_dir) / "output_news.json" 
    fallback_file = Path(STORAGE.output_dir) / "latest_files.json"

    # Determine which file to process
    target_file = None
    if primary_file.exists():
        target_file = primary_file
    elif fallback_file.exists():
        logger.info(f"Primary file not found. Falling back to {fallback_file}")
        target_file = fallback_file
    else:
        logger.error("❌ Neither primary nor fallback files were found.")
        return

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            file_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read or parse JSON from {target_file}: {e}")
        return

    # Extract the array from the "news" key if present (matching main.py's structure)
    if isinstance(file_data, dict) and "news" in file_data:
        raw_articles = file_data["news"]
    elif isinstance(file_data, list):
        raw_articles = file_data
    else:
        logger.error(f"Unexpected JSON format in {target_file}. Expected a list or an object containing a 'news' key.")
        return

    # Validate and instantiate full Article records
    records = []
    for item in raw_articles:
        if hasattr(Article, "model_validate"):
            art = Article.model_validate(item)
        elif hasattr(Article, "parse_obj"):
            art = Article.parse_obj(item)
        else:
            art = Article(**item)

        # Explicitly patch `publisher` if it's a raw dictionary
        if isinstance(art.publisher, dict):
            art.publisher = DictToClassWrapper(art.publisher)

        # Explicitly patch `topics` if they are raw dictionaries
        if art.topics and isinstance(art.topics, list):
            patched_topics = []
            for t in art.topics:
                if isinstance(t, dict):
                    patched_topics.append(DictToClassWrapper(t))
                else:
                    patched_topics.append(t)
            art.topics = patched_topics

        records.append(art)

    logger.info(f"Loaded {len(records)} validated news records from {target_file}.")

    if not records:
        logger.info("No records found to push.")
        return

    # Connect and ingest into Neon DB
    db = ArticleDatabase()
    await db.connect()
    
    logger.info("Ingesting articles into Neon DB...")
    upserted = await db.upsert_articles(records) 
    
    await db.close()
    logger.info(f"🎉 Successfully pushed {upserted} news records to Neon DB!")

if __name__ == "__main__":
    asyncio.run(push_local_news_to_neon())
