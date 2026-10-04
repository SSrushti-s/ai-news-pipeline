"""
Central configuration. All tunables live here so scaling from a trial run
(hundreds of records) to production (500k+ records) is a config change,
not a code change — this is what the assignment means by
'scale without code changes, only infrastructure scaling'.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def _parse_env_list(var_name: str, legacy_var_name: str | None = None) -> list[str]:
    values: list[str] = []
    for env_name in (var_name, legacy_var_name) if legacy_var_name else (var_name,):
        raw_value = os.getenv(env_name, "")
        if not raw_value:
            continue
        values.extend(part.strip() for part in raw_value.split(",") if part.strip())
    return values


@dataclass
class ScraperConfig:
    max_concurrency: int = int(os.getenv("SCRAPER_MAX_CONCURRENCY", 20))
    request_timeout_s: int = int(os.getenv("SCRAPER_TIMEOUT", 30))
    max_retries: int = int(os.getenv("SCRAPER_MAX_RETRIES", 4))
    base_backoff_s: float = 1.5
    max_backoff_s: float = 60.0
    user_agents: list[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    ])

@dataclass
class LLMConfig:
    fallback_chain: list[str] = field(
        default_factory=lambda: [
            "groq",
            "gemini",
            "groq",
        ]
    )
    max_input_tokens: int = 6000
    chunk_overlap_tokens: int = 200
    max_output_tokens: int = 2000

    max_retries_per_tier: int = 3
    base_backoff_s: float = 2.0
    max_backoff_s: float = 90.0

    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    groq_models: list[str] = field(default_factory=lambda: [
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    ])
    
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))

    # Default models
    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "openai/gpt-oss-120b"
    deepseek_model: str = "deepseek-chat"

    # Ordered fastest -> strongest
    groq_models: list[str] = field(default_factory=lambda: [
        "qwen/qwen3.8-27b",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
    ])

    gemini_models: list[str] = field(default_factory=lambda: [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ])


@dataclass
class FreshnessConfig:
    max_age_hours: int = 24
    dedupe_ttl_days: int = 30  # how long a seen-URL hash is kept to prevent reprocessing


@dataclass
class StorageConfig:
    output_dir: str = "output"
    dedupe_db_path: str = "output/seen_urls.sqlite3"
    entity_seed_path: str = "config/seed_entities.json"


SCRAPER = ScraperConfig()
LLM = LLMConfig()
FRESHNESS = FreshnessConfig()
STORAGE = StorageConfig()
