"""
Central configuration.

All tunables live here so scaling from a trial run
(hundreds of records) to production (500k+ records)
is a config change, not a code change.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _parse_env_list(
    var_name: str,
    legacy_var_name: str | None = None,
) -> list[str]:
    """
    Parse comma-separated environment variables into a clean list.

    Example:
        GROQ_API_KEYS=key1,key2,key3

    becomes:
        ["key1", "key2", "key3"]
    """

    values: list[str] = []

    env_names = (
        (var_name, legacy_var_name)
        if legacy_var_name
        else (var_name,)
    )

    for env_name in env_names:
        if not env_name:
            continue

        raw_value = os.getenv(env_name, "")

        if not raw_value:
            continue

        values.extend(
            part.strip()
            for part in raw_value.split(",")
            if part.strip()
        )

    return values


@dataclass
class ScraperConfig:
    max_concurrency: int = int(
        os.getenv("SCRAPER_MAX_CONCURRENCY", 20)
    )

    request_timeout_s: int = int(
        os.getenv("SCRAPER_TIMEOUT", 30)
    )

    max_retries: int = int(
        os.getenv("SCRAPER_MAX_RETRIES", 4)
    )

    base_backoff_s: float = 1.5
    max_backoff_s: float = 60.0

    user_agents: list[str] = field(
        default_factory=lambda: [
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        ]
    )


@dataclass
class LLMConfig:

    # ============================================================
    # FALLBACK CHAIN
    # ============================================================
    #
    # First:
    #   Groq using all configured keys × models
    #
    # Then:
    #   Gemini
    #
    # Finally:
    #   Groq again using the SAME keys × models
    #
    fallback_chain: list[str] = field(
        default_factory=lambda: [
            "groq_primary",
            "gemini",
            "groq_secondary",
        ]
    )

    # ============================================================
    # TOKEN / OUTPUT SETTINGS
    # ============================================================

    max_input_tokens: int = 6000
    chunk_overlap_tokens: int = 200
    max_output_tokens: int = 2000

    # ============================================================
    # RETRY / BACKOFF
    # ============================================================

    max_retries_per_tier: int = 3
    base_backoff_s: float = 2.0
    max_backoff_s: float = 90.0

    # ============================================================
    # API KEYS
    # ============================================================

    gemini_api_key: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_API_KEY",
            "",
        )
    )

    # Legacy/single Groq key support
    groq_api_key: str = field(
        default_factory=lambda: os.getenv(
            "GROQ_API_KEY",
            "",
        )
    )

    # ============================================================
    # GROQ API KEYS
    # ============================================================
    #
    # Example:
    #
    # GROQ_API_KEYS=key1,key2,key3,key4,key5
    #
    # Both Groq tiers use this SAME list.
    #

    groq_api_keys: list[str] = field(
        default_factory=lambda: _parse_env_list(
            "GROQ_API_KEYS",
            "GROQ_API_KEY",
        )
    )

    # Kept for compatibility with your existing configuration.
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv(
            "DEEPSEEK_API_KEY",
            "",
        )
    )

    # ============================================================
    # DEFAULT MODELS
    # ============================================================

    gemini_model: str = "gemini-2.5-flash"

    groq_model: str = "openai/gpt-oss-120b"

    deepseek_model: str = "deepseek-chat"

    # ============================================================
    # GROQ MODELS
    # ============================================================
    #
    # These 4 models are used with EVERY Groq API key.
    #
    # 5 keys × 4 models = 20 possible Groq attempts
    #
    # The same 20 combinations are available again
    # when the secondary Groq tier is reached.
    #

    groq_models: list[str] = field(
        default_factory=lambda: [
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "llama-3.1-8b-instant",
        ]
    )

    # ============================================================
    # GEMINI MODELS
    # ============================================================

    gemini_models: list[str] = field(
        default_factory=lambda: [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
        ]
    )


@dataclass
class FreshnessConfig:
    max_age_hours: int = 24

    dedupe_ttl_days: int = 30


@dataclass
class StorageConfig:
    output_dir: str = "output"

    dedupe_db_path: str = (
        "output/seen_urls.sqlite3"
    )

    entity_seed_path: str = (
        "config/seed_entities.json"
    )


# ================================================================
# GLOBAL CONFIG OBJECTS
# ================================================================

SCRAPER = ScraperConfig()
LLM = LLMConfig()
FRESHNESS = FreshnessConfig()
STORAGE = StorageConfig()
