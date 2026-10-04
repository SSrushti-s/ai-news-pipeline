from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class StorageConfig:
    output_dir: str = "./output"
    def __post_init__(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

STORAGE = StorageConfig()
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
REQUEST_TIMEOUT = 15          
MAX_CONCURRENT_REQUESTS = 40   # total in-flight requests across all sources
MAX_CONCURRENT_PER_HOST = 2    # cap per domain so no single publisher gets hammered
RECENCY_HOURS = 138           

# robots.txt is fetched once per host and cached in memory for this long.
ROBOTS_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours

# If a source fails (network error, disallowed by robots.txt, repeated
# empty/bozo feed) this many times in a row, it's skipped for
# SOURCE_BACKOFF_RUNS subsequent runs instead of retried every time.
# This is just politeness/efficiency at 500-source scale -- it's not a
# retry-harder-to-get-past-defenses mechanism.
SOURCE_FAILURE_THRESHOLD = 3
SOURCE_BACKOFF_RUNS = 5
SOURCE_HEALTH_DB = "./output/.source_health.sqlite3"

# Cap on how many articles any single source can contribute to one run's
# output, applied *before* the global recency sort/truncate in
# pipeline.run_news(). High-frequency feeds (arXiv, Hacker News, etc.) post
# far more often inside the RECENCY_HOURS window than low-frequency ones
# (company blogs), so without this cap a global "sort by publishedAt, take
# top N" naturally crowds out everything except the 1-2 most prolific
# sources. Set to None to disable the cap and go back to pure recency sort.
MAX_ARTICLES_PER_SOURCE = 3