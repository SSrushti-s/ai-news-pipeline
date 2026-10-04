from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class Publisher:
    """
    Required: name, domain, website
    Optional (defaults applied by the platform): logoUrl, faviconUrl,
    colorHex, followersLabel, credibilityScore
    """
    name: str
    domain: str
    website: str
    logoUrl: Optional[str] = None
    faviconUrl: Optional[str] = None
    colorHex: Optional[str] = None
    followersLabel: Optional[str] = None
    credibilityScore: float = 0.8

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class Topic:
    name: str

    def to_dict(self) -> dict:
        return {"name": self.name}

@dataclass
class Article:
    slug: str
    title: str
    dek: str
    articleUrl: str
    category: str
    publishedAt: str
    publisher: Publisher
    filterTags: list[str] = field(default_factory=list)
    topics: list[Topic] = field(default_factory=list)
    aiSummary: str = ""

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "title": self.title,
            "dek": self.dek,
            "aiSummary": self.aiSummary,
            "articleUrl": self.articleUrl,
            "category": self.category,
            "filterTags": list(self.filterTags),
            "publishedAt": self.publishedAt,
            "publisher": self.publisher.to_dict(),
            "topics": [t.to_dict() for t in self.topics],
        }