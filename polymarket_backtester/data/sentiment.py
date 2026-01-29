"""
Sentiment and news data connectors.

Provides access to:
- News article sentiment analysis
- Social media sentiment (Twitter/X)
- News volume and momentum metrics

These signals complement polling data for the Information Edge strategy.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any
import re

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from .base import SignalConnector, ConnectorConfig, CacheConfig


@dataclass
class SentimentReading:
    """A single sentiment measurement."""
    timestamp: datetime
    score: float  # -1 (very negative) to 1 (very positive)
    magnitude: float  # 0-1, strength of sentiment
    volume: int  # Number of items analyzed
    source: str
    keywords: list[str] = field(default_factory=list)
    sample_texts: list[str] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        """Score weighted by magnitude and volume."""
        volume_weight = min(self.volume / 100, 1.0)
        return self.score * self.magnitude * volume_weight


@dataclass
class NewsItem:
    """A news article or social media post."""
    id: str
    title: str
    source: str
    published_at: datetime
    url: Optional[str] = None
    description: Optional[str] = None
    sentiment_score: Optional[float] = None
    relevance_score: float = 1.0
    keywords: list[str] = field(default_factory=list)


class SentimentSource(ABC):
    """Abstract base for sentiment data sources."""

    @abstractmethod
    async def fetch_sentiment(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> list[SentimentReading]:
        """Fetch sentiment readings for a query."""
        pass

    @abstractmethod
    async def fetch_news(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> list[NewsItem]:
        """Fetch news items matching a query."""
        pass


class NewsAPISource(SentimentSource):
    """
    Fetches news and sentiment from NewsAPI.org.

    Requires an API key from https://newsapi.org/
    Free tier: 100 requests/day, 1 month historical
    """

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._session: Optional[Any] = None

    async def _ensure_session(self):
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_news(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> list[NewsItem]:
        """
        Fetch news articles from NewsAPI.

        Args:
            query: Search query
            start_time: Start of date range
            end_time: End of date range
            limit: Max articles to return
        """
        if not self.api_key:
            return []

        await self._ensure_session()

        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": min(limit, 100)
        }

        if start_time:
            params["from"] = start_time.strftime("%Y-%m-%d")
        if end_time:
            params["to"] = end_time.strftime("%Y-%m-%d")

        try:
            async with self._session.get(
                f"{self.BASE_URL}/everything",
                params=params
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception:
            return []

        articles = []
        for item in data.get("articles", []):
            try:
                published = datetime.fromisoformat(
                    item["publishedAt"].replace("Z", "+00:00")
                )
            except (ValueError, KeyError):
                published = datetime.now()

            # Simple sentiment analysis based on title
            sentiment = self._analyze_text_sentiment(
                item.get("title", "") + " " + item.get("description", "")
            )

            articles.append(NewsItem(
                id=item.get("url", ""),
                title=item.get("title", ""),
                source=item.get("source", {}).get("name", "Unknown"),
                published_at=published,
                url=item.get("url"),
                description=item.get("description"),
                sentiment_score=sentiment,
                keywords=self._extract_keywords(item.get("title", ""))
            ))

        return articles[:limit]

    async def fetch_sentiment(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> list[SentimentReading]:
        """
        Calculate sentiment from news articles.

        Aggregates article sentiment into time-bucketed readings.
        """
        articles = await self.fetch_news(query, start_time, end_time, limit=100)

        if not articles:
            return []

        # Group by day
        daily_articles: dict[str, list[NewsItem]] = {}
        for article in articles:
            day = article.published_at.strftime("%Y-%m-%d")
            if day not in daily_articles:
                daily_articles[day] = []
            daily_articles[day].append(article)

        # Calculate daily sentiment
        readings = []
        for day, day_articles in daily_articles.items():
            scores = [a.sentiment_score for a in day_articles if a.sentiment_score is not None]
            if not scores:
                continue

            avg_score = sum(scores) / len(scores)
            magnitude = sum(abs(s) for s in scores) / len(scores)

            readings.append(SentimentReading(
                timestamp=datetime.strptime(day, "%Y-%m-%d"),
                score=avg_score,
                magnitude=magnitude,
                volume=len(day_articles),
                source="newsapi",
                keywords=[kw for a in day_articles for kw in a.keywords[:3]],
                sample_texts=[a.title for a in day_articles[:3]]
            ))

        return sorted(readings, key=lambda r: r.timestamp)

    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Simple rule-based sentiment analysis.

        Returns score from -1 (negative) to 1 (positive).
        """
        if not text:
            return 0.0

        text = text.lower()

        # Positive words
        positive_words = [
            "win", "winning", "lead", "leading", "surge", "gain", "gains",
            "ahead", "rise", "rising", "up", "boost", "success", "victory",
            "strong", "strengthen", "increase", "positive", "confident",
            "optimistic", "rally", "support", "momentum"
        ]

        # Negative words
        negative_words = [
            "lose", "losing", "trail", "trailing", "drop", "fall", "falling",
            "behind", "decline", "down", "weak", "weaken", "decrease",
            "negative", "concern", "worried", "crisis", "fail", "failure",
            "scandal", "controversy", "attack", "slump"
        ]

        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        if not text:
            return []

        # Simple keyword extraction
        words = re.findall(r'\b[A-Z][a-z]+\b', text)  # Capitalized words
        return list(set(words))[:5]


class TwitterSource(SentimentSource):
    """
    Fetches sentiment from Twitter/X.

    Note: Twitter API access has become more restricted.
    This is a reference implementation that would need valid API credentials.
    """

    BASE_URL = "https://api.twitter.com/2"

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.api_secret = api_secret
        self._session: Optional[Any] = None

    async def _ensure_session(self):
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required")
        if self._session is None or self._session.closed:
            headers = {}
            if self.bearer_token:
                headers["Authorization"] = f"Bearer {self.bearer_token}"
            self._session = aiohttp.ClientSession(headers=headers)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_news(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> list[NewsItem]:
        """
        Fetch tweets matching a query.

        Requires Twitter API v2 access.
        """
        if not self.bearer_token:
            return []

        await self._ensure_session()

        params = {
            "query": f"{query} -is:retweet lang:en",
            "max_results": min(limit, 100),
            "tweet.fields": "created_at,public_metrics,author_id"
        }

        if start_time:
            params["start_time"] = start_time.isoformat() + "Z"
        if end_time:
            params["end_time"] = end_time.isoformat() + "Z"

        try:
            async with self._session.get(
                f"{self.BASE_URL}/tweets/search/recent",
                params=params
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception:
            return []

        tweets = []
        for tweet in data.get("data", []):
            try:
                published = datetime.fromisoformat(
                    tweet["created_at"].replace("Z", "+00:00")
                )
            except (ValueError, KeyError):
                published = datetime.now()

            # Analyze sentiment
            sentiment = self._analyze_tweet_sentiment(tweet.get("text", ""))

            # Relevance based on engagement
            metrics = tweet.get("public_metrics", {})
            engagement = (
                metrics.get("like_count", 0) +
                metrics.get("retweet_count", 0) * 2 +
                metrics.get("reply_count", 0)
            )
            relevance = min(engagement / 100, 1.0)

            tweets.append(NewsItem(
                id=tweet.get("id", ""),
                title=tweet.get("text", "")[:140],
                source="twitter",
                published_at=published,
                url=f"https://twitter.com/i/web/status/{tweet.get('id')}",
                description=tweet.get("text"),
                sentiment_score=sentiment,
                relevance_score=relevance
            ))

        return tweets

    async def fetch_sentiment(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> list[SentimentReading]:
        """Calculate aggregated sentiment from tweets."""
        tweets = await self.fetch_news(query, start_time, end_time, limit=100)

        if not tweets:
            return []

        # Group by hour for Twitter (more granular than news)
        hourly_tweets: dict[str, list[NewsItem]] = {}
        for tweet in tweets:
            hour = tweet.published_at.strftime("%Y-%m-%d %H:00")
            if hour not in hourly_tweets:
                hourly_tweets[hour] = []
            hourly_tweets[hour].append(tweet)

        readings = []
        for hour, hour_tweets in hourly_tweets.items():
            scores = [t.sentiment_score for t in hour_tweets if t.sentiment_score is not None]
            if not scores:
                continue

            # Weight by relevance
            weighted_sum = sum(
                t.sentiment_score * t.relevance_score
                for t in hour_tweets
                if t.sentiment_score is not None
            )
            weight_total = sum(t.relevance_score for t in hour_tweets)

            avg_score = weighted_sum / weight_total if weight_total > 0 else 0
            magnitude = sum(abs(s) for s in scores) / len(scores)

            readings.append(SentimentReading(
                timestamp=datetime.strptime(hour, "%Y-%m-%d %H:%M"),
                score=avg_score,
                magnitude=magnitude,
                volume=len(hour_tweets),
                source="twitter",
                sample_texts=[t.title for t in hour_tweets[:3]]
            ))

        return sorted(readings, key=lambda r: r.timestamp)

    def _analyze_tweet_sentiment(self, text: str) -> float:
        """Analyze sentiment of a tweet."""
        if not text:
            return 0.0

        text = text.lower()

        # Twitter-specific positive indicators
        positive = [
            "🚀", "📈", "💪", "🎉", "✅", "👍", "🔥", "💯",
            "win", "winning", "bullish", "moon", "lfg", "lets go",
            "amazing", "great", "love", "best", "excited"
        ]

        # Twitter-specific negative indicators
        negative = [
            "📉", "💀", "😢", "😡", "❌", "👎", "🤮",
            "lose", "losing", "bearish", "dump", "crash",
            "terrible", "worst", "hate", "disappointed", "worried"
        ]

        pos_count = sum(1 for word in positive if word in text)
        neg_count = sum(1 for word in negative if word in text)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total


class SentimentConnector(SignalConnector):
    """
    Unified connector for sentiment data from multiple sources.

    Aggregates sentiment from news APIs, social media, and other sources.
    """

    def __init__(
        self,
        config: Optional[ConnectorConfig] = None,
        sources: Optional[list[SentimentSource]] = None,
        news_api_key: Optional[str] = None,
        twitter_bearer_token: Optional[str] = None
    ):
        if config is None:
            config = ConnectorConfig(
                name="sentiment",
                cache=CacheConfig(ttl_seconds=900)  # 15 min cache
            )
        super().__init__(config)

        if sources:
            self.sources = sources
        else:
            self.sources = []
            if news_api_key:
                self.sources.append(NewsAPISource(api_key=news_api_key))
            if twitter_bearer_token:
                self.sources.append(TwitterSource(bearer_token=twitter_bearer_token))

    async def connect(self) -> bool:
        self._is_connected = True
        return True

    async def disconnect(self) -> None:
        for source in self.sources:
            if hasattr(source, 'close'):
                await source.close()
        self._is_connected = False

    async def health_check(self) -> bool:
        return self._is_connected and len(self.sources) > 0

    async def fetch_markets(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """Sentiment connector doesn't have markets - returns empty."""
        return []

    async def fetch_market_prices(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        resolution: str = "1h"
    ) -> list[dict]:
        """Not applicable for sentiment."""
        return []

    async def fetch_current_price(self, market_id: str) -> dict:
        """Not applicable for sentiment."""
        return {}

    async def fetch_signals(
        self,
        market_id: str,
        signal_types: Optional[list[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> list[dict]:
        """
        Fetch sentiment signals for a topic.

        Args:
            market_id: Topic/query to analyze (e.g., "Biden election")
            signal_types: Sources to include ("news", "twitter")
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of sentiment signal dictionaries
        """
        # Use market_id as the search query
        query = market_id.replace("_", " ")

        signals = []

        for source in self.sources:
            source_name = source.__class__.__name__.lower()

            if signal_types and source_name not in [s.lower() for s in signal_types]:
                continue

            try:
                readings = await source.fetch_sentiment(query, start_time, end_time)
                for reading in readings:
                    signals.append({
                        "type": "sentiment",
                        "source": reading.source,
                        "timestamp": reading.timestamp,
                        "score": reading.score,
                        "magnitude": reading.magnitude,
                        "volume": reading.volume,
                        "weighted_score": reading.weighted_score,
                        "keywords": reading.keywords,
                        "samples": reading.sample_texts
                    })
            except Exception:
                continue

        return sorted(signals, key=lambda s: s["timestamp"])

    async def fetch_current_signals(
        self,
        market_id: str,
        signal_types: Optional[list[str]] = None
    ) -> dict:
        """
        Fetch current sentiment summary.

        Returns aggregated sentiment suitable for trading strategies.
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        signals = await self.fetch_signals(
            market_id,
            signal_types,
            start_time,
            end_time
        )

        if not signals:
            return {}

        # Aggregate recent sentiment
        recent_scores = [s["weighted_score"] for s in signals[-10:]]
        recent_volumes = [s["volume"] for s in signals[-10:]]

        avg_sentiment = sum(recent_scores) / len(recent_scores) if recent_scores else 0
        total_volume = sum(recent_volumes)

        # Calculate momentum (trend in sentiment)
        if len(signals) >= 4:
            early = signals[:len(signals)//2]
            late = signals[len(signals)//2:]
            early_avg = sum(s["weighted_score"] for s in early) / len(early)
            late_avg = sum(s["weighted_score"] for s in late) / len(late)
            momentum = late_avg - early_avg
        else:
            momentum = 0

        return {
            "sentiment_score": avg_sentiment,
            "sentiment_momentum": momentum,
            "volume": total_volume,
            "confidence": min(total_volume / 50, 1.0),
            "timestamp": end_time,
            "sources": list(set(s["source"] for s in signals))
        }


class MockSentimentSource(SentimentSource):
    """Mock sentiment source for testing."""

    def __init__(self, base_sentiment: float = 0.0):
        self.base_sentiment = base_sentiment

    async def fetch_sentiment(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> list[SentimentReading]:
        import random

        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=7)

        readings = []
        current = start_time
        while current <= end_time:
            noise = random.gauss(0, 0.2)
            score = max(-1, min(1, self.base_sentiment + noise))

            readings.append(SentimentReading(
                timestamp=current,
                score=score,
                magnitude=random.uniform(0.3, 0.8),
                volume=random.randint(10, 200),
                source="mock"
            ))
            current += timedelta(hours=6)

        return readings

    async def fetch_news(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> list[NewsItem]:
        import random

        items = []
        for i in range(min(limit, 10)):
            items.append(NewsItem(
                id=f"mock_{i}",
                title=f"Mock article about {query}",
                source="MockNews",
                published_at=datetime.now() - timedelta(hours=i),
                sentiment_score=random.uniform(-0.5, 0.5)
            ))
        return items
