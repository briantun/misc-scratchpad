"""
Data loader that integrates connectors with the backtesting engine.

Provides utilities to:
- Load real market data from Polymarket
- Enrich market snapshots with polling and sentiment signals
- Convert connector data to engine-compatible formats
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional

from ..models import (
    Market, MarketSnapshot, MarketStatus, PricePoint, Side
)
from .base import ConnectorRegistry, DataConnector, SignalConnector
from .polymarket import PolymarketConnector, MockPolymarketConnector
from .polling import PollingConnector, MockPollingSource
from .sentiment import SentimentConnector, MockSentimentSource


class DataLoader:
    """
    Loads and transforms data from connectors for the backtesting engine.

    This class bridges the gap between raw API data and the engine's
    expected data structures.
    """

    def __init__(
        self,
        polymarket: Optional[DataConnector] = None,
        polling: Optional[SignalConnector] = None,
        sentiment: Optional[SignalConnector] = None
    ):
        """
        Initialize with data connectors.

        Args:
            polymarket: Connector for market data (prices, order book)
            polling: Connector for polling signals
            sentiment: Connector for sentiment signals
        """
        self.polymarket = polymarket
        self.polling = polling
        self.sentiment = sentiment
        self._connected = False

    @classmethod
    def create_mock(cls) -> "DataLoader":
        """Create a DataLoader with mock connectors for testing."""
        return cls(
            polymarket=MockPolymarketConnector(),
            polling=PollingConnector(sources=[MockPollingSource()]),
            sentiment=SentimentConnector(sources=[MockSentimentSource()])
        )

    @classmethod
    def create_live(
        cls,
        polymarket_api_key: Optional[str] = None,
        news_api_key: Optional[str] = None,
        twitter_token: Optional[str] = None
    ) -> "DataLoader":
        """
        Create a DataLoader with live connectors.

        Args:
            polymarket_api_key: Polymarket API key (optional for public data)
            news_api_key: NewsAPI.org API key
            twitter_token: Twitter bearer token
        """
        from .polymarket import PolymarketConfig

        polymarket = PolymarketConnector(
            PolymarketConfig(api_key=polymarket_api_key)
        )

        polling = PollingConnector()  # Uses default sources

        sentiment = SentimentConnector(
            news_api_key=news_api_key,
            twitter_bearer_token=twitter_token
        )

        return cls(polymarket=polymarket, polling=polling, sentiment=sentiment)

    async def connect(self) -> bool:
        """Connect all configured connectors."""
        results = []

        if self.polymarket:
            results.append(await self.polymarket.connect())
        if self.polling:
            results.append(await self.polling.connect())
        if self.sentiment:
            results.append(await self.sentiment.connect())

        self._connected = all(results) if results else False
        return self._connected

    async def disconnect(self) -> None:
        """Disconnect all connectors."""
        if self.polymarket:
            await self.polymarket.disconnect()
        if self.polling:
            await self.polling.disconnect()
        if self.sentiment:
            await self.sentiment.disconnect()
        self._connected = False

    async def load_markets(
        self,
        category: Optional[str] = None,
        status: str = "active",
        limit: int = 50
    ) -> list[Market]:
        """
        Load markets from Polymarket.

        Args:
            category: Filter by category
            status: Market status filter
            limit: Maximum markets to load

        Returns:
            List of Market objects
        """
        if not self.polymarket:
            return []

        raw_markets = await self.polymarket.fetch_markets(
            category=category,
            status=status,
            limit=limit
        )

        markets = []
        for raw in raw_markets:
            # Determine resolved outcome if available
            resolved_outcome = None
            if raw.get("resolved"):
                resolution = raw.get("resolution", "").lower()
                if "yes" in resolution:
                    resolved_outcome = Side.YES
                elif "no" in resolution:
                    resolved_outcome = Side.NO

            # Parse dates
            created_at = self._parse_datetime(raw.get("created_at"))
            end_date = self._parse_datetime(raw.get("end_date"))

            market = Market(
                id=raw["id"],
                question=raw.get("question", "Unknown"),
                category=raw.get("category", "general"),
                created_at=created_at or datetime.now(),
                resolution_date=end_date,
                status=MarketStatus.RESOLVED if raw.get("resolved") else MarketStatus.OPEN,
                resolved_outcome=resolved_outcome,
                tags=raw.get("tags", []),
                related_market_ids=[]
            )
            markets.append(market)

        return markets

    async def load_market_snapshots(
        self,
        market: Market,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        resolution: str = "1h",
        include_signals: bool = True
    ) -> list[MarketSnapshot]:
        """
        Load historical snapshots for a market.

        Args:
            market: Market to load data for
            start_time: Start of time range
            end_time: End of time range
            resolution: Price data resolution
            include_signals: Whether to include polling/sentiment signals

        Returns:
            List of MarketSnapshot objects
        """
        if not self.polymarket:
            return []

        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=30)

        # Fetch price data
        prices = await self.polymarket.fetch_market_prices(
            market.id,
            start_time=start_time,
            end_time=end_time,
            resolution=resolution
        )

        if not prices:
            return []

        # Fetch signals if requested
        poll_signals = {}
        sentiment_signals = {}

        if include_signals:
            # Create search query from market question
            search_query = self._create_search_query(market.question)

            if self.polling:
                try:
                    signals = await self.polling.fetch_signals(
                        search_query,
                        start_time=start_time,
                        end_time=end_time
                    )
                    # Index by date for lookup
                    for s in signals:
                        date_key = s["timestamp"].strftime("%Y-%m-%d")
                        if date_key not in poll_signals:
                            poll_signals[date_key] = []
                        poll_signals[date_key].append(s)
                except Exception:
                    pass

            if self.sentiment:
                try:
                    signals = await self.sentiment.fetch_signals(
                        search_query,
                        start_time=start_time,
                        end_time=end_time
                    )
                    for s in signals:
                        date_key = s["timestamp"].strftime("%Y-%m-%d")
                        if date_key not in sentiment_signals:
                            sentiment_signals[date_key] = []
                        sentiment_signals[date_key].append(s)
                except Exception:
                    pass

        # Build snapshots
        snapshots = []
        for price_data in prices:
            timestamp = price_data["timestamp"]
            if isinstance(timestamp, str):
                timestamp = self._parse_datetime(timestamp)

            # Create price point
            yes_price = price_data["yes_price"]
            no_price = price_data.get("no_price", 1 - yes_price)
            volume = price_data.get("volume", 0)

            # Estimate bid/ask from mid price
            spread = 0.02  # Default 2% spread
            price_point = PricePoint(
                timestamp=timestamp,
                yes_price=yes_price,
                no_price=no_price,
                yes_volume=volume * yes_price,
                no_volume=volume * no_price,
                bid_yes=yes_price - spread / 2,
                ask_yes=yes_price + spread / 2,
                bid_no=no_price - spread / 2,
                ask_no=no_price + spread / 2
            )

            # Gather external signals for this timestamp
            external_signals = {}
            date_key = timestamp.strftime("%Y-%m-%d")

            # Add polling signals
            if date_key in poll_signals:
                # Average the polls for this date
                day_polls = poll_signals[date_key]
                if day_polls:
                    avg_signal = self._aggregate_poll_signals(day_polls)
                    external_signals.update(avg_signal)

            # Add sentiment signals
            if date_key in sentiment_signals:
                day_sentiment = sentiment_signals[date_key]
                if day_sentiment:
                    avg_sentiment = self._aggregate_sentiment_signals(day_sentiment)
                    external_signals.update(avg_sentiment)

            snapshot = MarketSnapshot(
                market=market,
                price=price_point,
                external_signals=external_signals
            )
            snapshots.append(snapshot)

        return snapshots

    def _parse_datetime(self, value) -> Optional[datetime]:
        """Parse various datetime formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)

        # Try string formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(str(value).replace("+00:00", "Z"), fmt)
            except ValueError:
                continue
        return None

    def _create_search_query(self, question: str) -> str:
        """
        Create a search query from a market question.

        Extracts key terms for polling and sentiment search.
        """
        # Remove common question words
        stopwords = {"will", "the", "a", "an", "be", "is", "are", "was", "were",
                    "by", "on", "in", "at", "to", "for", "of", "with", "?"}

        words = question.lower().split()
        keywords = [w.strip("?.,!") for w in words if w.lower() not in stopwords]

        # Return top keywords
        return " ".join(keywords[:5])

    def _aggregate_poll_signals(self, polls: list[dict]) -> dict:
        """Aggregate multiple poll signals into one."""
        if not polls:
            return {}

        # Find averages if present
        averages = [p for p in polls if p.get("type") == "average"]
        if averages:
            latest = averages[-1]
            return {
                "poll_average": latest.get("averages", {}),
                "poll_confidence": latest.get("confidence", 0.5),
                "poll_source": latest.get("source", "unknown")
            }

        # Otherwise aggregate individual polls
        all_results = {}
        for poll in polls:
            for candidate, pct in poll.get("results", {}).items():
                if candidate not in all_results:
                    all_results[candidate] = []
                all_results[candidate].append(pct)

        if not all_results:
            return {}

        averages = {c: sum(v) / len(v) for c, v in all_results.items()}

        return {
            "poll_average": averages,
            "poll_confidence": min(len(polls) / 5, 1.0),
            "poll_source": "aggregated"
        }

    def _aggregate_sentiment_signals(self, sentiments: list[dict]) -> dict:
        """Aggregate sentiment signals."""
        if not sentiments:
            return {}

        scores = [s.get("score", 0) for s in sentiments]
        volumes = [s.get("volume", 1) for s in sentiments]

        # Volume-weighted average
        total_volume = sum(volumes)
        if total_volume == 0:
            return {}

        weighted_score = sum(s * v for s, v in zip(scores, volumes)) / total_volume

        return {
            "sentiment_score": weighted_score,
            "sentiment_volume": total_volume,
            "sentiment_confidence": min(len(sentiments) / 10, 1.0)
        }


async def load_backtest_data(
    loader: DataLoader,
    category: Optional[str] = None,
    market_limit: int = 10,
    days: int = 60,
    resolution: str = "1h"
) -> dict[str, list[MarketSnapshot]]:
    """
    Convenience function to load data for backtesting.

    Args:
        loader: Configured DataLoader
        category: Market category filter
        market_limit: Max markets to load
        days: Days of historical data
        resolution: Price resolution

    Returns:
        Dict of market_id -> list of snapshots
    """
    await loader.connect()

    try:
        # Load markets
        markets = await loader.load_markets(
            category=category,
            status="active",
            limit=market_limit
        )

        # Load snapshots for each market
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        result = {}
        for market in markets:
            snapshots = await loader.load_market_snapshots(
                market,
                start_time=start_time,
                end_time=end_time,
                resolution=resolution,
                include_signals=True
            )
            if snapshots:
                result[market.id] = snapshots

        return result

    finally:
        await loader.disconnect()
