"""
Data loader that integrates connectors with the backtesting engine.

CRITICAL: Implements point-in-time correctness to avoid look-ahead bias.
When simulating time T, we only use data that was available at time T.

Provides utilities to:
- Load real market data from Polymarket
- Enrich market snapshots with polling and sentiment signals
- Convert connector data to engine-compatible formats
"""
import asyncio
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from ..models import (
    Market, MarketSnapshot, MarketStatus, PricePoint, Side
)
from .base import ConnectorRegistry, DataConnector, SignalConnector
from .polymarket import PolymarketConnector, MockPolymarketConnector
from .polling import PollingConnector, MockPollingSource
from .sentiment import SentimentConnector, MockSentimentSource


@dataclass
class SignalRecord:
    """
    A signal with timing metadata for point-in-time correctness.

    Attributes:
        signal_type: Type of signal (poll, sentiment, etc.)
        timestamp: When the signal was generated/published
        available_at: When the signal became available for trading decisions
                     (may be later than timestamp due to publication delay)
        data: The actual signal data
    """
    signal_type: str
    timestamp: datetime
    available_at: datetime
    data: dict
    source: str = ""


class PointInTimeSignalStore:
    """
    Stores signals indexed by time for efficient point-in-time queries.

    Ensures we only access signals that were available at the query time,
    preventing look-ahead bias in backtesting.
    """

    def __init__(self):
        # Signals sorted by available_at time
        self._signals: list[SignalRecord] = []
        self._sorted = True

    def add(self, signal: SignalRecord) -> None:
        """Add a signal to the store."""
        self._signals.append(signal)
        self._sorted = False

    def add_many(self, signals: list[SignalRecord]) -> None:
        """Add multiple signals."""
        self._signals.extend(signals)
        self._sorted = False

    def _ensure_sorted(self) -> None:
        """Ensure signals are sorted by available_at."""
        if not self._sorted:
            self._signals.sort(key=lambda s: s.available_at)
            self._sorted = True

    def get_available_at(
        self,
        query_time: datetime,
        signal_types: Optional[list[str]] = None,
        lookback: Optional[timedelta] = None
    ) -> list[SignalRecord]:
        """
        Get all signals available at a specific time.

        Args:
            query_time: The simulated current time
            signal_types: Filter by signal types (None = all)
            lookback: Only include signals from the last N time period
                     (None = all historical signals)

        Returns:
            List of signals that were available at query_time
        """
        self._ensure_sorted()

        # Find the rightmost signal with available_at <= query_time
        # Using binary search for efficiency
        available_times = [s.available_at for s in self._signals]
        idx = bisect_right(available_times, query_time)

        # Get all signals up to this index
        available_signals = self._signals[:idx]

        # Apply lookback filter
        if lookback:
            cutoff = query_time - lookback
            available_signals = [s for s in available_signals if s.available_at >= cutoff]

        # Apply type filter
        if signal_types:
            available_signals = [s for s in available_signals if s.signal_type in signal_types]

        return available_signals

    def get_latest_by_type(
        self,
        query_time: datetime,
        signal_type: str,
        lookback: Optional[timedelta] = None
    ) -> Optional[SignalRecord]:
        """Get the most recent signal of a given type available at query_time."""
        signals = self.get_available_at(query_time, signal_types=[signal_type], lookback=lookback)
        return signals[-1] if signals else None

    def clear(self) -> None:
        """Clear all signals."""
        self._signals.clear()
        self._sorted = True


@dataclass
class DataLoaderConfig:
    """Configuration for data loading with timing parameters."""
    # Publication delays - time between signal creation and availability
    poll_publication_delay: timedelta = field(default_factory=lambda: timedelta(hours=2))
    sentiment_publication_delay: timedelta = field(default_factory=lambda: timedelta(hours=1))
    news_publication_delay: timedelta = field(default_factory=lambda: timedelta(minutes=30))

    # How far back to look for signals when building a snapshot
    poll_lookback: timedelta = field(default_factory=lambda: timedelta(days=14))
    sentiment_lookback: timedelta = field(default_factory=lambda: timedelta(days=3))

    # Whether to use strict point-in-time filtering
    strict_point_in_time: bool = True


class DataLoader:
    """
    Loads and transforms data from connectors for the backtesting engine.

    CRITICAL: Implements point-in-time correctness. When simulating a trade
    at time T, we only use data that would have been available at time T.
    """

    def __init__(
        self,
        polymarket: Optional[DataConnector] = None,
        polling: Optional[SignalConnector] = None,
        sentiment: Optional[SignalConnector] = None,
        config: Optional[DataLoaderConfig] = None
    ):
        """
        Initialize with data connectors.

        Args:
            polymarket: Connector for market data (prices, order book)
            polling: Connector for polling signals
            sentiment: Connector for sentiment signals
            config: Timing configuration for point-in-time correctness
        """
        self.polymarket = polymarket
        self.polling = polling
        self.sentiment = sentiment
        self.config = config or DataLoaderConfig()
        self._connected = False
        self._signal_store = PointInTimeSignalStore()

    @classmethod
    def create_mock(cls, config: Optional[DataLoaderConfig] = None) -> "DataLoader":
        """Create a DataLoader with mock connectors for testing."""
        return cls(
            polymarket=MockPolymarketConnector(),
            polling=PollingConnector(sources=[MockPollingSource()]),
            sentiment=SentimentConnector(sources=[MockSentimentSource()]),
            config=config
        )

    @classmethod
    def create_live(
        cls,
        polymarket_api_key: Optional[str] = None,
        news_api_key: Optional[str] = None,
        twitter_token: Optional[str] = None,
        config: Optional[DataLoaderConfig] = None
    ) -> "DataLoader":
        """
        Create a DataLoader with live connectors.

        Args:
            polymarket_api_key: Polymarket API key (optional for public data)
            news_api_key: NewsAPI.org API key
            twitter_token: Twitter bearer token
            config: Timing configuration
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

        return cls(
            polymarket=polymarket,
            polling=polling,
            sentiment=sentiment,
            config=config
        )

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

    async def _load_signals_for_market(
        self,
        market: Market,
        start_time: datetime,
        end_time: datetime
    ) -> None:
        """
        Load all signals for a market into the signal store.

        Signals are loaded with proper publication delays applied.
        """
        search_query = self._create_search_query(market.question)

        # Load polling signals
        if self.polling:
            try:
                signals = await self.polling.fetch_signals(
                    search_query,
                    start_time=start_time,
                    end_time=end_time
                )
                for s in signals:
                    timestamp = s.get("timestamp", datetime.now())
                    if isinstance(timestamp, str):
                        timestamp = self._parse_datetime(timestamp) or datetime.now()

                    # Apply publication delay
                    available_at = timestamp + self.config.poll_publication_delay

                    self._signal_store.add(SignalRecord(
                        signal_type="poll",
                        timestamp=timestamp,
                        available_at=available_at,
                        data=s,
                        source=s.get("source", "unknown")
                    ))
            except Exception:
                pass

        # Load sentiment signals
        if self.sentiment:
            try:
                signals = await self.sentiment.fetch_signals(
                    search_query,
                    start_time=start_time,
                    end_time=end_time
                )
                for s in signals:
                    timestamp = s.get("timestamp", datetime.now())
                    if isinstance(timestamp, str):
                        timestamp = self._parse_datetime(timestamp) or datetime.now()

                    # Apply publication delay
                    available_at = timestamp + self.config.sentiment_publication_delay

                    self._signal_store.add(SignalRecord(
                        signal_type="sentiment",
                        timestamp=timestamp,
                        available_at=available_at,
                        data=s,
                        source=s.get("source", "unknown")
                    ))
            except Exception:
                pass

    def _get_signals_for_snapshot(self, snapshot_time: datetime) -> dict:
        """
        Get external signals available at snapshot time.

        CRITICAL: Only returns signals that were available BEFORE snapshot_time.
        This prevents look-ahead bias.
        """
        if not self.config.strict_point_in_time:
            # Non-strict mode (not recommended for backtesting)
            return {}

        external_signals = {}

        # Get available poll signals (looking back poll_lookback period)
        poll_signals = self._signal_store.get_available_at(
            query_time=snapshot_time,
            signal_types=["poll"],
            lookback=self.config.poll_lookback
        )
        if poll_signals:
            aggregated = self._aggregate_poll_signals([s.data for s in poll_signals])
            if aggregated:
                # Add metadata about data freshness
                latest_poll = poll_signals[-1]
                aggregated["_data_timestamp"] = latest_poll.timestamp.isoformat()
                aggregated["_data_age_hours"] = (snapshot_time - latest_poll.timestamp).total_seconds() / 3600
                external_signals.update(aggregated)

        # Get available sentiment signals
        sentiment_signals = self._signal_store.get_available_at(
            query_time=snapshot_time,
            signal_types=["sentiment"],
            lookback=self.config.sentiment_lookback
        )
        if sentiment_signals:
            aggregated = self._aggregate_sentiment_signals([s.data for s in sentiment_signals])
            if aggregated:
                latest_sentiment = sentiment_signals[-1]
                aggregated["_sentiment_timestamp"] = latest_sentiment.timestamp.isoformat()
                aggregated["_sentiment_age_hours"] = (snapshot_time - latest_sentiment.timestamp).total_seconds() / 3600
                external_signals.update(aggregated)

        return external_signals

    async def load_market_snapshots(
        self,
        market: Market,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        resolution: str = "1h",
        include_signals: bool = True
    ) -> list[MarketSnapshot]:
        """
        Load historical snapshots for a market with point-in-time correctness.

        CRITICAL: Each snapshot only contains signals that were available
        at that snapshot's timestamp. This prevents look-ahead bias.

        Args:
            market: Market to load data for
            start_time: Start of time range
            end_time: End of time range
            resolution: Price data resolution
            include_signals: Whether to include polling/sentiment signals

        Returns:
            List of MarketSnapshot objects with point-in-time correct signals
        """
        if not self.polymarket:
            return []

        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=30)

        # Clear previous signals
        self._signal_store.clear()

        # Fetch price data
        prices = await self.polymarket.fetch_market_prices(
            market.id,
            start_time=start_time,
            end_time=end_time,
            resolution=resolution
        )

        if not prices:
            return []

        # Pre-load all signals for the time range (with buffer for lookback)
        if include_signals:
            signal_start = start_time - max(
                self.config.poll_lookback,
                self.config.sentiment_lookback
            )
            await self._load_signals_for_market(market, signal_start, end_time)

        # Build snapshots with point-in-time correct signals
        snapshots = []
        for price_data in prices:
            timestamp = price_data["timestamp"]
            if isinstance(timestamp, str):
                timestamp = self._parse_datetime(timestamp)

            if timestamp is None:
                continue

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

            # Get ONLY signals available at this timestamp (point-in-time)
            external_signals = {}
            if include_signals:
                external_signals = self._get_signals_for_snapshot(timestamp)

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
                "poll_source": latest.get("source", "unknown"),
                "poll_count": len(polls)
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
            "poll_source": "aggregated",
            "poll_count": len(polls)
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
            "sentiment_confidence": min(len(sentiments) / 10, 1.0),
            "sentiment_count": len(sentiments)
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

    All data is loaded with point-in-time correctness - each snapshot
    only contains signals that were available at that point in time.

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
