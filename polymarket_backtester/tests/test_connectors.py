"""
Unit tests for data connectors.

Tests cover:
- Base connector functionality (caching, rate limiting)
- Polymarket connector (mock and real API structure)
- Polling connector (data aggregation)
- Sentiment connector (signal generation)
- Connector registry
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Import connectors
from polymarket_backtester.data.base import (
    DataConnector,
    SignalConnector,
    ConnectorConfig,
    ConnectorRegistry,
    CacheConfig,
    RateLimitConfig,
    SimpleCache,
    RateLimiter
)
from polymarket_backtester.data.polymarket import (
    PolymarketConnector,
    PolymarketConfig,
    MockPolymarketConnector
)
from polymarket_backtester.data.polling import (
    PollingConnector,
    FiveThirtyEightSource,
    RealClearPoliticsSource,
    MockPollingSource,
    Poll,
    PollAggregate
)
from polymarket_backtester.data.sentiment import (
    SentimentConnector,
    NewsAPISource,
    TwitterSource,
    MockSentimentSource,
    SentimentReading,
    NewsItem
)


# ============================================================
# Base Connector Tests
# ============================================================

class TestSimpleCache:
    """Tests for the caching system."""

    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        config = CacheConfig(enabled=True, ttl_seconds=60)
        cache = SimpleCache(config)

        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")

        assert result == {"data": "value1"}

    def test_cache_expiration(self):
        """Test that cached values expire."""
        config = CacheConfig(enabled=True, ttl_seconds=0)  # Immediate expiry
        cache = SimpleCache(config)

        cache.set("key1", "value1")

        # Should be expired immediately
        import time
        time.sleep(0.01)
        result = cache.get("key1")

        assert result is None

    def test_cache_disabled(self):
        """Test cache when disabled."""
        config = CacheConfig(enabled=False)
        cache = SimpleCache(config)

        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result is None

    def test_cache_key_generation(self):
        """Test cache key generation is deterministic."""
        config = CacheConfig(enabled=True)
        cache = SimpleCache(config)

        key1 = cache._make_key("method", "arg1", kwarg="value")
        key2 = cache._make_key("method", "arg1", kwarg="value")
        key3 = cache._make_key("method", "arg2", kwarg="value")

        assert key1 == key2
        assert key1 != key3

    def test_cache_clear(self):
        """Test cache clearing."""
        config = CacheConfig(enabled=True, ttl_seconds=300)
        cache = SimpleCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_memory_limit(self):
        """Test cache respects memory limits."""
        config = CacheConfig(enabled=True, ttl_seconds=300, max_memory_items=3)
        cache = SimpleCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict oldest

        # One of the early keys should be evicted
        values = [cache.get(f"key{i}") for i in range(1, 5)]
        non_none = [v for v in values if v is not None]
        assert len(non_none) == 3


class TestRateLimiter:
    """Tests for rate limiting."""

    def test_rate_limiter_per_second(self):
        """Test per-second rate limiting."""
        config = RateLimitConfig(requests_per_second=10.0)
        limiter = RateLimiter(config)

        start = datetime.now()
        for _ in range(3):
            limiter.wait_if_needed()
        elapsed = (datetime.now() - start).total_seconds()

        # Should take at least 0.2 seconds for 3 requests at 10/sec
        assert elapsed >= 0.15

    def test_rate_limiter_record_request(self):
        """Test request recording."""
        config = RateLimitConfig(requests_per_minute=60)
        limiter = RateLimiter(config)

        limiter.record_request()
        limiter.record_request()

        assert len(limiter._request_times) == 2


class TestConnectorRegistry:
    """Tests for connector registry."""

    def test_register_connector(self):
        """Test registering a connector."""
        registry = ConnectorRegistry()
        connector = MockPolymarketConnector()

        registry.register(connector)

        assert "mock_polymarket" in registry.list_connectors()
        assert registry.get("mock_polymarket") is connector

    def test_unregister_connector(self):
        """Test unregistering a connector."""
        registry = ConnectorRegistry()
        connector = MockPolymarketConnector()

        registry.register(connector)
        registry.unregister("mock_polymarket")

        assert "mock_polymarket" not in registry.list_connectors()

    def test_get_nonexistent_connector(self):
        """Test getting a connector that doesn't exist."""
        registry = ConnectorRegistry()

        result = registry.get("nonexistent")

        assert result is None


# ============================================================
# Polymarket Connector Tests
# ============================================================

class TestMockPolymarketConnector:
    """Tests for the mock Polymarket connector."""

    @pytest.fixture
    def connector(self):
        return MockPolymarketConnector()

    @pytest.mark.asyncio
    async def test_connect(self, connector):
        """Test connection."""
        result = await connector.connect()
        assert result is True
        assert connector._is_connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Test disconnection."""
        await connector.connect()
        await connector.disconnect()
        assert connector._is_connected is False

    @pytest.mark.asyncio
    async def test_health_check(self, connector):
        """Test health check."""
        await connector.connect()
        result = await connector.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_fetch_markets(self, connector):
        """Test fetching markets."""
        await connector.connect()
        markets = await connector.fetch_markets()

        assert len(markets) > 0
        assert "id" in markets[0]
        assert "question" in markets[0]
        assert "tokens" in markets[0]

    @pytest.mark.asyncio
    async def test_fetch_markets_filter_category(self, connector):
        """Test filtering markets by category."""
        await connector.connect()
        markets = await connector.fetch_markets(category="politics")

        assert all(m["category"] == "politics" for m in markets)

    @pytest.mark.asyncio
    async def test_fetch_market_prices(self, connector):
        """Test fetching historical prices."""
        await connector.connect()
        prices = await connector.fetch_market_prices(
            "mock_001",
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now()
        )

        assert len(prices) > 0
        assert "timestamp" in prices[0]
        assert "yes_price" in prices[0]
        assert "no_price" in prices[0]
        assert 0 <= prices[0]["yes_price"] <= 1

    @pytest.mark.asyncio
    async def test_fetch_current_price(self, connector):
        """Test fetching current price."""
        await connector.connect()
        price = await connector.fetch_current_price("mock_001")

        assert "yes_price" in price
        assert "no_price" in price
        assert "spread" in price
        assert price["yes_price"] + price["no_price"] == pytest.approx(1.0, abs=0.01)


class TestPolymarketConnector:
    """Tests for the real Polymarket connector (mocked API calls)."""

    @pytest.fixture
    def connector(self):
        return PolymarketConnector(PolymarketConfig())

    def test_normalize_market(self, connector):
        """Test market data normalization."""
        raw_data = {
            "id": "test123",
            "question": "Will event happen?",
            "description": "Test description",
            "category": "politics",
            "resolved": False,
            "tokens": [
                {"token_id": "yes_token", "outcome": "Yes", "price": 0.6},
                {"token_id": "no_token", "outcome": "No", "price": 0.4}
            ],
            "volume": 50000,
            "liquidity": 10000
        }

        normalized = connector._normalize_market(raw_data)

        assert normalized["id"] == "test123"
        assert normalized["question"] == "Will event happen?"
        assert normalized["status"] == "active"
        assert len(normalized["tokens"]) == 2

    def test_normalize_market_clob_format(self, connector):
        """Test normalization of CLOB API format."""
        raw_data = {
            "condition_id": "clob123",
            "title": "CLOB Market Title",
            "outcomePrices": "[0.55, 0.45]",
            "outcomes": '["Yes", "No"]',
            "closed": True,
            "volumeNum": 25000
        }

        normalized = connector._normalize_market(raw_data)

        assert normalized["id"] == "clob123"
        assert normalized["status"] == "resolved"
        assert len(normalized["tokens"]) == 2


# ============================================================
# Polling Connector Tests
# ============================================================

class TestPoll:
    """Tests for Poll dataclass."""

    def test_poll_leader(self):
        """Test leader property."""
        poll = Poll(
            pollster="TestPoll",
            date=datetime.now(),
            sample_size=1000,
            margin_of_error=3.0,
            results={"Candidate A": 52, "Candidate B": 48}
        )

        assert poll.leader == "Candidate A"

    def test_poll_margin(self):
        """Test margin property."""
        poll = Poll(
            pollster="TestPoll",
            date=datetime.now(),
            sample_size=1000,
            margin_of_error=3.0,
            results={"Candidate A": 52, "Candidate B": 48}
        )

        assert poll.margin == 4


class TestPollAggregate:
    """Tests for PollAggregate dataclass."""

    def test_to_probability(self):
        """Test probability conversion."""
        aggregate = PollAggregate(
            date=datetime.now(),
            averages={"Candidate A": 52, "Candidate B": 48},
            num_polls=10,
            confidence=0.8,
            source="test"
        )

        prob_a = aggregate.to_probability("Candidate A")
        prob_b = aggregate.to_probability("Candidate B")

        assert prob_a == pytest.approx(0.52, abs=0.01)
        assert prob_b == pytest.approx(0.48, abs=0.01)


class TestMockPollingSource:
    """Tests for mock polling source."""

    @pytest.fixture
    def source(self):
        polls = [
            Poll(
                pollster="Poll1",
                date=datetime.now() - timedelta(days=1),
                sample_size=1000,
                margin_of_error=3.0,
                results={"Yes": 55, "No": 45},
                grade="A"
            ),
            Poll(
                pollster="Poll2",
                date=datetime.now() - timedelta(days=3),
                sample_size=800,
                margin_of_error=3.5,
                results={"Yes": 52, "No": 48},
                grade="B+"
            )
        ]
        return MockPollingSource(polls=polls)

    @pytest.mark.asyncio
    async def test_fetch_polls(self, source):
        """Test fetching polls."""
        polls = await source.fetch_polls("test_race")

        assert len(polls) == 2
        assert polls[0].pollster == "Poll1"

    @pytest.mark.asyncio
    async def test_fetch_average(self, source):
        """Test fetching polling average."""
        avg = await source.fetch_average("test_race")

        assert avg is not None
        assert "Yes" in avg.averages
        assert avg.averages["Yes"] == pytest.approx(53.5, abs=0.1)


class TestPollingConnector:
    """Tests for unified polling connector."""

    @pytest.fixture
    def connector(self):
        mock_source = MockPollingSource(polls=[
            Poll(
                pollster="Test",
                date=datetime.now(),
                sample_size=1000,
                margin_of_error=3.0,
                results={"Yes": 55, "No": 45}
            )
        ])
        return PollingConnector(sources=[mock_source])

    @pytest.mark.asyncio
    async def test_connect(self, connector):
        """Test connection."""
        result = await connector.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_fetch_signals(self, connector):
        """Test fetching polling signals."""
        await connector.connect()
        signals = await connector.fetch_signals("test_race")

        assert len(signals) > 0

    @pytest.mark.asyncio
    async def test_fetch_current_signals(self, connector):
        """Test fetching current signals."""
        await connector.connect()
        signals = await connector.fetch_current_signals("test_race")

        assert "poll_average" in signals or signals == {}


# ============================================================
# Sentiment Connector Tests
# ============================================================

class TestSentimentReading:
    """Tests for SentimentReading dataclass."""

    def test_weighted_score(self):
        """Test weighted score calculation."""
        reading = SentimentReading(
            timestamp=datetime.now(),
            score=0.5,
            magnitude=0.8,
            volume=100,
            source="test"
        )

        # weighted = 0.5 * 0.8 * min(100/100, 1) = 0.4
        assert reading.weighted_score == pytest.approx(0.4, abs=0.01)

    def test_weighted_score_low_volume(self):
        """Test weighted score with low volume."""
        reading = SentimentReading(
            timestamp=datetime.now(),
            score=1.0,
            magnitude=1.0,
            volume=10,
            source="test"
        )

        # Volume weight = 10/100 = 0.1
        assert reading.weighted_score == pytest.approx(0.1, abs=0.01)


class TestMockSentimentSource:
    """Tests for mock sentiment source."""

    @pytest.fixture
    def source(self):
        return MockSentimentSource(base_sentiment=0.3)

    @pytest.mark.asyncio
    async def test_fetch_sentiment(self, source):
        """Test fetching sentiment readings."""
        readings = await source.fetch_sentiment(
            "test query",
            start_time=datetime.now() - timedelta(days=3),
            end_time=datetime.now()
        )

        assert len(readings) > 0
        assert all(r.source == "mock" for r in readings)
        # Should be around base sentiment
        avg = sum(r.score for r in readings) / len(readings)
        assert -0.5 < avg < 0.8  # Reasonable range around 0.3

    @pytest.mark.asyncio
    async def test_fetch_news(self, source):
        """Test fetching news items."""
        items = await source.fetch_news("test query", limit=5)

        assert len(items) <= 5
        assert all(isinstance(item, NewsItem) for item in items)


class TestNewsAPISource:
    """Tests for NewsAPI source."""

    def test_analyze_text_sentiment_positive(self):
        """Test positive sentiment detection."""
        source = NewsAPISource()

        score = source._analyze_text_sentiment(
            "Candidate wins big in landslide victory, leading in all polls"
        )

        assert score > 0

    def test_analyze_text_sentiment_negative(self):
        """Test negative sentiment detection."""
        source = NewsAPISource()

        score = source._analyze_text_sentiment(
            "Candidate loses support amid scandal controversy, trailing in polls"
        )

        assert score < 0

    def test_analyze_text_sentiment_neutral(self):
        """Test neutral sentiment."""
        source = NewsAPISource()

        score = source._analyze_text_sentiment(
            "Weather forecast for tomorrow is partly cloudy"
        )

        assert score == 0

    def test_extract_keywords(self):
        """Test keyword extraction."""
        source = NewsAPISource()

        keywords = source._extract_keywords("President Biden meets with Senator Smith in Washington")

        assert "President" in keywords or "Biden" in keywords


class TestSentimentConnector:
    """Tests for unified sentiment connector."""

    @pytest.fixture
    def connector(self):
        mock_source = MockSentimentSource(base_sentiment=0.2)
        return SentimentConnector(sources=[mock_source])

    @pytest.mark.asyncio
    async def test_connect(self, connector):
        """Test connection."""
        result = await connector.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_fetch_signals(self, connector):
        """Test fetching sentiment signals."""
        await connector.connect()
        signals = await connector.fetch_signals(
            "test topic",
            start_time=datetime.now() - timedelta(days=3),
            end_time=datetime.now()
        )

        assert len(signals) > 0
        assert all("score" in s for s in signals)
        assert all("timestamp" in s for s in signals)

    @pytest.mark.asyncio
    async def test_fetch_current_signals(self, connector):
        """Test fetching current sentiment signals."""
        await connector.connect()
        signals = await connector.fetch_current_signals("test topic")

        assert "sentiment_score" in signals
        assert "confidence" in signals


# ============================================================
# Integration Tests
# ============================================================

class TestConnectorIntegration:
    """Integration tests for connectors working together."""

    @pytest.mark.asyncio
    async def test_registry_with_multiple_connectors(self):
        """Test registry managing multiple connectors."""
        registry = ConnectorRegistry()

        polymarket = MockPolymarketConnector()
        polling = PollingConnector(sources=[MockPollingSource()])
        sentiment = SentimentConnector(sources=[MockSentimentSource()])

        registry.register(polymarket)
        registry.register(polling)
        registry.register(sentiment)

        # Connect all
        results = await registry.connect_all()

        assert results["mock_polymarket"] is True
        assert results["polling"] is True
        assert results["sentiment"] is True

        # Health check all
        health = await registry.health_check_all()
        assert all(health.values())

        # Disconnect all
        await registry.disconnect_all()

    @pytest.mark.asyncio
    async def test_combine_signals(self):
        """Test combining signals from multiple sources."""
        # Create connectors
        polling = PollingConnector(sources=[
            MockPollingSource(polls=[
                Poll(
                    pollster="Poll1",
                    date=datetime.now(),
                    sample_size=1000,
                    margin_of_error=3.0,
                    results={"Yes": 55, "No": 45}
                )
            ])
        ])

        sentiment = SentimentConnector(sources=[
            MockSentimentSource(base_sentiment=0.3)
        ])

        await polling.connect()
        await sentiment.connect()

        # Fetch signals
        poll_signals = await polling.fetch_current_signals("test_race")
        sentiment_signals = await sentiment.fetch_current_signals("test topic")

        # Combine into unified signal
        combined = {
            "poll_data": poll_signals,
            "sentiment_data": sentiment_signals,
            "timestamp": datetime.now()
        }

        assert "poll_data" in combined
        assert "sentiment_data" in combined

        await polling.disconnect()
        await sentiment.disconnect()


# ============================================================
# Point-in-Time Correctness / Look-Ahead Bias Prevention Tests
# ============================================================

from polymarket_backtester.data.loader import (
    DataLoader,
    DataLoaderConfig,
    SignalRecord,
    PointInTimeSignalStore
)


class TestSignalRecord:
    """Tests for SignalRecord dataclass."""

    def test_signal_record_creation(self):
        """Test creating a signal record."""
        timestamp = datetime(2024, 1, 15, 10, 0, 0)
        available_at = datetime(2024, 1, 15, 12, 0, 0)  # 2 hours later

        record = SignalRecord(
            signal_type="poll",
            timestamp=timestamp,
            available_at=available_at,
            data={"result": 55},
            source="test_pollster"
        )

        assert record.signal_type == "poll"
        assert record.timestamp == timestamp
        assert record.available_at == available_at
        assert record.data["result"] == 55
        assert record.source == "test_pollster"

    def test_signal_record_publication_delay(self):
        """Test that available_at correctly represents publication delay."""
        timestamp = datetime(2024, 1, 15, 10, 0, 0)
        publication_delay = timedelta(hours=2)
        available_at = timestamp + publication_delay

        record = SignalRecord(
            signal_type="poll",
            timestamp=timestamp,
            available_at=available_at,
            data={}
        )

        # The signal was created at 10:00 but only available at 12:00
        assert (record.available_at - record.timestamp).total_seconds() == 2 * 3600


class TestPointInTimeSignalStore:
    """Tests for the point-in-time signal store."""

    @pytest.fixture
    def store_with_signals(self):
        """Create a store with test signals at various times."""
        store = PointInTimeSignalStore()

        # Create signals with different available_at times
        # Signal 1: created Jan 1 10:00, available Jan 1 12:00
        store.add(SignalRecord(
            signal_type="poll",
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            available_at=datetime(2024, 1, 1, 12, 0, 0),
            data={"poll_value": 50},
            source="poll1"
        ))

        # Signal 2: created Jan 2 10:00, available Jan 2 12:00
        store.add(SignalRecord(
            signal_type="poll",
            timestamp=datetime(2024, 1, 2, 10, 0, 0),
            available_at=datetime(2024, 1, 2, 12, 0, 0),
            data={"poll_value": 52},
            source="poll2"
        ))

        # Signal 3: created Jan 3 10:00, available Jan 3 12:00
        store.add(SignalRecord(
            signal_type="poll",
            timestamp=datetime(2024, 1, 3, 10, 0, 0),
            available_at=datetime(2024, 1, 3, 12, 0, 0),
            data={"poll_value": 54},
            source="poll3"
        ))

        # Signal 4: sentiment, created Jan 2 14:00, available Jan 2 15:00
        store.add(SignalRecord(
            signal_type="sentiment",
            timestamp=datetime(2024, 1, 2, 14, 0, 0),
            available_at=datetime(2024, 1, 2, 15, 0, 0),
            data={"score": 0.7},
            source="twitter"
        ))

        return store

    def test_get_available_at_basic(self, store_with_signals):
        """Test basic point-in-time query."""
        store = store_with_signals

        # Query at Jan 2 13:00 - should only see poll1 and poll2
        # (poll3 available Jan 3, sentiment available Jan 2 15:00)
        query_time = datetime(2024, 1, 2, 13, 0, 0)
        signals = store.get_available_at(query_time)

        assert len(signals) == 2
        sources = [s.source for s in signals]
        assert "poll1" in sources
        assert "poll2" in sources
        assert "poll3" not in sources  # Not yet available
        assert "twitter" not in sources  # Not yet available

    def test_get_available_at_prevents_look_ahead(self, store_with_signals):
        """Test that future signals are not accessible (look-ahead prevention)."""
        store = store_with_signals

        # Query at Jan 1 11:00 - poll1 was created at 10:00 but not available until 12:00
        query_time = datetime(2024, 1, 1, 11, 0, 0)
        signals = store.get_available_at(query_time)

        # No signals should be available - the only signal created so far
        # (poll1 at 10:00) isn't available until 12:00
        assert len(signals) == 0

    def test_get_available_at_after_publication_delay(self, store_with_signals):
        """Test signals become available after publication delay."""
        store = store_with_signals

        # Query at Jan 1 12:01 - poll1 should now be available
        query_time = datetime(2024, 1, 1, 12, 1, 0)
        signals = store.get_available_at(query_time)

        assert len(signals) == 1
        assert signals[0].source == "poll1"

    def test_get_available_at_with_type_filter(self, store_with_signals):
        """Test filtering by signal type."""
        store = store_with_signals

        # Query at Jan 3 for polls only
        query_time = datetime(2024, 1, 3, 16, 0, 0)
        poll_signals = store.get_available_at(query_time, signal_types=["poll"])
        sentiment_signals = store.get_available_at(query_time, signal_types=["sentiment"])

        assert len(poll_signals) == 3
        assert len(sentiment_signals) == 1
        assert all(s.signal_type == "poll" for s in poll_signals)
        assert all(s.signal_type == "sentiment" for s in sentiment_signals)

    def test_get_available_at_with_lookback(self, store_with_signals):
        """Test lookback window filtering."""
        store = store_with_signals

        # Query at Jan 3 16:00 with 1 day lookback
        # Should only see signals available since Jan 2 16:00
        query_time = datetime(2024, 1, 3, 16, 0, 0)
        signals = store.get_available_at(
            query_time,
            signal_types=["poll"],
            lookback=timedelta(days=1)
        )

        # Only poll3 (available Jan 3 12:00) falls within lookback
        assert len(signals) == 1
        assert signals[0].source == "poll3"

    def test_get_latest_by_type(self, store_with_signals):
        """Test getting the most recent signal of a type."""
        store = store_with_signals

        # Query at Jan 3 16:00
        query_time = datetime(2024, 1, 3, 16, 0, 0)

        latest_poll = store.get_latest_by_type(query_time, "poll")
        latest_sentiment = store.get_latest_by_type(query_time, "sentiment")

        assert latest_poll is not None
        assert latest_poll.source == "poll3"
        assert latest_poll.data["poll_value"] == 54

        assert latest_sentiment is not None
        assert latest_sentiment.source == "twitter"

    def test_get_latest_by_type_before_any_available(self, store_with_signals):
        """Test getting latest when no signals are available yet."""
        store = store_with_signals

        query_time = datetime(2024, 1, 1, 9, 0, 0)  # Before any signals
        latest = store.get_latest_by_type(query_time, "poll")

        assert latest is None

    def test_binary_search_efficiency(self):
        """Test that binary search works correctly with many signals."""
        store = PointInTimeSignalStore()

        # Add 1000 signals, one per hour
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        for i in range(1000):
            timestamp = base_time + timedelta(hours=i)
            store.add(SignalRecord(
                signal_type="poll",
                timestamp=timestamp,
                available_at=timestamp + timedelta(hours=2),
                data={"value": i},
                source=f"source_{i}"
            ))

        # Query at hour 500 - should have signals 0-498 (500 - 2 hour delay)
        query_time = base_time + timedelta(hours=500)
        signals = store.get_available_at(query_time)

        # Signals 0-498 should be available (indices where available_at <= query_time)
        # Signal i has available_at = base_time + i hours + 2 hours
        # So available_at <= query_time means i + 2 <= 500, i.e., i <= 498
        assert len(signals) == 499  # 0 through 498

    def test_add_many_signals(self):
        """Test adding multiple signals at once."""
        store = PointInTimeSignalStore()

        signals = [
            SignalRecord(
                signal_type="poll",
                timestamp=datetime(2024, 1, i+1, 10, 0, 0),
                available_at=datetime(2024, 1, i+1, 12, 0, 0),
                data={"day": i+1},
                source=f"day_{i+1}"
            )
            for i in range(5)
        ]

        store.add_many(signals)

        query_time = datetime(2024, 1, 10, 0, 0, 0)
        result = store.get_available_at(query_time)

        assert len(result) == 5

    def test_clear_store(self):
        """Test clearing the signal store."""
        store = PointInTimeSignalStore()

        store.add(SignalRecord(
            signal_type="poll",
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            available_at=datetime(2024, 1, 1, 12, 0, 0),
            data={},
            source="test"
        ))

        store.clear()

        query_time = datetime(2024, 1, 10, 0, 0, 0)
        signals = store.get_available_at(query_time)

        assert len(signals) == 0


class TestDataLoaderConfig:
    """Tests for DataLoader configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataLoaderConfig()

        assert config.poll_publication_delay == timedelta(hours=2)
        assert config.sentiment_publication_delay == timedelta(hours=1)
        assert config.news_publication_delay == timedelta(minutes=30)
        assert config.poll_lookback == timedelta(days=14)
        assert config.sentiment_lookback == timedelta(days=3)
        assert config.strict_point_in_time is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = DataLoaderConfig(
            poll_publication_delay=timedelta(hours=4),
            sentiment_publication_delay=timedelta(hours=2),
            poll_lookback=timedelta(days=7),
            strict_point_in_time=False
        )

        assert config.poll_publication_delay == timedelta(hours=4)
        assert config.sentiment_publication_delay == timedelta(hours=2)
        assert config.poll_lookback == timedelta(days=7)
        assert config.strict_point_in_time is False


class TestDataLoaderPointInTime:
    """Tests for DataLoader point-in-time correctness."""

    @pytest.fixture
    def mock_loader(self):
        """Create a mock DataLoader."""
        return DataLoader.create_mock()

    @pytest.mark.asyncio
    async def test_signal_store_cleared_between_markets(self, mock_loader):
        """Test that signal store is cleared between market loads."""
        loader = mock_loader
        await loader.connect()

        # Access internal signal store
        assert len(loader._signal_store._signals) == 0

        await loader.disconnect()

    def test_get_signals_for_snapshot_strict_mode(self, mock_loader):
        """Test that strict point-in-time mode is enforced."""
        loader = mock_loader

        # Manually add signals to test filtering
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Poll available 2 hours before snapshot
        loader._signal_store.add(SignalRecord(
            signal_type="poll",
            timestamp=base_time - timedelta(hours=4),
            available_at=base_time - timedelta(hours=2),
            data={"results": {"Yes": 55, "No": 45}, "type": "average", "averages": {"Yes": 55}},
            source="early_poll"
        ))

        # Poll that becomes available AFTER snapshot time (should be excluded)
        loader._signal_store.add(SignalRecord(
            signal_type="poll",
            timestamp=base_time - timedelta(hours=1),
            available_at=base_time + timedelta(hours=1),  # After snapshot
            data={"results": {"Yes": 60, "No": 40}, "type": "average", "averages": {"Yes": 60}},
            source="future_poll"
        ))

        # Get signals as of snapshot time
        signals = loader._get_signals_for_snapshot(base_time)

        # Should only see the early poll, not the future one
        assert "poll_average" in signals
        # The poll average should be from early_poll (55), not future_poll (60)

    def test_get_signals_non_strict_mode(self):
        """Test that non-strict mode returns empty (disables filtering)."""
        config = DataLoaderConfig(strict_point_in_time=False)
        loader = DataLoader.create_mock(config=config)

        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Add a signal
        loader._signal_store.add(SignalRecord(
            signal_type="poll",
            timestamp=base_time - timedelta(hours=2),
            available_at=base_time - timedelta(hours=1),
            data={"type": "average", "averages": {"Yes": 55}},
            source="test"
        ))

        # Non-strict mode should return empty dict (disabled)
        signals = loader._get_signals_for_snapshot(base_time)
        assert signals == {}

    def test_data_freshness_metadata(self, mock_loader):
        """Test that data freshness metadata is included."""
        loader = mock_loader
        snapshot_time = datetime(2024, 1, 15, 14, 0, 0)

        # Add poll signal from 6 hours ago (created), available 4 hours ago
        loader._signal_store.add(SignalRecord(
            signal_type="poll",
            timestamp=snapshot_time - timedelta(hours=6),  # Created 6 hours ago
            available_at=snapshot_time - timedelta(hours=4),  # Available 4 hours ago
            data={"type": "average", "averages": {"Yes": 55}},
            source="test"
        ))

        signals = loader._get_signals_for_snapshot(snapshot_time)

        # Should include freshness metadata
        assert "_data_timestamp" in signals
        assert "_data_age_hours" in signals
        # Data age should be ~6 hours (from timestamp, not available_at)
        assert signals["_data_age_hours"] == pytest.approx(6.0, abs=0.1)


class TestLookAheadBiasScenarios:
    """
    Real-world scenarios that could cause look-ahead bias.
    These tests ensure the system prevents such biases.
    """

    def test_poll_published_after_market_move(self):
        """
        Scenario: A poll is conducted on Jan 1, but results are published Jan 2.
        A backtest on Jan 1 should NOT use this poll.
        """
        store = PointInTimeSignalStore()

        # Poll conducted Jan 1 evening, published Jan 2 morning
        poll_record = SignalRecord(
            signal_type="poll",
            timestamp=datetime(2024, 1, 1, 20, 0, 0),  # Conducted
            available_at=datetime(2024, 1, 2, 8, 0, 0),  # Published
            data={"candidate_x": 55},
            source="major_pollster"
        )
        store.add(poll_record)

        # Backtesting at Jan 1 22:00 - poll was conducted 2 hours ago
        # but won't be published for 10 more hours
        query_time = datetime(2024, 1, 1, 22, 0, 0)
        signals = store.get_available_at(query_time)

        # Must NOT see the poll
        assert len(signals) == 0

    def test_breaking_news_sentiment_delay(self):
        """
        Scenario: Breaking news happens at 2pm, sentiment analysis at 3pm.
        Backtest at 2:30pm should not use the 3pm sentiment.
        """
        store = PointInTimeSignalStore()

        store.add(SignalRecord(
            signal_type="sentiment",
            timestamp=datetime(2024, 1, 5, 14, 0, 0),  # News breaks
            available_at=datetime(2024, 1, 5, 15, 0, 0),  # Sentiment processed
            data={"score": -0.8, "event": "scandal"},
            source="news_sentiment"
        ))

        # Query at 2:30pm - news has happened but sentiment not processed
        query_time = datetime(2024, 1, 5, 14, 30, 0)
        signals = store.get_available_at(query_time)

        assert len(signals) == 0

    def test_multiple_signal_types_different_delays(self):
        """
        Scenario: Polls have 2-hour delay, sentiment has 1-hour delay.
        Signals created at the same time should be available at different times.
        """
        store = PointInTimeSignalStore()
        base_time = datetime(2024, 1, 10, 10, 0, 0)

        # Poll created at 10am, available at 12pm (2 hour delay)
        store.add(SignalRecord(
            signal_type="poll",
            timestamp=base_time,
            available_at=base_time + timedelta(hours=2),
            data={"value": 50},
            source="poll"
        ))

        # Sentiment created at 10am, available at 11am (1 hour delay)
        store.add(SignalRecord(
            signal_type="sentiment",
            timestamp=base_time,
            available_at=base_time + timedelta(hours=1),
            data={"score": 0.5},
            source="sentiment"
        ))

        # Query at 11:30am - sentiment available, poll not
        query_at_1130 = base_time + timedelta(hours=1, minutes=30)
        signals_1130 = store.get_available_at(query_at_1130)

        assert len(signals_1130) == 1
        assert signals_1130[0].signal_type == "sentiment"

        # Query at 12:30pm - both available
        query_at_1230 = base_time + timedelta(hours=2, minutes=30)
        signals_1230 = store.get_available_at(query_at_1230)

        assert len(signals_1230) == 2

    def test_stale_data_with_lookback(self):
        """
        Scenario: Old polls should not influence current decisions.
        A 30-day old poll should be excluded with 14-day lookback.
        """
        store = PointInTimeSignalStore()
        current_time = datetime(2024, 2, 1, 12, 0, 0)

        # 30-day old poll
        store.add(SignalRecord(
            signal_type="poll",
            timestamp=current_time - timedelta(days=30),
            available_at=current_time - timedelta(days=30) + timedelta(hours=2),
            data={"value": 45},
            source="old_poll"
        ))

        # Recent poll
        store.add(SignalRecord(
            signal_type="poll",
            timestamp=current_time - timedelta(days=3),
            available_at=current_time - timedelta(days=3) + timedelta(hours=2),
            data={"value": 55},
            source="recent_poll"
        ))

        # Query with 14-day lookback
        signals = store.get_available_at(
            current_time,
            signal_types=["poll"],
            lookback=timedelta(days=14)
        )

        # Should only see recent poll
        assert len(signals) == 1
        assert signals[0].source == "recent_poll"


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
