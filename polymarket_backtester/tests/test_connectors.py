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
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
