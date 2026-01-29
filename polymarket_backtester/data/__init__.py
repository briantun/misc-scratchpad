"""
Data connectors for fetching market data, signals, and external information.

All connectors implement a common interface for easy swapping and testing.
"""
from .base import (
    DataConnector,
    SignalConnector,
    ConnectorRegistry,
    ConnectorConfig,
    CacheConfig,
    RateLimitConfig
)
from .polymarket import PolymarketConnector, MockPolymarketConnector
from .polling import (
    PollingConnector,
    FiveThirtyEightSource,
    RealClearPoliticsSource,
    MockPollingSource
)
from .sentiment import (
    SentimentConnector,
    NewsAPISource,
    TwitterSource,
    MockSentimentSource
)
from .loader import DataLoader, load_backtest_data

__all__ = [
    # Base
    "DataConnector",
    "SignalConnector",
    "ConnectorRegistry",
    "ConnectorConfig",
    "CacheConfig",
    "RateLimitConfig",
    # Polymarket
    "PolymarketConnector",
    "MockPolymarketConnector",
    # Polling
    "PollingConnector",
    "FiveThirtyEightSource",
    "RealClearPoliticsSource",
    "MockPollingSource",
    # Sentiment
    "SentimentConnector",
    "NewsAPISource",
    "TwitterSource",
    "MockSentimentSource",
    # Loader
    "DataLoader",
    "load_backtest_data",
]
