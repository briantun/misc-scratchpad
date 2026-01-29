"""
Base interfaces and utilities for data connectors.

Provides:
- Abstract DataConnector interface
- ConnectorRegistry for managing multiple connectors
- Caching and rate limiting utilities
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, TypeVar, Generic
import time
import json
import hashlib
from pathlib import Path


@dataclass
class CacheConfig:
    """Configuration for connector caching."""
    enabled: bool = True
    ttl_seconds: int = 300  # 5 minutes default
    cache_dir: Optional[str] = None  # None = in-memory only
    max_memory_items: int = 1000


@dataclass
class RateLimitConfig:
    """Configuration for API rate limiting."""
    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    retry_on_rate_limit: bool = True
    max_retries: int = 3
    backoff_factor: float = 2.0


@dataclass
class ConnectorConfig:
    """Base configuration for all connectors."""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    timeout_seconds: int = 30
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)


T = TypeVar('T')


class CacheEntry(Generic[T]):
    """A cached value with expiration."""

    def __init__(self, value: T, ttl_seconds: int):
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(seconds=ttl_seconds)

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


class SimpleCache:
    """Simple in-memory cache with optional disk persistence."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._memory_cache: dict[str, CacheEntry] = {}
        self._cache_dir = Path(config.cache_dir) if config.cache_dir else None

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired."""
        if not self.config.enabled:
            return None

        # Check memory cache
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired():
                return entry.value
            else:
                del self._memory_cache[key]

        # Check disk cache
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text())
                    expires_at = datetime.fromisoformat(data["expires_at"])
                    if datetime.now() < expires_at:
                        return data["value"]
                    else:
                        cache_file.unlink()
                except (json.JSONDecodeError, KeyError):
                    pass

        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if not self.config.enabled:
            return

        entry = CacheEntry(value, self.config.ttl_seconds)

        # Memory cache with LRU-like eviction
        if len(self._memory_cache) >= self.config.max_memory_items:
            # Remove oldest entry
            oldest_key = min(self._memory_cache.keys(),
                           key=lambda k: self._memory_cache[k].created_at)
            del self._memory_cache[oldest_key]

        self._memory_cache[key] = entry

        # Disk cache
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key}.json"
            try:
                cache_file.write_text(json.dumps({
                    "value": value,
                    "expires_at": entry.expires_at.isoformat()
                }, default=str))
            except (TypeError, OSError):
                pass  # Silently fail disk caching

    def clear(self) -> None:
        """Clear all cached values."""
        self._memory_cache.clear()
        if self._cache_dir:
            for f in self._cache_dir.glob("*.json"):
                f.unlink()


class RateLimiter:
    """Simple rate limiter with exponential backoff."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._request_times: list[float] = []
        self._last_request: float = 0

    def wait_if_needed(self) -> None:
        """Block until we can make another request."""
        now = time.time()

        # Clean old request times (older than 1 minute)
        self._request_times = [t for t in self._request_times if now - t < 60]

        # Check per-minute limit
        if len(self._request_times) >= self.config.requests_per_minute:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Check per-second limit
        min_interval = 1.0 / self.config.requests_per_second
        time_since_last = now - self._last_request
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)

        # Record this request
        self._last_request = time.time()
        self._request_times.append(self._last_request)

    def record_request(self) -> None:
        """Record that a request was made."""
        now = time.time()
        self._last_request = now
        self._request_times.append(now)


class DataConnector(ABC):
    """
    Abstract base class for all data connectors.

    Provides:
    - Standardized interface for fetching data
    - Built-in caching
    - Rate limiting
    - Error handling
    """

    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.cache = SimpleCache(config.cache)
        self.rate_limiter = RateLimiter(config.rate_limit)
        self._is_connected = False

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the data source.
        Returns True if successful.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the connection is healthy."""
        pass

    @abstractmethod
    async def fetch_markets(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """
        Fetch available markets.

        Args:
            category: Filter by category (politics, sports, crypto, etc.)
            status: Filter by status (open, closed, resolved)
            limit: Maximum number of markets to return

        Returns:
            List of market dictionaries
        """
        pass

    @abstractmethod
    async def fetch_market_prices(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        resolution: str = "1h"
    ) -> list[dict]:
        """
        Fetch historical price data for a market.

        Args:
            market_id: Market identifier
            start_time: Start of time range
            end_time: End of time range
            resolution: Time resolution (1m, 5m, 1h, 1d)

        Returns:
            List of price point dictionaries
        """
        pass

    @abstractmethod
    async def fetch_current_price(self, market_id: str) -> dict:
        """
        Fetch current price for a market.

        Args:
            market_id: Market identifier

        Returns:
            Current price dictionary with bid/ask/last
        """
        pass

    def _get_cached(self, method: str, *args, **kwargs) -> Optional[Any]:
        """Get cached result for a method call."""
        key = self.cache._make_key(self.name, method, *args, **kwargs)
        return self.cache.get(key)

    def _set_cached(self, method: str, value: Any, *args, **kwargs) -> None:
        """Cache result for a method call."""
        key = self.cache._make_key(self.name, method, *args, **kwargs)
        self.cache.set(key, value)

    async def _with_retry(self, func, *args, **kwargs) -> Any:
        """Execute function with retry on failure."""
        last_error = None
        for attempt in range(self.config.rate_limit.max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.rate_limit.max_retries - 1:
                    sleep_time = self.config.rate_limit.backoff_factor ** attempt
                    time.sleep(sleep_time)
        raise last_error


class SignalConnector(DataConnector):
    """
    Extended interface for connectors that provide trading signals.

    Signals are external data that inform trading decisions:
    - Poll data
    - Sentiment scores
    - Expert predictions
    - News events
    """

    @abstractmethod
    async def fetch_signals(
        self,
        market_id: str,
        signal_types: Optional[list[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> list[dict]:
        """
        Fetch trading signals for a market.

        Args:
            market_id: Market identifier
            signal_types: Types of signals to fetch
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of signal dictionaries
        """
        pass

    @abstractmethod
    async def fetch_current_signals(
        self,
        market_id: str,
        signal_types: Optional[list[str]] = None
    ) -> dict:
        """
        Fetch current signals for a market.

        Args:
            market_id: Market identifier
            signal_types: Types of signals to fetch

        Returns:
            Dictionary of current signals
        """
        pass


class ConnectorRegistry:
    """
    Registry for managing multiple data connectors.

    Allows easy swapping of connectors and aggregation of data
    from multiple sources.
    """

    def __init__(self):
        self._connectors: dict[str, DataConnector] = {}
        self._signal_connectors: dict[str, SignalConnector] = {}

    def register(self, connector: DataConnector) -> None:
        """Register a connector."""
        self._connectors[connector.name] = connector
        if isinstance(connector, SignalConnector):
            self._signal_connectors[connector.name] = connector

    def unregister(self, name: str) -> None:
        """Unregister a connector by name."""
        if name in self._connectors:
            del self._connectors[name]
        if name in self._signal_connectors:
            del self._signal_connectors[name]

    def get(self, name: str) -> Optional[DataConnector]:
        """Get a connector by name."""
        return self._connectors.get(name)

    def get_signal_connector(self, name: str) -> Optional[SignalConnector]:
        """Get a signal connector by name."""
        return self._signal_connectors.get(name)

    def list_connectors(self) -> list[str]:
        """List all registered connector names."""
        return list(self._connectors.keys())

    def list_signal_connectors(self) -> list[str]:
        """List all registered signal connector names."""
        return list(self._signal_connectors.keys())

    async def connect_all(self) -> dict[str, bool]:
        """Connect all registered connectors."""
        results = {}
        for name, connector in self._connectors.items():
            try:
                results[name] = await connector.connect()
            except Exception as e:
                results[name] = False
        return results

    async def disconnect_all(self) -> None:
        """Disconnect all registered connectors."""
        for connector in self._connectors.values():
            try:
                await connector.disconnect()
            except Exception:
                pass

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all connectors."""
        results = {}
        for name, connector in self._connectors.items():
            try:
                results[name] = await connector.health_check()
            except Exception:
                results[name] = False
        return results
