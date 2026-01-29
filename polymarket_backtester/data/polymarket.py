"""
Polymarket API connector.

Provides access to:
- Market listings and metadata
- Real-time and historical prices
- Order book data
- Trade history

Uses the Polymarket CLOB API and Gamma API.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Any
import json

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from .base import DataConnector, ConnectorConfig, CacheConfig, RateLimitConfig


@dataclass
class PolymarketConfig(ConnectorConfig):
    """Configuration specific to Polymarket connector."""
    clob_base_url: str = "https://clob.polymarket.com"
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    strapi_base_url: str = "https://strapi-matic.poly.market"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            name="polymarket",
            api_key=api_key,
            api_secret=api_secret,
            **kwargs
        )


class PolymarketConnector(DataConnector):
    """
    Connector for Polymarket prediction markets.

    Supports:
    - Fetching market listings with filters
    - Historical and real-time price data
    - Order book snapshots
    - Market metadata and resolution info
    """

    def __init__(self, config: Optional[PolymarketConfig] = None):
        if config is None:
            config = PolymarketConfig()
        super().__init__(config)
        self._session: Optional[Any] = None  # aiohttp.ClientSession

    @property
    def clob_url(self) -> str:
        return getattr(self.config, 'clob_base_url', 'https://clob.polymarket.com')

    @property
    def gamma_url(self) -> str:
        return getattr(self.config, 'gamma_base_url', 'https://gamma-api.polymarket.com')

    async def connect(self) -> bool:
        """Establish HTTP session."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for PolymarketConnector. Install with: pip install aiohttp")

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

        self._is_connected = True
        return True

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._is_connected = False

    async def health_check(self) -> bool:
        """Check if Polymarket API is reachable."""
        try:
            async with self._session.get(f"{self.gamma_url}/markets", params={"limit": 1}) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _request(
        self,
        method: str,
        url: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None
    ) -> dict:
        """Make an HTTP request with rate limiting and retries."""
        if self._session is None:
            await self.connect()

        self.rate_limiter.wait_if_needed()

        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        async with self._session.request(method, url, params=params, json=data, headers=headers) as resp:
            if resp.status == 429:  # Rate limited
                retry_after = int(resp.headers.get("Retry-After", 60))
                await asyncio.sleep(retry_after)
                return await self._request(method, url, params, data)

            resp.raise_for_status()
            return await resp.json()

    async def fetch_markets(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None
    ) -> list[dict]:
        """
        Fetch available markets from Polymarket.

        Args:
            category: Filter by category (e.g., "politics", "sports", "crypto")
            status: Filter by status ("active", "resolved", "archived")
            limit: Maximum markets to return
            offset: Pagination offset
            search: Search query for market questions

        Returns:
            List of market dictionaries with structure:
            {
                "id": str,
                "question": str,
                "description": str,
                "category": str,
                "status": str,
                "created_at": str,
                "end_date": str,
                "resolved": bool,
                "resolution": str | None,
                "tokens": [{"token_id": str, "outcome": str, "price": float}]
            }
        """
        # Check cache
        cached = self._get_cached("fetch_markets", category, status, limit, offset, search)
        if cached is not None:
            return cached

        params = {
            "limit": limit,
            "offset": offset,
        }

        if status == "active":
            params["active"] = "true"
            params["closed"] = "false"
        elif status == "resolved":
            params["closed"] = "true"

        if search:
            params["_q"] = search

        # Fetch from Gamma API (more comprehensive market data)
        url = f"{self.gamma_url}/markets"
        result = await self._request("GET", url, params=params)

        # Normalize response
        markets = []
        for m in result if isinstance(result, list) else result.get("data", []):
            market = self._normalize_market(m)

            # Apply category filter client-side if needed
            if category and market.get("category", "").lower() != category.lower():
                continue

            markets.append(market)

        self._set_cached("fetch_markets", markets, category, status, limit, offset, search)
        return markets

    def _normalize_market(self, raw: dict) -> dict:
        """Normalize market data to standard format."""
        # Handle different API response formats
        tokens = []
        if "tokens" in raw:
            for t in raw["tokens"]:
                tokens.append({
                    "token_id": t.get("token_id", t.get("id")),
                    "outcome": t.get("outcome", "unknown"),
                    "price": float(t.get("price", 0.5))
                })
        elif "outcomePrices" in raw:
            # CLOB format
            prices = json.loads(raw["outcomePrices"]) if isinstance(raw["outcomePrices"], str) else raw["outcomePrices"]
            outcomes = json.loads(raw.get("outcomes", "[]")) if isinstance(raw.get("outcomes"), str) else raw.get("outcomes", [])
            for i, price in enumerate(prices):
                tokens.append({
                    "token_id": raw.get("clobTokenIds", [""])[i] if "clobTokenIds" in raw else str(i),
                    "outcome": outcomes[i] if i < len(outcomes) else f"outcome_{i}",
                    "price": float(price)
                })

        return {
            "id": raw.get("id", raw.get("condition_id", "")),
            "question": raw.get("question", raw.get("title", "")),
            "description": raw.get("description", ""),
            "category": raw.get("category", raw.get("tags", ["general"])[0] if raw.get("tags") else "general"),
            "status": "resolved" if raw.get("resolved") or raw.get("closed") else "active",
            "created_at": raw.get("created_at", raw.get("createdAt", "")),
            "end_date": raw.get("end_date", raw.get("endDate", raw.get("end_date_iso", ""))),
            "resolved": raw.get("resolved", False),
            "resolution": raw.get("resolution", raw.get("resolutionSource")),
            "tokens": tokens,
            "volume": float(raw.get("volume", raw.get("volumeNum", 0))),
            "liquidity": float(raw.get("liquidity", raw.get("liquidityNum", 0))),
            "raw": raw  # Keep raw data for debugging
        }

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
            market_id: Market condition ID or token ID
            start_time: Start of time range (default: 30 days ago)
            end_time: End of time range (default: now)
            resolution: Time resolution - "1m", "5m", "15m", "1h", "4h", "1d"

        Returns:
            List of price points:
            {
                "timestamp": datetime,
                "yes_price": float,
                "no_price": float,
                "volume": float
            }
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=30)

        # Check cache
        cache_key = f"{market_id}_{start_time.isoformat()}_{end_time.isoformat()}_{resolution}"
        cached = self._get_cached("fetch_market_prices", cache_key)
        if cached is not None:
            return cached

        # Map resolution to CLOB fidelity parameter
        fidelity_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        fidelity = fidelity_map.get(resolution, 60)

        # Fetch price history from CLOB API
        url = f"{self.clob_url}/prices-history"
        params = {
            "market": market_id,
            "startTs": int(start_time.timestamp()),
            "endTs": int(end_time.timestamp()),
            "fidelity": fidelity
        }

        try:
            result = await self._request("GET", url, params=params)
        except Exception:
            # Fallback: return empty if API fails
            return []

        # Normalize response
        prices = []
        history = result.get("history", [])

        for point in history:
            ts = point.get("t", point.get("timestamp"))
            if isinstance(ts, (int, float)):
                timestamp = datetime.fromtimestamp(ts)
            else:
                timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))

            # Price data - handle both formats
            yes_price = float(point.get("p", point.get("price", 0.5)))

            prices.append({
                "timestamp": timestamp,
                "yes_price": yes_price,
                "no_price": 1.0 - yes_price,
                "volume": float(point.get("v", point.get("volume", 0)))
            })

        self._set_cached("fetch_market_prices", prices, cache_key)
        return prices

    async def fetch_current_price(self, market_id: str) -> dict:
        """
        Fetch current price and order book for a market.

        Args:
            market_id: Market token ID

        Returns:
            {
                "yes_price": float,
                "no_price": float,
                "yes_bid": float,
                "yes_ask": float,
                "no_bid": float,
                "no_ask": float,
                "spread": float,
                "timestamp": datetime
            }
        """
        # Check cache (short TTL for current prices)
        cached = self._get_cached("fetch_current_price", market_id)
        if cached is not None:
            return cached

        # Fetch order book from CLOB
        url = f"{self.clob_url}/book"
        params = {"token_id": market_id}

        try:
            result = await self._request("GET", url, params=params)
        except Exception:
            # Return default prices on error
            return {
                "yes_price": 0.5,
                "no_price": 0.5,
                "yes_bid": 0.49,
                "yes_ask": 0.51,
                "no_bid": 0.49,
                "no_ask": 0.51,
                "spread": 0.02,
                "timestamp": datetime.now()
            }

        # Parse order book
        bids = result.get("bids", [])
        asks = result.get("asks", [])

        yes_bid = float(bids[0]["price"]) if bids else 0.49
        yes_ask = float(asks[0]["price"]) if asks else 0.51
        yes_mid = (yes_bid + yes_ask) / 2

        price_data = {
            "yes_price": yes_mid,
            "no_price": 1.0 - yes_mid,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": 1.0 - yes_ask,
            "no_ask": 1.0 - yes_bid,
            "spread": yes_ask - yes_bid,
            "timestamp": datetime.now()
        }

        self._set_cached("fetch_current_price", price_data, market_id)
        return price_data

    async def fetch_order_book(
        self,
        market_id: str,
        depth: int = 10
    ) -> dict:
        """
        Fetch full order book for a market.

        Args:
            market_id: Market token ID
            depth: Number of levels on each side

        Returns:
            {
                "bids": [{"price": float, "size": float}],
                "asks": [{"price": float, "size": float}],
                "timestamp": datetime
            }
        """
        url = f"{self.clob_url}/book"
        params = {"token_id": market_id}

        result = await self._request("GET", url, params=params)

        return {
            "bids": [
                {"price": float(b["price"]), "size": float(b["size"])}
                for b in result.get("bids", [])[:depth]
            ],
            "asks": [
                {"price": float(a["price"]), "size": float(a["size"])}
                for a in result.get("asks", [])[:depth]
            ],
            "timestamp": datetime.now()
        }

    async def fetch_trades(
        self,
        market_id: str,
        limit: int = 100
    ) -> list[dict]:
        """
        Fetch recent trades for a market.

        Args:
            market_id: Market token ID
            limit: Maximum trades to return

        Returns:
            List of trades:
            {
                "id": str,
                "price": float,
                "size": float,
                "side": str,
                "timestamp": datetime
            }
        """
        url = f"{self.clob_url}/trades"
        params = {
            "token_id": market_id,
            "limit": limit
        }

        result = await self._request("GET", url, params=params)

        trades = []
        for t in result if isinstance(result, list) else result.get("data", []):
            ts = t.get("timestamp", t.get("created_at"))
            if isinstance(ts, (int, float)):
                timestamp = datetime.fromtimestamp(ts)
            else:
                timestamp = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))

            trades.append({
                "id": t.get("id", ""),
                "price": float(t.get("price", 0)),
                "size": float(t.get("size", t.get("amount", 0))),
                "side": t.get("side", "unknown"),
                "timestamp": timestamp
            })

        return trades


class MockPolymarketConnector(DataConnector):
    """
    Mock Polymarket connector for testing without API calls.

    Returns realistic fake data that matches the real API structure.
    """

    def __init__(self, config: Optional[ConnectorConfig] = None):
        if config is None:
            config = ConnectorConfig(name="mock_polymarket")
        super().__init__(config)
        self._mock_markets = self._generate_mock_markets()

    def _generate_mock_markets(self) -> list[dict]:
        """Generate mock market data."""
        import random
        random.seed(42)

        markets = [
            {
                "id": "mock_001",
                "question": "Will the incumbent win the presidential election?",
                "category": "politics",
                "status": "active",
                "tokens": [
                    {"token_id": "mock_001_yes", "outcome": "Yes", "price": 0.55},
                    {"token_id": "mock_001_no", "outcome": "No", "price": 0.45}
                ]
            },
            {
                "id": "mock_002",
                "question": "Will Bitcoin exceed $100,000 by end of year?",
                "category": "crypto",
                "status": "active",
                "tokens": [
                    {"token_id": "mock_002_yes", "outcome": "Yes", "price": 0.35},
                    {"token_id": "mock_002_no", "outcome": "No", "price": 0.65}
                ]
            },
            {
                "id": "mock_003",
                "question": "Will Team A win the championship?",
                "category": "sports",
                "status": "active",
                "tokens": [
                    {"token_id": "mock_003_yes", "outcome": "Yes", "price": 0.25},
                    {"token_id": "mock_003_no", "outcome": "No", "price": 0.75}
                ]
            }
        ]
        return markets

    async def connect(self) -> bool:
        self._is_connected = True
        return True

    async def disconnect(self) -> None:
        self._is_connected = False

    async def health_check(self) -> bool:
        return self._is_connected

    async def fetch_markets(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        **kwargs
    ) -> list[dict]:
        markets = self._mock_markets

        if category:
            markets = [m for m in markets if m["category"] == category]
        if status:
            markets = [m for m in markets if m["status"] == status]

        return markets[:limit]

    async def fetch_market_prices(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        resolution: str = "1h"
    ) -> list[dict]:
        import random

        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=30)

        # Generate mock price history
        prices = []
        current = start_time
        price = 0.5

        while current <= end_time:
            price += random.gauss(0, 0.02)
            price = max(0.05, min(0.95, price))

            prices.append({
                "timestamp": current,
                "yes_price": price,
                "no_price": 1 - price,
                "volume": random.uniform(100, 10000)
            })

            if resolution == "1h":
                current += timedelta(hours=1)
            elif resolution == "1d":
                current += timedelta(days=1)
            else:
                current += timedelta(minutes=int(resolution.replace("m", "")))

        return prices

    async def fetch_current_price(self, market_id: str) -> dict:
        import random
        price = random.uniform(0.3, 0.7)
        spread = random.uniform(0.01, 0.03)

        return {
            "yes_price": price,
            "no_price": 1 - price,
            "yes_bid": price - spread / 2,
            "yes_ask": price + spread / 2,
            "no_bid": (1 - price) - spread / 2,
            "no_ask": (1 - price) + spread / 2,
            "spread": spread,
            "timestamp": datetime.now()
        }
