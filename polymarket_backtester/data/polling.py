"""
Polling data connectors for political prediction markets.

Provides access to:
- FiveThirtyEight polling averages and forecasts
- RealClearPolitics polling aggregations
- Generic polling data interfaces

These signals are crucial for the Information Edge strategy.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Any
import json
import re

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from .base import SignalConnector, ConnectorConfig, CacheConfig


@dataclass
class Poll:
    """Represents a single poll result."""
    pollster: str
    date: datetime
    sample_size: int
    margin_of_error: Optional[float]
    results: dict[str, float]  # candidate/option -> percentage
    grade: Optional[str] = None  # Pollster grade (A+, B-, etc.)
    population: str = "lv"  # lv=likely voters, rv=registered voters, a=adults
    url: Optional[str] = None

    @property
    def leader(self) -> str:
        """Return the leading option."""
        return max(self.results, key=self.results.get)

    @property
    def margin(self) -> float:
        """Return margin between top two options."""
        sorted_results = sorted(self.results.values(), reverse=True)
        if len(sorted_results) >= 2:
            return sorted_results[0] - sorted_results[1]
        return 0.0


@dataclass
class PollAggregate:
    """Aggregated polling average."""
    date: datetime
    averages: dict[str, float]  # candidate/option -> average percentage
    num_polls: int
    confidence: float  # 0-1 confidence in the aggregate
    source: str

    def to_probability(self, option: str) -> float:
        """Convert polling average to win probability (simplified)."""
        if option not in self.averages:
            return 0.5

        pct = self.averages[option]
        total = sum(self.averages.values())
        if total == 0:
            return 0.5

        # Simple conversion: polling percentage -> probability
        # This is a naive approach; real models are more sophisticated
        normalized = pct / total
        return normalized


class PollingSource(ABC):
    """Abstract base for polling data sources."""

    @abstractmethod
    async def fetch_polls(
        self,
        race_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[Poll]:
        """Fetch individual polls for a race."""
        pass

    @abstractmethod
    async def fetch_average(
        self,
        race_id: str,
        date: Optional[datetime] = None
    ) -> Optional[PollAggregate]:
        """Fetch polling average for a race."""
        pass

    @abstractmethod
    async def list_races(self) -> list[dict]:
        """List available races to poll."""
        pass


class FiveThirtyEightSource(PollingSource):
    """
    Fetches polling data from FiveThirtyEight.

    Note: 538's data availability has changed over time.
    This connector attempts to use their public data feeds.
    """

    BASE_URL = "https://projects.fivethirtyeight.com"
    GITHUB_DATA_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master"

    def __init__(self):
        self._session: Optional[Any] = None

    async def _ensure_session(self):
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_polls(
        self,
        race_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[Poll]:
        """
        Fetch polls from 538's data repository.

        race_id format: "{year}_{race_type}_{state}"
        e.g., "2024_president_national", "2024_senate_arizona"
        """
        await self._ensure_session()

        # Parse race_id
        parts = race_id.split("_")
        if len(parts) < 2:
            return []

        year = parts[0]
        race_type = parts[1]

        # Construct URL for 538 polling data
        # This URL structure may need updates based on 538's current format
        if race_type == "president":
            url = f"{self.GITHUB_DATA_URL}/polls/president_polls.csv"
        elif race_type == "senate":
            url = f"{self.GITHUB_DATA_URL}/polls/senate_polls.csv"
        elif race_type == "governor":
            url = f"{self.GITHUB_DATA_URL}/polls/governor_polls.csv"
        else:
            return []

        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                text = await resp.text()
        except Exception:
            return []

        # Parse CSV
        polls = []
        lines = text.strip().split("\n")
        if not lines:
            return []

        headers = lines[0].split(",")
        for line in lines[1:]:
            try:
                values = self._parse_csv_line(line)
                if len(values) != len(headers):
                    continue

                row = dict(zip(headers, values))

                # Filter by race/state if specified
                state = parts[2] if len(parts) > 2 else "national"
                if state != "national" and row.get("state", "").lower() != state.lower():
                    continue

                # Parse date
                date_str = row.get("end_date", row.get("created_at", ""))
                try:
                    poll_date = datetime.strptime(date_str, "%m/%d/%y")
                except ValueError:
                    try:
                        poll_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        continue

                # Apply date filters
                if start_date and poll_date < start_date:
                    continue
                if end_date and poll_date > end_date:
                    continue

                # Parse results
                results = {}
                for key in row:
                    if key.startswith("pct_") or key.endswith("_pct"):
                        candidate = key.replace("pct_", "").replace("_pct", "")
                        try:
                            results[candidate] = float(row[key])
                        except ValueError:
                            pass

                # Also try candidate-specific columns
                if "candidate_name" in row and "pct" in row:
                    results[row["candidate_name"]] = float(row.get("pct", 0))

                if not results:
                    continue

                poll = Poll(
                    pollster=row.get("pollster", row.get("pollster_rating_name", "Unknown")),
                    date=poll_date,
                    sample_size=int(row.get("sample_size", 0) or 0),
                    margin_of_error=float(row.get("margin_of_error", 0) or 0) or None,
                    results=results,
                    grade=row.get("fte_grade", row.get("pollster_rating", None)),
                    population=row.get("population", "lv"),
                    url=row.get("url")
                )
                polls.append(poll)

            except Exception:
                continue

        return polls

    def _parse_csv_line(self, line: str) -> list[str]:
        """Parse a CSV line handling quoted fields."""
        result = []
        current = ""
        in_quotes = False

        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                result.append(current.strip())
                current = ""
            else:
                current += char

        result.append(current.strip())
        return result

    async def fetch_average(
        self,
        race_id: str,
        date: Optional[datetime] = None
    ) -> Optional[PollAggregate]:
        """
        Calculate polling average from recent polls.

        Uses a weighted average based on:
        - Pollster grade
        - Sample size
        - Recency
        """
        if date is None:
            date = datetime.now()

        # Fetch polls from last 30 days
        start_date = date - timedelta(days=30)
        polls = await self.fetch_polls(race_id, start_date, date)

        if not polls:
            return None

        # Grade weights
        grade_weights = {
            "A+": 1.5, "A": 1.4, "A-": 1.3,
            "A/B": 1.2, "B+": 1.1, "B": 1.0, "B-": 0.9,
            "B/C": 0.85, "C+": 0.8, "C": 0.7, "C-": 0.6,
            "C/D": 0.5, "D+": 0.4, "D": 0.3, "D-": 0.2
        }

        # Calculate weighted averages
        weighted_sums: dict[str, float] = {}
        total_weights: dict[str, float] = {}

        for poll in polls:
            # Weight factors
            grade_weight = grade_weights.get(poll.grade or "C", 0.7)
            size_weight = (poll.sample_size / 1000) ** 0.5 if poll.sample_size > 0 else 0.5
            days_old = (date - poll.date).days
            recency_weight = 0.5 ** (days_old / 14)  # Half-life of 14 days

            weight = grade_weight * size_weight * recency_weight

            for candidate, pct in poll.results.items():
                if candidate not in weighted_sums:
                    weighted_sums[candidate] = 0
                    total_weights[candidate] = 0
                weighted_sums[candidate] += pct * weight
                total_weights[candidate] += weight

        # Calculate averages
        averages = {}
        for candidate in weighted_sums:
            if total_weights[candidate] > 0:
                averages[candidate] = weighted_sums[candidate] / total_weights[candidate]

        if not averages:
            return None

        return PollAggregate(
            date=date,
            averages=averages,
            num_polls=len(polls),
            confidence=min(len(polls) / 10, 1.0),
            source="fivethirtyeight"
        )

    async def list_races(self) -> list[dict]:
        """List available races."""
        # Return common race types
        return [
            {"id": "2024_president_national", "name": "2024 Presidential - National"},
            {"id": "2024_president_pennsylvania", "name": "2024 Presidential - Pennsylvania"},
            {"id": "2024_president_michigan", "name": "2024 Presidential - Michigan"},
            {"id": "2024_president_wisconsin", "name": "2024 Presidential - Wisconsin"},
            {"id": "2024_president_arizona", "name": "2024 Presidential - Arizona"},
            {"id": "2024_president_georgia", "name": "2024 Presidential - Georgia"},
            {"id": "2024_president_nevada", "name": "2024 Presidential - Nevada"},
        ]


class RealClearPoliticsSource(PollingSource):
    """
    Fetches polling data from RealClearPolitics.

    Note: RCP doesn't have a public API, so this uses web scraping
    which may break if their site changes.
    """

    BASE_URL = "https://www.realclearpolitics.com"

    def __init__(self):
        self._session: Optional[Any] = None

    async def _ensure_session(self):
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_polls(
        self,
        race_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[Poll]:
        """
        Fetch polls from RCP.

        Note: This is a simplified implementation. Real scraping would
        need to handle RCP's specific page structure.
        """
        # RCP doesn't have an easy API, return empty for now
        # In production, would implement web scraping
        return []

    async def fetch_average(
        self,
        race_id: str,
        date: Optional[datetime] = None
    ) -> Optional[PollAggregate]:
        """
        Fetch RCP polling average.

        RCP shows simple averages of recent polls.
        """
        await self._ensure_session()

        # Map race_id to RCP URL path
        # This is a simplified mapping
        url_map = {
            "2024_president_national": "/epolls/2024/president/us/general-election-trump-vs-biden-7383.html",
            "2024_president_pennsylvania": "/epolls/2024/president/pa/pennsylvania-trump-vs-biden-7394.html",
        }

        if race_id not in url_map:
            return None

        # In production, would scrape the actual page
        # For now, return None
        return None

    async def list_races(self) -> list[dict]:
        return [
            {"id": "2024_president_national", "name": "2024 Presidential - National"},
        ]


class PollingConnector(SignalConnector):
    """
    Unified connector for polling data from multiple sources.

    Aggregates data from FiveThirtyEight, RealClearPolitics, and other sources
    to provide trading signals for political prediction markets.
    """

    def __init__(
        self,
        config: Optional[ConnectorConfig] = None,
        sources: Optional[list[PollingSource]] = None
    ):
        if config is None:
            config = ConnectorConfig(
                name="polling",
                cache=CacheConfig(ttl_seconds=1800)  # 30 min cache for polls
            )
        super().__init__(config)

        self.sources = sources or [
            FiveThirtyEightSource(),
            RealClearPoliticsSource()
        ]

    async def connect(self) -> bool:
        self._is_connected = True
        return True

    async def disconnect(self) -> None:
        for source in self.sources:
            if hasattr(source, 'close'):
                await source.close()
        self._is_connected = False

    async def health_check(self) -> bool:
        return self._is_connected

    async def fetch_markets(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """List available races across all sources."""
        races = []
        for source in self.sources:
            try:
                source_races = await source.list_races()
                races.extend(source_races)
            except Exception:
                continue

        # Deduplicate by ID
        seen = set()
        unique_races = []
        for race in races:
            if race["id"] not in seen:
                seen.add(race["id"])
                unique_races.append(race)

        return unique_races[:limit]

    async def fetch_market_prices(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        resolution: str = "1d"
    ) -> list[dict]:
        """Not applicable for polling - use fetch_signals instead."""
        return []

    async def fetch_current_price(self, market_id: str) -> dict:
        """Not applicable for polling - use fetch_current_signals instead."""
        return {}

    async def fetch_signals(
        self,
        market_id: str,
        signal_types: Optional[list[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> list[dict]:
        """
        Fetch polling signals for a race.

        Args:
            market_id: Race identifier (e.g., "2024_president_pennsylvania")
            signal_types: Types to include ("poll", "average")
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of signal dictionaries
        """
        if signal_types is None:
            signal_types = ["poll", "average"]

        signals = []

        for source in self.sources:
            try:
                # Fetch individual polls
                if "poll" in signal_types:
                    polls = await source.fetch_polls(market_id, start_time, end_time)
                    for poll in polls:
                        signals.append({
                            "type": "poll",
                            "source": source.__class__.__name__,
                            "timestamp": poll.date,
                            "pollster": poll.pollster,
                            "grade": poll.grade,
                            "sample_size": poll.sample_size,
                            "results": poll.results,
                            "leader": poll.leader,
                            "margin": poll.margin
                        })

                # Fetch averages
                if "average" in signal_types:
                    avg = await source.fetch_average(market_id, end_time)
                    if avg:
                        signals.append({
                            "type": "average",
                            "source": avg.source,
                            "timestamp": avg.date,
                            "averages": avg.averages,
                            "num_polls": avg.num_polls,
                            "confidence": avg.confidence
                        })

            except Exception:
                continue

        return signals

    async def fetch_current_signals(
        self,
        market_id: str,
        signal_types: Optional[list[str]] = None
    ) -> dict:
        """
        Fetch current polling signals.

        Returns aggregated signal suitable for trading strategies.
        """
        signals = await self.fetch_signals(
            market_id,
            signal_types=["average"],
            end_time=datetime.now()
        )

        if not signals:
            return {}

        # Find most recent average
        averages = [s for s in signals if s["type"] == "average"]
        if not averages:
            return {}

        latest = max(averages, key=lambda s: s["timestamp"])

        # Convert to probability signal
        result = {
            "poll_average": {},
            "confidence": latest["confidence"],
            "timestamp": latest["timestamp"],
            "source": latest["source"]
        }

        # Convert percentages to probabilities
        total = sum(latest["averages"].values())
        for candidate, pct in latest["averages"].items():
            result["poll_average"][candidate] = pct / total if total > 0 else 0.5

        return result


class MockPollingSource(PollingSource):
    """Mock polling source for testing."""

    def __init__(self, polls: Optional[list[Poll]] = None):
        self._polls = polls or []

    async def fetch_polls(
        self,
        race_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[Poll]:
        return self._polls

    async def fetch_average(
        self,
        race_id: str,
        date: Optional[datetime] = None
    ) -> Optional[PollAggregate]:
        if not self._polls:
            return None

        # Simple average
        all_results: dict[str, list[float]] = {}
        for poll in self._polls:
            for candidate, pct in poll.results.items():
                if candidate not in all_results:
                    all_results[candidate] = []
                all_results[candidate].append(pct)

        averages = {c: sum(v) / len(v) for c, v in all_results.items()}

        return PollAggregate(
            date=date or datetime.now(),
            averages=averages,
            num_polls=len(self._polls),
            confidence=0.8,
            source="mock"
        )

    async def list_races(self) -> list[dict]:
        return [{"id": "mock_race", "name": "Mock Race"}]
