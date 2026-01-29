"""
Synthetic data generator for backtesting.

Generates realistic market data with configurable characteristics
including trends, volatility, news events, and resolution outcomes.
"""
import random
import math
from datetime import datetime, timedelta
from typing import Optional
import uuid

from ..models import (
    Market, MarketSnapshot, MarketStatus, PricePoint, Side
)


class SyntheticDataGenerator:
    """
    Generates synthetic market data for backtesting.

    Creates realistic price paths with:
    - Random walks with drift
    - Mean reversion
    - Volatility clustering
    - News/event shocks
    - External signals with noise
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        self.seed = seed

    def generate_market(
        self,
        market_id: Optional[str] = None,
        question: str = "Will event X happen?",
        category: str = "general",
        start_date: datetime = None,
        resolution_date: datetime = None,
        true_probability: float = 0.5,
        tags: Optional[list[str]] = None,
        related_market_ids: Optional[list[str]] = None
    ) -> Market:
        """Generate a market definition."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        if resolution_date is None:
            resolution_date = datetime.now() + timedelta(days=30)

        return Market(
            id=market_id or str(uuid.uuid4())[:8],
            question=question,
            category=category,
            created_at=start_date,
            resolution_date=resolution_date,
            status=MarketStatus.OPEN,
            resolved_outcome=Side.YES if random.random() < true_probability else Side.NO,
            tags=tags or [],
            related_market_ids=related_market_ids or []
        )

    def generate_price_path(
        self,
        market: Market,
        num_days: int = 90,
        initial_price: float = 0.5,
        daily_volatility: float = 0.03,
        mean_reversion_strength: float = 0.1,
        drift_toward_resolution: bool = True,
        num_news_events: int = 5,
        news_impact_range: tuple[float, float] = (-0.15, 0.15),
        points_per_day: int = 24
    ) -> list[MarketSnapshot]:
        """
        Generate a realistic price path for a market.

        Args:
            market: Market definition
            num_days: Number of days to simulate
            initial_price: Starting YES price
            daily_volatility: Standard deviation of daily returns
            mean_reversion_strength: How strongly price reverts to fair value
            drift_toward_resolution: Whether to drift toward resolved outcome
            num_news_events: Number of random news shocks
            news_impact_range: Range of news event price impacts
            points_per_day: Data points per day (24 = hourly)

        Returns:
            List of MarketSnapshot objects
        """
        snapshots = []

        # Determine true outcome for drift calculation
        true_outcome = market.resolved_outcome
        true_price = 1.0 if true_outcome == Side.YES else 0.0

        # Pre-generate news event times and impacts
        total_points = num_days * points_per_day
        news_times = sorted(random.sample(range(total_points), min(num_news_events, total_points)))
        news_impacts = [random.uniform(*news_impact_range) for _ in news_times]

        current_price = initial_price
        current_time = market.created_at

        # Volatility state (for clustering)
        current_vol = daily_volatility / (points_per_day ** 0.5)

        for i in range(total_points):
            # Update volatility (GARCH-like clustering)
            vol_innovation = random.gauss(0, 0.1)
            current_vol = 0.9 * current_vol + 0.1 * daily_volatility / (points_per_day ** 0.5) * (1 + vol_innovation)
            current_vol = max(0.001, min(0.1, current_vol))  # Bound volatility

            # Random walk component
            random_move = random.gauss(0, current_vol)

            # Mean reversion toward fair value (estimated as 0.5 early, trends toward outcome later)
            days_elapsed = i / points_per_day
            days_remaining = num_days - days_elapsed

            if drift_toward_resolution and days_remaining > 0:
                # Fair value drifts toward true outcome as resolution approaches
                time_factor = 1 - (days_remaining / num_days)
                fair_value = 0.5 + (true_price - 0.5) * time_factor * 0.8  # 80% drift
            else:
                fair_value = 0.5

            mean_reversion = mean_reversion_strength * (fair_value - current_price) / points_per_day

            # News event shock
            news_shock = 0
            if i in news_times:
                idx = news_times.index(i)
                news_shock = news_impacts[idx]
                # News tends to move price toward truth
                if drift_toward_resolution:
                    if (true_price > 0.5 and news_shock > 0) or (true_price < 0.5 and news_shock < 0):
                        news_shock *= 1.5  # Amplify correct-direction news

            # Update price
            current_price = current_price + random_move + mean_reversion + news_shock

            # Bound price to (0.01, 0.99)
            current_price = max(0.01, min(0.99, current_price))

            # Generate spread (wider when volatile or illiquid)
            base_spread = 0.02
            vol_spread = current_vol * 2
            spread = base_spread + vol_spread + random.uniform(0, 0.01)

            # Generate volume (correlated with absolute price change and news)
            base_volume = 1000
            change_factor = 1 + abs(random_move + news_shock) * 20
            news_factor = 3 if news_shock != 0 else 1
            volume = base_volume * change_factor * news_factor * random.uniform(0.5, 1.5)

            # Create price point
            yes_mid = current_price
            no_mid = 1 - current_price

            price_point = PricePoint(
                timestamp=current_time,
                yes_price=yes_mid,
                no_price=no_mid,
                yes_volume=volume * (0.4 + random.random() * 0.2),
                no_volume=volume * (0.4 + random.random() * 0.2),
                bid_yes=yes_mid - spread / 2,
                ask_yes=yes_mid + spread / 2,
                bid_no=no_mid - spread / 2,
                ask_no=no_mid + spread / 2
            )

            # Generate external signals with noise
            external_signals = self._generate_external_signals(
                true_price, current_price, days_remaining / num_days
            )

            snapshot = MarketSnapshot(
                market=market,
                price=price_point,
                external_signals=external_signals
            )
            snapshots.append(snapshot)

            # Advance time
            current_time += timedelta(hours=24 / points_per_day)

        # Mark final snapshot as resolved if past resolution date
        if snapshots and current_time >= market.resolution_date:
            final_market = Market(
                id=market.id,
                question=market.question,
                category=market.category,
                created_at=market.created_at,
                resolution_date=market.resolution_date,
                status=MarketStatus.RESOLVED,
                resolved_outcome=market.resolved_outcome,
                tags=market.tags,
                related_market_ids=market.related_market_ids
            )
            snapshots[-1] = MarketSnapshot(
                market=final_market,
                price=snapshots[-1].price,
                external_signals=snapshots[-1].external_signals
            )

        return snapshots

    def _generate_external_signals(
        self,
        true_probability: float,
        market_price: float,
        time_remaining_pct: float
    ) -> dict:
        """
        Generate external signals that have some predictive power.

        Signals have noise but on average point toward truth,
        especially as resolution approaches.
        """
        signals = {}

        # Signal accuracy improves as resolution approaches
        accuracy = 0.5 + 0.3 * (1 - time_remaining_pct)

        # Poll average (most reliable signal)
        poll_noise = random.gauss(0, 0.08 * time_remaining_pct + 0.02)
        if random.random() < accuracy:
            # Signal points toward truth
            poll_value = true_probability + poll_noise
        else:
            # Signal is misleading
            poll_value = (1 - true_probability) + poll_noise

        poll_value = max(0.01, min(0.99, poll_value))
        signals["poll_average"] = {
            "value": poll_value,
            "confidence": 0.7 + random.uniform(0, 0.2),
            "timestamp": datetime.now() - timedelta(hours=random.randint(1, 48))
        }

        # Sentiment score (-1 to 1, converted to 0-1 probability adjustment)
        sentiment_noise = random.gauss(0, 0.3)
        if random.random() < accuracy * 0.8:  # Sentiment is less accurate
            sentiment_direction = 1 if true_probability > 0.5 else -1
        else:
            sentiment_direction = -1 if true_probability > 0.5 else 1

        raw_sentiment = sentiment_direction * (0.3 + random.uniform(0, 0.4)) + sentiment_noise
        signals["sentiment_score"] = max(-1, min(1, raw_sentiment))

        # Expert consensus
        if random.random() < 0.3:  # Only sometimes available
            expert_noise = random.gauss(0, 0.1)
            if random.random() < accuracy * 0.9:  # Experts are more accurate
                expert_value = true_probability + expert_noise
            else:
                expert_value = market_price + expert_noise  # Herding with market
            signals["expert_consensus"] = max(0.01, min(0.99, expert_value))

        # Statistical model output
        model_noise = random.gauss(0, 0.06)
        if random.random() < accuracy:
            model_value = true_probability + model_noise
        else:
            model_value = 0.5 + model_noise  # Model is uncertain
        signals["statistical_model"] = max(0.01, min(0.99, model_value))

        return signals

    def generate_correlated_markets(
        self,
        num_markets: int = 3,
        base_question: str = "Election outcome",
        correlation: float = 0.7,
        **kwargs
    ) -> list[tuple[Market, list[MarketSnapshot]]]:
        """
        Generate multiple correlated markets (e.g., related election outcomes).

        Args:
            num_markets: Number of markets to generate
            base_question: Base question text
            correlation: Correlation between market outcomes
            **kwargs: Additional arguments passed to generate_price_path

        Returns:
            List of (Market, snapshots) tuples
        """
        results = []

        # Determine correlated outcomes
        base_outcome = random.random() < 0.5
        outcomes = [base_outcome]

        for i in range(1, num_markets):
            if random.random() < correlation:
                outcomes.append(base_outcome)
            else:
                outcomes.append(not base_outcome)

        # Generate markets
        market_ids = [str(uuid.uuid4())[:8] for _ in range(num_markets)]

        for i in range(num_markets):
            # Set up relationships
            related_ids = [mid for j, mid in enumerate(market_ids) if j != i]
            tags = []
            if i > 0:
                tags.append(f"related_to:{market_ids[0]}")

            market = self.generate_market(
                market_id=market_ids[i],
                question=f"{base_question} - Scenario {i+1}",
                true_probability=0.7 if outcomes[i] else 0.3,
                related_market_ids=related_ids,
                tags=tags
            )

            snapshots = self.generate_price_path(market, **kwargs)
            results.append((market, snapshots))

        return results

    def generate_scenario(
        self,
        scenario_type: str = "election",
        num_markets: int = 5,
        num_days: int = 90,
        seed: Optional[int] = None
    ) -> dict[str, list[MarketSnapshot]]:
        """
        Generate a complete scenario with multiple markets.

        Args:
            scenario_type: Type of scenario ("election", "sports", "crypto", "random")
            num_markets: Number of markets to generate
            num_days: Days of historical data
            seed: Random seed for this scenario

        Returns:
            Dict of market_id -> list of snapshots
        """
        if seed is not None:
            random.seed(seed)

        all_data = {}

        if scenario_type == "election":
            # Presidential election with state-level markets
            states = ["Pennsylvania", "Michigan", "Wisconsin", "Arizona", "Georgia",
                     "Nevada", "North Carolina", "Florida", "Ohio", "Texas"][:num_markets]

            for state in states:
                true_prob = random.uniform(0.3, 0.7)
                market = self.generate_market(
                    question=f"Will Candidate A win {state}?",
                    category="politics",
                    true_probability=true_prob
                )
                snapshots = self.generate_price_path(
                    market,
                    num_days=num_days,
                    initial_price=0.5 + random.uniform(-0.1, 0.1),
                    daily_volatility=0.025,
                    num_news_events=3
                )
                all_data[market.id] = snapshots

        elif scenario_type == "sports":
            # Championship outcomes
            teams = ["Team A", "Team B", "Team C", "Team D", "Team E"][:num_markets]

            for team in teams:
                true_prob = random.uniform(0.1, 0.4)
                market = self.generate_market(
                    question=f"Will {team} win the championship?",
                    category="sports",
                    true_probability=true_prob
                )
                snapshots = self.generate_price_path(
                    market,
                    num_days=num_days,
                    initial_price=true_prob + random.uniform(-0.05, 0.05),
                    daily_volatility=0.02,
                    num_news_events=8  # More frequent sports news
                )
                all_data[market.id] = snapshots

        elif scenario_type == "crypto":
            # Crypto price predictions
            targets = ["$50K", "$75K", "$100K", "$150K", "$200K"][:num_markets]

            for target in targets:
                # Higher targets less likely
                target_val = int(target.replace("$", "").replace("K", "000"))
                true_prob = max(0.1, 0.9 - target_val / 300000)

                market = self.generate_market(
                    question=f"Will BTC reach {target} by end of year?",
                    category="crypto",
                    true_probability=true_prob
                )
                snapshots = self.generate_price_path(
                    market,
                    num_days=num_days,
                    initial_price=true_prob + random.uniform(-0.1, 0.1),
                    daily_volatility=0.04,  # Higher vol for crypto
                    num_news_events=10
                )
                all_data[market.id] = snapshots

        else:  # random
            for i in range(num_markets):
                true_prob = random.uniform(0.2, 0.8)
                market = self.generate_market(
                    question=f"Random event {i+1}",
                    category="general",
                    true_probability=true_prob
                )
                snapshots = self.generate_price_path(
                    market,
                    num_days=num_days,
                    initial_price=random.uniform(0.3, 0.7),
                    daily_volatility=random.uniform(0.02, 0.05),
                    num_news_events=random.randint(2, 8)
                )
                all_data[market.id] = snapshots

        return all_data
