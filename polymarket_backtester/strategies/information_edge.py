"""
Strategy 1: Information Edge

This strategy exploits informational advantages by comparing market prices
to external signals such as:
- Poll data and aggregations
- Sentiment analysis scores
- News volume and momentum
- Expert predictions
- Statistical models

The core thesis is that markets may lag behind available information,
creating opportunities when our signal suggests a different probability
than the market price implies.
"""
from datetime import datetime
from typing import Optional

from ..models import (
    MarketSnapshot, PortfolioState, Side, StrategySignal
)


class InformationEdgeStrategy:
    """
    Trades based on discrepancies between market prices and external information signals.
    """

    name = "information_edge"

    def __init__(
        self,
        min_edge_threshold: float = 0.05,
        max_signal_age_hours: float = 24.0,
        signal_weights: Optional[dict[str, float]] = None,
        confidence_decay_rate: float = 0.1
    ):
        """
        Initialize the information edge strategy.

        Args:
            min_edge_threshold: Minimum difference between signal and price to trade (0.05 = 5%)
            max_signal_age_hours: Ignore signals older than this
            signal_weights: Custom weights for different signal types
            confidence_decay_rate: How fast signal confidence decays over time
        """
        self.min_edge_threshold = min_edge_threshold
        self.max_signal_age_hours = max_signal_age_hours
        self.confidence_decay_rate = confidence_decay_rate

        # Default signal weights (higher = more trusted)
        self.signal_weights = signal_weights or {
            "poll_average": 1.0,
            "poll_538": 1.2,  # FiveThirtyEight-style aggregation
            "poll_rcp": 1.0,  # RealClearPolitics-style
            "expert_consensus": 0.8,
            "sentiment_score": 0.5,
            "news_momentum": 0.4,
            "statistical_model": 1.1,
            "base_rate": 0.6,
            "insider_signal": 1.5,  # High weight but rare
        }

    def _extract_fair_value(
        self,
        snapshot: MarketSnapshot,
        current_time: datetime
    ) -> Optional[tuple[float, float]]:
        """
        Calculate fair value estimate from external signals.

        Returns:
            Tuple of (fair_value, confidence) or None if no valid signals
        """
        signals = snapshot.external_signals
        if not signals:
            return None

        weighted_sum = 0.0
        total_weight = 0.0

        for signal_name, signal_value in signals.items():
            if signal_name not in self.signal_weights:
                continue

            # Check for signal metadata (timestamp, confidence)
            if isinstance(signal_value, dict):
                value = signal_value.get("value", signal_value.get("probability"))
                signal_time = signal_value.get("timestamp")
                signal_confidence = signal_value.get("confidence", 1.0)

                # Apply time decay if timestamp available
                if signal_time and isinstance(signal_time, datetime):
                    hours_old = (current_time - signal_time).total_seconds() / 3600
                    if hours_old > self.max_signal_age_hours:
                        continue
                    time_decay = max(0, 1 - self.confidence_decay_rate * hours_old)
                    signal_confidence *= time_decay
            else:
                value = signal_value
                signal_confidence = 1.0

            if value is None or not (0 <= value <= 1):
                continue

            weight = self.signal_weights[signal_name] * signal_confidence
            weighted_sum += value * weight
            total_weight += weight

        if total_weight == 0:
            return None

        fair_value = weighted_sum / total_weight
        confidence = min(total_weight / 2.0, 1.0)  # Normalize confidence

        return fair_value, confidence

    def _calculate_edge(
        self,
        fair_value: float,
        market_price: float
    ) -> tuple[Side, float]:
        """
        Calculate trading edge and direction.

        Returns:
            Tuple of (side_to_buy, edge_magnitude)
        """
        # If fair value > market price, YES is underpriced
        edge = fair_value - market_price

        if edge > 0:
            return Side.YES, edge
        else:
            return Side.NO, -edge

    def _calculate_position_size(
        self,
        edge: float,
        confidence: float,
        price: float
    ) -> float:
        """
        Calculate suggested position size using modified Kelly criterion.

        The Kelly formula is: f* = (bp - q) / b
        where b = odds, p = win probability, q = 1 - p

        We use a fractional Kelly (1/4) for safety.
        """
        # Convert edge and price to implied win probability
        # If we think fair value is p and market is at price m,
        # our edge suggests win prob of p, and the odds are (1-m)/m

        if price <= 0 or price >= 1:
            return 0.0

        # Implied win probability from our signal
        win_prob = min(0.95, max(0.05, price + edge))  # Clamp to reasonable range

        # Odds received (profit per dollar if we win)
        odds = (1 - price) / price

        # Kelly fraction
        kelly = (odds * win_prob - (1 - win_prob)) / odds

        # Apply fractional Kelly and confidence adjustment
        fractional_kelly = 0.25
        position_size = kelly * fractional_kelly * confidence

        # Clamp to reasonable range
        return max(0, min(0.15, position_size))

    def generate_signals(
        self,
        snapshots: list[MarketSnapshot],
        portfolio: PortfolioState,
        history: list[MarketSnapshot]
    ) -> list[StrategySignal]:
        """
        Generate trading signals based on information edge.
        """
        signals = []
        current_time = portfolio.timestamp

        for snapshot in snapshots:
            # Extract fair value from external signals
            result = self._extract_fair_value(snapshot, current_time)
            if result is None:
                continue

            fair_value, confidence = result

            # Get current market price
            market_price = snapshot.price.mid_yes

            # Calculate edge
            side, edge = self._calculate_edge(fair_value, market_price)

            # Check minimum threshold
            if edge < self.min_edge_threshold:
                continue

            # Calculate position size
            if side == Side.YES:
                position_size = self._calculate_position_size(
                    edge, confidence, market_price
                )
            else:
                position_size = self._calculate_position_size(
                    edge, confidence, 1 - market_price
                )

            if position_size <= 0:
                continue

            # Create signal
            signal = StrategySignal(
                strategy_name=self.name,
                market_id=snapshot.market.id,
                side=side,
                strength=edge * confidence,
                suggested_size=position_size,
                timestamp=current_time,
                metadata={
                    "fair_value": fair_value,
                    "market_price": market_price,
                    "edge": edge,
                    "confidence": confidence,
                    "signal_sources": list(snapshot.external_signals.keys())
                }
            )
            signals.append(signal)

        return signals


class PollBasedSubStrategy:
    """
    Specialized sub-strategy for political markets using polling data.

    This demonstrates how to build specialized signal processors
    that feed into the main information edge strategy.
    """

    def __init__(
        self,
        poll_weight_by_grade: Optional[dict[str, float]] = None,
        recency_halflife_days: float = 7.0
    ):
        self.poll_weight_by_grade = poll_weight_by_grade or {
            "A+": 1.5, "A": 1.4, "A-": 1.3,
            "B+": 1.2, "B": 1.1, "B-": 1.0,
            "C+": 0.9, "C": 0.8, "C-": 0.7,
            "D": 0.5, "F": 0.3
        }
        self.recency_halflife_days = recency_halflife_days

    def aggregate_polls(
        self,
        polls: list[dict],
        current_time: datetime
    ) -> dict:
        """
        Aggregate multiple polls into a single probability estimate.

        Args:
            polls: List of poll data dicts with keys:
                - value: probability (0-1)
                - timestamp: when poll was conducted
                - pollster_grade: letter grade of pollster
                - sample_size: number of respondents

        Returns:
            Dict with aggregated probability and confidence
        """
        if not polls:
            return None

        weighted_sum = 0.0
        total_weight = 0.0

        for poll in polls:
            value = poll.get("value")
            if value is None or not (0 <= value <= 1):
                continue

            # Base weight from pollster grade
            grade = poll.get("pollster_grade", "C")
            grade_weight = self.poll_weight_by_grade.get(grade, 0.8)

            # Sample size adjustment (sqrt scaling)
            sample_size = poll.get("sample_size", 500)
            size_weight = (sample_size / 500) ** 0.5

            # Recency weighting (exponential decay)
            poll_time = poll.get("timestamp")
            if poll_time and isinstance(poll_time, datetime):
                days_old = (current_time - poll_time).total_seconds() / 86400
                recency_weight = 0.5 ** (days_old / self.recency_halflife_days)
            else:
                recency_weight = 0.5  # Assume somewhat stale if no timestamp

            # Combined weight
            weight = grade_weight * size_weight * recency_weight
            weighted_sum += value * weight
            total_weight += weight

        if total_weight == 0:
            return None

        probability = weighted_sum / total_weight

        # Confidence based on number of polls and total weight
        confidence = min(1.0, (len(polls) / 5) * (total_weight / len(polls)))

        return {
            "value": probability,
            "confidence": confidence,
            "timestamp": current_time,
            "polls_used": len(polls)
        }


class SentimentAnalysisSubStrategy:
    """
    Sub-strategy that processes sentiment signals from social media,
    news articles, and other text sources.
    """

    def __init__(
        self,
        sentiment_smoothing_window: int = 24,  # hours
        momentum_threshold: float = 0.1
    ):
        self.sentiment_smoothing_window = sentiment_smoothing_window
        self.momentum_threshold = momentum_threshold

    def process_sentiment_history(
        self,
        sentiment_history: list[dict],
        current_time: datetime
    ) -> dict:
        """
        Process sentiment history to generate a trading signal.

        Args:
            sentiment_history: List of sentiment readings with:
                - score: sentiment score (-1 to 1)
                - timestamp: when measured
                - volume: number of mentions/posts

        Returns:
            Dict with sentiment-based probability adjustment
        """
        if not sentiment_history:
            return None

        # Filter to recent window
        window_start = current_time.timestamp() - self.sentiment_smoothing_window * 3600
        recent = [
            s for s in sentiment_history
            if s.get("timestamp", datetime.min).timestamp() >= window_start
        ]

        if not recent:
            return None

        # Volume-weighted average sentiment
        weighted_sum = 0.0
        total_volume = 0.0

        for reading in recent:
            score = reading.get("score", 0)
            volume = reading.get("volume", 1)
            weighted_sum += score * volume
            total_volume += volume

        if total_volume == 0:
            return None

        avg_sentiment = weighted_sum / total_volume

        # Calculate momentum (rate of change)
        if len(recent) >= 2:
            recent_sorted = sorted(recent, key=lambda x: x.get("timestamp", datetime.min))
            early_scores = [r["score"] for r in recent_sorted[:len(recent_sorted)//2]]
            late_scores = [r["score"] for r in recent_sorted[len(recent_sorted)//2:]]

            early_avg = sum(early_scores) / len(early_scores) if early_scores else 0
            late_avg = sum(late_scores) / len(late_scores) if late_scores else 0
            momentum = late_avg - early_avg
        else:
            momentum = 0

        # Convert sentiment to probability adjustment
        # Sentiment of 0 = no adjustment, sentiment of +/-1 = +/-20% adjustment
        probability_adjustment = avg_sentiment * 0.2

        # Boost if strong momentum
        if abs(momentum) > self.momentum_threshold:
            probability_adjustment *= 1.5

        # Confidence based on volume
        confidence = min(1.0, total_volume / 1000)

        return {
            "adjustment": probability_adjustment,
            "momentum": momentum,
            "confidence": confidence,
            "timestamp": current_time,
            "volume": total_volume
        }
