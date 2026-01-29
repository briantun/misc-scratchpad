"""
Strategy 2: Market Inefficiency

This strategy exploits structural inefficiencies in prediction markets:
- Arbitrage between related markets
- Long-shot bias (overpriced unlikely outcomes)
- Favorite bias (underpriced likely outcomes)
- Yes/No spread mispricing
- Recency bias (overreaction to recent news)
- Cross-platform arbitrage opportunities

The core thesis is that markets have predictable biases and
inconsistencies that can be systematically exploited.
"""
from datetime import datetime, timedelta
from typing import Optional
import statistics

from ..models import (
    MarketSnapshot, PortfolioState, Side, StrategySignal
)


class MarketInefficiencyStrategy:
    """
    Trades based on market structure inefficiencies and biases.
    """

    name = "market_inefficiency"

    def __init__(
        self,
        longshot_threshold: float = 0.15,
        favorite_threshold: float = 0.85,
        spread_arbitrage_threshold: float = 0.02,
        mean_reversion_zscore: float = 2.0,
        related_market_tolerance: float = 0.10
    ):
        """
        Initialize the market inefficiency strategy.

        Args:
            longshot_threshold: Price below which to consider selling (long-shot bias)
            favorite_threshold: Price above which to consider buying (favorite bias)
            spread_arbitrage_threshold: Min Yes+No deviation from 1.0 to trade
            mean_reversion_zscore: Z-score threshold for mean reversion trades
            related_market_tolerance: Max acceptable inconsistency in related markets
        """
        self.longshot_threshold = longshot_threshold
        self.favorite_threshold = favorite_threshold
        self.spread_arbitrage_threshold = spread_arbitrage_threshold
        self.mean_reversion_zscore = mean_reversion_zscore
        self.related_market_tolerance = related_market_tolerance

    def _check_yes_no_arbitrage(
        self,
        snapshot: MarketSnapshot
    ) -> Optional[StrategySignal]:
        """
        Check if Yes + No prices don't sum to 1.0, indicating arbitrage.

        If YES_ask + NO_ask < 1.0, we can buy both sides and guarantee profit.
        If YES_bid + NO_bid > 1.0, we can sell both sides and guarantee profit.
        """
        price = snapshot.price
        current_time = snapshot.price.timestamp

        # Check for buying arbitrage (both sides underpriced)
        buy_total = price.ask_yes + price.ask_no
        if buy_total < 1.0 - self.spread_arbitrage_threshold:
            # Guaranteed profit by buying both sides
            edge = 1.0 - buy_total
            return StrategySignal(
                strategy_name=f"{self.name}:spread_arb_buy",
                market_id=snapshot.market.id,
                side=Side.YES,  # Buy both sides
                strength=edge * 5,  # High confidence for arbitrage
                suggested_size=0.10,  # Size up for true arbitrage
                timestamp=current_time,
                metadata={
                    "type": "spread_arbitrage",
                    "direction": "buy_both",
                    "yes_ask": price.ask_yes,
                    "no_ask": price.ask_no,
                    "edge": edge
                }
            )

        # Check for selling arbitrage (both sides overpriced)
        sell_total = price.bid_yes + price.bid_no
        if sell_total > 1.0 + self.spread_arbitrage_threshold:
            edge = sell_total - 1.0
            return StrategySignal(
                strategy_name=f"{self.name}:spread_arb_sell",
                market_id=snapshot.market.id,
                side=Side.NO,  # Sell both sides
                strength=edge * 5,
                suggested_size=0.10,
                timestamp=current_time,
                metadata={
                    "type": "spread_arbitrage",
                    "direction": "sell_both",
                    "yes_bid": price.bid_yes,
                    "no_bid": price.bid_no,
                    "edge": edge
                }
            )

        return None

    def _check_longshot_bias(
        self,
        snapshot: MarketSnapshot,
        history: list[MarketSnapshot]
    ) -> Optional[StrategySignal]:
        """
        Exploit long-shot bias: low probability events are systematically overpriced.

        Research shows bettors overweight unlikely outcomes, so selling
        cheap contracts can be profitable over many trades.
        """
        price = snapshot.price
        current_time = snapshot.price.timestamp

        # Look for overpriced long-shots
        yes_price = price.mid_yes

        if yes_price <= self.longshot_threshold:
            # Calculate historical resolution rate for similar prices
            # If historical YES resolution at this price range is lower than price implies,
            # selling NO (buying YES at low prices) may be -EV, so we sell YES

            # Simple heuristic: long-shots at 10% resolve <10% of time empirically
            # Academic research suggests ~30% overpricing for very low probability events
            implied_prob = yes_price
            estimated_true_prob = implied_prob * 0.7  # Assume 30% overpricing

            edge = implied_prob - estimated_true_prob

            if edge > 0.02:  # Minimum 2% edge
                return StrategySignal(
                    strategy_name=f"{self.name}:longshot_bias",
                    market_id=snapshot.market.id,
                    side=Side.NO,  # Sell YES / Buy NO
                    strength=edge * 2,
                    suggested_size=0.05,  # Conservative sizing for bias plays
                    timestamp=current_time,
                    metadata={
                        "type": "longshot_bias",
                        "yes_price": yes_price,
                        "implied_prob": implied_prob,
                        "estimated_true_prob": estimated_true_prob,
                        "edge": edge
                    }
                )

        return None

    def _check_favorite_bias(
        self,
        snapshot: MarketSnapshot,
        history: list[MarketSnapshot]
    ) -> Optional[StrategySignal]:
        """
        Exploit favorite bias: high probability events may be slightly underpriced.

        This is the inverse of long-shot bias - favorites offer slightly better
        odds than their true probability would suggest.
        """
        price = snapshot.price
        current_time = snapshot.price.timestamp

        yes_price = price.mid_yes

        if yes_price >= self.favorite_threshold:
            # High probability events may be underpriced
            # Research suggests ~5-10% underpricing for heavy favorites
            implied_prob = yes_price
            estimated_true_prob = min(0.98, implied_prob * 1.05)  # Assume 5% underpricing

            edge = estimated_true_prob - implied_prob

            if edge > 0.02:
                return StrategySignal(
                    strategy_name=f"{self.name}:favorite_bias",
                    market_id=snapshot.market.id,
                    side=Side.YES,  # Buy YES at underpriced level
                    strength=edge * 2,
                    suggested_size=0.05,
                    timestamp=current_time,
                    metadata={
                        "type": "favorite_bias",
                        "yes_price": yes_price,
                        "implied_prob": implied_prob,
                        "estimated_true_prob": estimated_true_prob,
                        "edge": edge
                    }
                )

        return None

    def _check_mean_reversion(
        self,
        snapshot: MarketSnapshot,
        history: list[MarketSnapshot]
    ) -> Optional[StrategySignal]:
        """
        Trade mean reversion after extreme price moves.

        Markets often overreact to news, then revert. If price has moved
        significantly from recent average, bet on reversion.
        """
        if len(history) < 10:
            return None

        current_time = snapshot.price.timestamp
        current_price = snapshot.price.mid_yes

        # Calculate historical statistics
        historical_prices = [h.price.mid_yes for h in history[-30:]]  # Last 30 observations

        if len(historical_prices) < 10:
            return None

        mean_price = statistics.mean(historical_prices)
        std_price = statistics.stdev(historical_prices) if len(historical_prices) > 1 else 0.1

        if std_price < 0.01:  # Not enough volatility
            return None

        # Calculate z-score
        z_score = (current_price - mean_price) / std_price

        if abs(z_score) >= self.mean_reversion_zscore:
            # Price is extended, bet on reversion
            if z_score > 0:
                # Price spiked up, expect reversion down
                side = Side.NO
                edge = (current_price - mean_price) * 0.5  # Expect 50% reversion
            else:
                # Price dropped, expect reversion up
                side = Side.YES
                edge = (mean_price - current_price) * 0.5

            # Strength proportional to z-score
            strength = min(abs(z_score) / 4, 1.0) * edge

            return StrategySignal(
                strategy_name=f"{self.name}:mean_reversion",
                market_id=snapshot.market.id,
                side=side,
                strength=strength,
                suggested_size=0.04,  # Smaller size for mean reversion
                timestamp=current_time,
                metadata={
                    "type": "mean_reversion",
                    "current_price": current_price,
                    "mean_price": mean_price,
                    "std_price": std_price,
                    "z_score": z_score,
                    "edge": edge
                }
            )

        return None

    def _check_related_market_arbitrage(
        self,
        snapshots: list[MarketSnapshot]
    ) -> list[StrategySignal]:
        """
        Find inconsistencies between related markets.

        Examples:
        - "Team A wins championship" should be <= "Team A makes playoffs"
        - "Candidate wins state X" + "Candidate wins state Y" relates to "Candidate wins election"
        - Time-based: "Event by March" should be <= "Event by June"
        """
        signals = []

        # Build market lookup
        market_map = {s.market.id: s for s in snapshots}

        for snapshot in snapshots:
            related_ids = snapshot.market.related_market_ids

            for related_id in related_ids:
                if related_id not in market_map:
                    continue

                related = market_map[related_id]

                # Check for logical inconsistencies
                # If market A implies market B, then P(A) <= P(B)
                # We detect this through tags or naming conventions

                main_price = snapshot.price.mid_yes
                related_price = related.price.mid_yes

                # Check if main market is a subset of related market
                # (e.g., "wins championship" is subset of "makes playoffs")
                main_tags = set(snapshot.market.tags)
                related_tags = set(related.market.tags)

                if "subset_of:" + related_id in main_tags:
                    # Main should be <= related
                    if main_price > related_price + self.related_market_tolerance:
                        # Arbitrage: sell main, buy related
                        edge = main_price - related_price
                        signals.append(StrategySignal(
                            strategy_name=f"{self.name}:related_arb",
                            market_id=snapshot.market.id,
                            side=Side.NO,  # Sell main (overpriced)
                            strength=edge * 3,
                            suggested_size=0.06,
                            timestamp=snapshot.price.timestamp,
                            metadata={
                                "type": "related_market_arbitrage",
                                "main_price": main_price,
                                "related_price": related_price,
                                "related_id": related_id,
                                "edge": edge
                            }
                        ))

                elif "superset_of:" + related_id in main_tags:
                    # Main should be >= related
                    if main_price < related_price - self.related_market_tolerance:
                        edge = related_price - main_price
                        signals.append(StrategySignal(
                            strategy_name=f"{self.name}:related_arb",
                            market_id=snapshot.market.id,
                            side=Side.YES,  # Buy main (underpriced)
                            strength=edge * 3,
                            suggested_size=0.06,
                            timestamp=snapshot.price.timestamp,
                            metadata={
                                "type": "related_market_arbitrage",
                                "main_price": main_price,
                                "related_price": related_price,
                                "related_id": related_id,
                                "edge": edge
                            }
                        ))

        return signals

    def generate_signals(
        self,
        snapshots: list[MarketSnapshot],
        portfolio: PortfolioState,
        history: list[MarketSnapshot]
    ) -> list[StrategySignal]:
        """
        Generate trading signals based on market inefficiencies.
        """
        signals = []

        # Build history lookup by market
        history_by_market: dict[str, list[MarketSnapshot]] = {}
        for h in history:
            if h.market.id not in history_by_market:
                history_by_market[h.market.id] = []
            history_by_market[h.market.id].append(h)

        for snapshot in snapshots:
            market_history = history_by_market.get(snapshot.market.id, [])

            # Check each type of inefficiency
            arb_signal = self._check_yes_no_arbitrage(snapshot)
            if arb_signal:
                signals.append(arb_signal)

            longshot_signal = self._check_longshot_bias(snapshot, market_history)
            if longshot_signal:
                signals.append(longshot_signal)

            favorite_signal = self._check_favorite_bias(snapshot, market_history)
            if favorite_signal:
                signals.append(favorite_signal)

            reversion_signal = self._check_mean_reversion(snapshot, market_history)
            if reversion_signal:
                signals.append(reversion_signal)

        # Check related market arbitrage (needs all snapshots)
        related_signals = self._check_related_market_arbitrage(snapshots)
        signals.extend(related_signals)

        return signals


class CrossPlatformArbitrage:
    """
    Identifies arbitrage opportunities between Polymarket and other platforms.

    This is a reference implementation - actual cross-platform trading
    requires accounts and execution on multiple platforms.
    """

    def __init__(
        self,
        min_edge_after_fees: float = 0.02,
        platforms: Optional[list[str]] = None
    ):
        self.min_edge_after_fees = min_edge_after_fees
        self.platforms = platforms or ["polymarket", "kalshi", "predictit", "metaculus"]

        # Approximate fee structures
        self.platform_fees = {
            "polymarket": 0.02,
            "kalshi": 0.01,
            "predictit": 0.10,  # 10% on profits
            "metaculus": 0.0   # No fees but limited trading
        }

    def find_arbitrage(
        self,
        polymarket_snapshot: MarketSnapshot,
        external_prices: dict[str, dict]
    ) -> Optional[dict]:
        """
        Find arbitrage between Polymarket and external platforms.

        Args:
            polymarket_snapshot: Current Polymarket state
            external_prices: Dict of platform -> {"yes_price": float, "no_price": float}

        Returns:
            Arbitrage opportunity if found, None otherwise
        """
        poly_yes = polymarket_snapshot.price.mid_yes
        poly_no = polymarket_snapshot.price.mid_no

        for platform, prices in external_prices.items():
            if platform not in self.platform_fees:
                continue

            ext_yes = prices.get("yes_price", 0.5)
            ext_no = prices.get("no_price", 0.5)

            # Calculate potential arbitrage
            # Buy YES on cheaper platform, sell YES (buy NO) on expensive platform
            poly_fee = self.platform_fees["polymarket"]
            ext_fee = self.platform_fees[platform]

            # Case 1: Poly YES cheaper than external YES
            if poly_yes < ext_yes:
                gross_edge = ext_yes - poly_yes
                net_edge = gross_edge - poly_fee - ext_fee
                if net_edge > self.min_edge_after_fees:
                    return {
                        "type": "cross_platform_arbitrage",
                        "buy_platform": "polymarket",
                        "sell_platform": platform,
                        "side": "YES",
                        "buy_price": poly_yes,
                        "sell_price": ext_yes,
                        "gross_edge": gross_edge,
                        "net_edge": net_edge
                    }

            # Case 2: External YES cheaper than Poly YES
            if ext_yes < poly_yes:
                gross_edge = poly_yes - ext_yes
                net_edge = gross_edge - poly_fee - ext_fee
                if net_edge > self.min_edge_after_fees:
                    return {
                        "type": "cross_platform_arbitrage",
                        "buy_platform": platform,
                        "sell_platform": "polymarket",
                        "side": "YES",
                        "buy_price": ext_yes,
                        "sell_price": poly_yes,
                        "gross_edge": gross_edge,
                        "net_edge": net_edge
                    }

        return None
