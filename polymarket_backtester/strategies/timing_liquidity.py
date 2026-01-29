"""
Strategy 3: Timing & Liquidity

This strategy exploits timing-based opportunities:
- Market making (capturing bid-ask spread)
- Pre-event positioning (buying before scheduled catalysts)
- Post-news mean reversion (fading overreactions)
- Resolution timing (buying near-certain outcomes before resolution)
- Volume/liquidity patterns (trading when liquidity is favorable)

The core thesis is that timing of entry and exit can significantly
impact returns, independent of directional accuracy.
"""
from datetime import datetime, timedelta
from typing import Optional
import statistics

from ..models import (
    MarketSnapshot, PortfolioState, Side, StrategySignal, MarketStatus
)


class TimingLiquidityStrategy:
    """
    Trades based on timing, liquidity conditions, and market microstructure.
    """

    name = "timing_liquidity"

    def __init__(
        self,
        min_spread_for_mm: float = 0.03,
        resolution_threshold: float = 0.95,
        pre_event_days: int = 3,
        volume_spike_multiplier: float = 2.0,
        min_liquidity_depth: float = 1000.0
    ):
        """
        Initialize the timing and liquidity strategy.

        Args:
            min_spread_for_mm: Minimum bid-ask spread to engage in market making
            resolution_threshold: Price threshold to consider outcome "near-certain"
            pre_event_days: Days before resolution to start looking for pre-event plays
            volume_spike_multiplier: Volume vs average to detect spikes
            min_liquidity_depth: Minimum order book depth to trade
        """
        self.min_spread_for_mm = min_spread_for_mm
        self.resolution_threshold = resolution_threshold
        self.pre_event_days = pre_event_days
        self.volume_spike_multiplier = volume_spike_multiplier
        self.min_liquidity_depth = min_liquidity_depth

    def _check_market_making_opportunity(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState
    ) -> list[StrategySignal]:
        """
        Identify market making opportunities when spread is wide.

        Market making involves placing limit orders on both sides to
        capture the bid-ask spread. This is profitable when:
        1. Spread is wide enough to cover fees
        2. Market is relatively stable (low volatility)
        3. There's enough volume to get fills
        """
        signals = []
        price = snapshot.price
        current_time = price.timestamp

        spread_yes = price.spread_yes
        spread_no = price.spread_no

        # Check YES side spread
        if spread_yes >= self.min_spread_for_mm:
            # Place orders to capture spread
            # Buy at bid, sell at ask
            mid = price.mid_yes
            edge = spread_yes / 2  # Half spread per side

            # Signal to buy at bid
            signals.append(StrategySignal(
                strategy_name=f"{self.name}:market_making",
                market_id=snapshot.market.id,
                side=Side.YES,
                strength=edge,
                suggested_size=0.03,  # Small size for MM
                timestamp=current_time,
                metadata={
                    "type": "market_making",
                    "action": "buy_bid",
                    "bid": price.bid_yes,
                    "ask": price.ask_yes,
                    "spread": spread_yes,
                    "target_price": price.bid_yes + 0.005  # Slightly above bid
                }
            ))

        # Could add NO side market making similarly

        return signals

    def _check_pre_event_positioning(
        self,
        snapshot: MarketSnapshot,
        history: list[MarketSnapshot]
    ) -> Optional[StrategySignal]:
        """
        Position before scheduled events when implied volatility is low.

        Before major events (debates, earnings, rulings), markets often
        underprice the expected move. Position early when vol is cheap.
        """
        market = snapshot.market
        current_time = snapshot.price.timestamp

        # Check if resolution date is approaching
        if not market.resolution_date:
            return None

        days_to_resolution = (market.resolution_date - current_time).days

        if not (0 < days_to_resolution <= self.pre_event_days):
            return None

        # Calculate recent volatility
        if len(history) < 5:
            return None

        recent_prices = [h.price.mid_yes for h in history[-10:]]
        if len(recent_prices) < 5:
            return None

        volatility = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0

        # Low volatility before event = opportunity
        # Markets should be pricing in more uncertainty
        if volatility < 0.02:  # Less than 2% daily vol
            current_price = snapshot.price.mid_yes

            # If price is near 50%, expect big move either direction
            # Position based on any slight edge or momentum
            if 0.35 <= current_price <= 0.65:
                # Look at recent momentum
                if len(history) >= 3:
                    recent_trend = recent_prices[-1] - recent_prices[-3]
                    if recent_trend > 0.01:
                        side = Side.YES
                        strength = 0.3 + recent_trend
                    elif recent_trend < -0.01:
                        side = Side.NO
                        strength = 0.3 + abs(recent_trend)
                    else:
                        return None  # No clear direction

                    return StrategySignal(
                        strategy_name=f"{self.name}:pre_event",
                        market_id=market.id,
                        side=side,
                        strength=strength,
                        suggested_size=0.04,
                        timestamp=current_time,
                        metadata={
                            "type": "pre_event_positioning",
                            "days_to_resolution": days_to_resolution,
                            "current_volatility": volatility,
                            "recent_trend": recent_trend,
                            "current_price": current_price
                        }
                    )

        return None

    def _check_resolution_timing(
        self,
        snapshot: MarketSnapshot
    ) -> Optional[StrategySignal]:
        """
        Buy near-certain outcomes just before resolution for small gains.

        When outcome is 95%+ certain, buying and holding to resolution
        gives 5%+ return in a short time. Risk is a surprise reversal.
        """
        market = snapshot.market
        price = snapshot.price
        current_time = price.timestamp

        # Need resolution date
        if not market.resolution_date:
            return None

        days_to_resolution = (market.resolution_date - current_time).days

        # Only play this close to resolution (1-5 days)
        if not (1 <= days_to_resolution <= 5):
            return None

        yes_price = price.mid_yes
        no_price = price.mid_no

        # Check for near-certain YES
        if yes_price >= self.resolution_threshold:
            # Expected profit = 1.0 - yes_price (if YES wins)
            # Risk = yes_price (if NO wins)
            expected_profit = 1.0 - yes_price
            win_prob = yes_price  # Market-implied probability

            # Simple expected value check
            ev = win_prob * expected_profit - (1 - win_prob) * yes_price

            if ev > 0.01:  # Positive EV
                return StrategySignal(
                    strategy_name=f"{self.name}:resolution_timing",
                    market_id=market.id,
                    side=Side.YES,
                    strength=expected_profit * win_prob,
                    suggested_size=0.08,  # Larger size for high-confidence
                    timestamp=current_time,
                    metadata={
                        "type": "resolution_timing",
                        "side": "YES",
                        "price": yes_price,
                        "expected_profit": expected_profit,
                        "days_to_resolution": days_to_resolution,
                        "ev": ev
                    }
                )

        # Check for near-certain NO
        if no_price >= self.resolution_threshold:
            expected_profit = 1.0 - no_price
            win_prob = no_price

            ev = win_prob * expected_profit - (1 - win_prob) * no_price

            if ev > 0.01:
                return StrategySignal(
                    strategy_name=f"{self.name}:resolution_timing",
                    market_id=market.id,
                    side=Side.NO,
                    strength=expected_profit * win_prob,
                    suggested_size=0.08,
                    timestamp=current_time,
                    metadata={
                        "type": "resolution_timing",
                        "side": "NO",
                        "price": no_price,
                        "expected_profit": expected_profit,
                        "days_to_resolution": days_to_resolution,
                        "ev": ev
                    }
                )

        return None

    def _check_volume_spike(
        self,
        snapshot: MarketSnapshot,
        history: list[MarketSnapshot]
    ) -> Optional[StrategySignal]:
        """
        Detect abnormal volume as potential signal.

        Volume spikes often precede price moves as informed traders
        accumulate positions. Trade in direction of volume.
        """
        if len(history) < 10:
            return None

        current_time = snapshot.price.timestamp
        current_volume = snapshot.price.yes_volume + snapshot.price.no_volume

        # Calculate average volume
        historical_volumes = [
            h.price.yes_volume + h.price.no_volume
            for h in history[-20:]
        ]

        if not historical_volumes:
            return None

        avg_volume = statistics.mean(historical_volumes)

        if avg_volume <= 0:
            return None

        volume_ratio = current_volume / avg_volume

        # Detect spike
        if volume_ratio >= self.volume_spike_multiplier:
            # Determine direction from volume imbalance
            yes_vol = snapshot.price.yes_volume
            no_vol = snapshot.price.no_volume
            total_vol = yes_vol + no_vol

            if total_vol <= 0:
                return None

            yes_ratio = yes_vol / total_vol

            # If most volume is YES buys, follow that direction
            if yes_ratio > 0.6:
                side = Side.YES
                strength = (yes_ratio - 0.5) * volume_ratio / 5
            elif yes_ratio < 0.4:
                side = Side.NO
                strength = (0.5 - yes_ratio) * volume_ratio / 5
            else:
                return None  # Balanced volume, no clear signal

            return StrategySignal(
                strategy_name=f"{self.name}:volume_spike",
                market_id=snapshot.market.id,
                side=side,
                strength=min(strength, 0.5),
                suggested_size=0.04,
                timestamp=current_time,
                metadata={
                    "type": "volume_spike",
                    "volume_ratio": volume_ratio,
                    "yes_volume_pct": yes_ratio,
                    "current_volume": current_volume,
                    "avg_volume": avg_volume
                }
            )

        return None

    def _check_liquidity_opportunity(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState
    ) -> Optional[StrategySignal]:
        """
        Trade when liquidity conditions are favorable.

        Sometimes markets have temporary liquidity gaps that allow
        better entry prices than normal.
        """
        price = snapshot.price
        current_time = price.timestamp

        # Check for unusually tight spreads (good for entry)
        spread = price.spread_yes

        if spread < 0.01:  # Very tight spread
            # Good time to enter if we have a position to build
            # or exit if we have a position to close
            market_id = snapshot.market.id

            if market_id in portfolio.positions:
                # Could exit at good prices
                pos = portfolio.positions[market_id]
                current_price = price.mid_yes if pos.side == Side.YES else price.mid_no

                # Check if in profit
                if pos.side == Side.YES:
                    pnl_pct = (current_price - pos.average_entry_price) / pos.average_entry_price
                else:
                    pnl_pct = (current_price - pos.average_entry_price) / pos.average_entry_price

                if pnl_pct > 0.05:  # 5%+ profit, tight spread = good exit
                    # Signal to reduce position (opposite side)
                    exit_side = Side.NO if pos.side == Side.YES else Side.YES
                    return StrategySignal(
                        strategy_name=f"{self.name}:liquidity_exit",
                        market_id=market_id,
                        side=exit_side,
                        strength=0.3,
                        suggested_size=0.5 * pos.quantity / portfolio.total_value,  # Exit half
                        timestamp=current_time,
                        metadata={
                            "type": "liquidity_exit",
                            "spread": spread,
                            "pnl_pct": pnl_pct,
                            "position_side": pos.side.value
                        }
                    )

        return None

    def generate_signals(
        self,
        snapshots: list[MarketSnapshot],
        portfolio: PortfolioState,
        history: list[MarketSnapshot]
    ) -> list[StrategySignal]:
        """
        Generate trading signals based on timing and liquidity.
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

            # Market making
            mm_signals = self._check_market_making_opportunity(snapshot, portfolio)
            signals.extend(mm_signals)

            # Pre-event positioning
            pre_event = self._check_pre_event_positioning(snapshot, market_history)
            if pre_event:
                signals.append(pre_event)

            # Resolution timing
            resolution = self._check_resolution_timing(snapshot)
            if resolution:
                signals.append(resolution)

            # Volume spike detection
            volume = self._check_volume_spike(snapshot, market_history)
            if volume:
                signals.append(volume)

            # Liquidity-based exits
            liquidity = self._check_liquidity_opportunity(snapshot, portfolio)
            if liquidity:
                signals.append(liquidity)

        return signals


class OrderFlowAnalyzer:
    """
    Analyzes order flow patterns to detect informed trading.

    This is more advanced and requires tick-level data.
    """

    def __init__(
        self,
        large_order_threshold: float = 500.0,
        imbalance_threshold: float = 0.7
    ):
        self.large_order_threshold = large_order_threshold
        self.imbalance_threshold = imbalance_threshold

    def analyze_order_flow(
        self,
        trades: list[dict],  # List of recent trades with size, side, timestamp
        current_time: datetime
    ) -> dict:
        """
        Analyze recent trade flow for signals.

        Args:
            trades: List of recent trades with:
                - size: trade size
                - side: "yes" or "no"
                - timestamp: when trade occurred
                - is_taker: whether trade was market order

        Returns:
            Analysis results including imbalance and large order detection
        """
        if not trades:
            return {"signal": None}

        # Filter to recent trades (last hour)
        cutoff = current_time - timedelta(hours=1)
        recent = [t for t in trades if t.get("timestamp", datetime.min) >= cutoff]

        if not recent:
            return {"signal": None}

        # Calculate volume imbalance
        yes_volume = sum(t["size"] for t in recent if t.get("side") == "yes")
        no_volume = sum(t["size"] for t in recent if t.get("side") == "no")
        total_volume = yes_volume + no_volume

        if total_volume == 0:
            return {"signal": None}

        imbalance = (yes_volume - no_volume) / total_volume  # -1 to 1

        # Detect large orders (potential informed trading)
        large_yes = sum(t["size"] for t in recent
                       if t.get("side") == "yes" and t["size"] >= self.large_order_threshold)
        large_no = sum(t["size"] for t in recent
                      if t.get("side") == "no" and t["size"] >= self.large_order_threshold)

        large_order_imbalance = 0
        if large_yes + large_no > 0:
            large_order_imbalance = (large_yes - large_no) / (large_yes + large_no)

        # Generate signal if significant imbalance
        signal = None
        if abs(imbalance) >= self.imbalance_threshold:
            signal = {
                "direction": "YES" if imbalance > 0 else "NO",
                "strength": abs(imbalance),
                "confidence": min(total_volume / 10000, 1.0)
            }

        return {
            "signal": signal,
            "imbalance": imbalance,
            "large_order_imbalance": large_order_imbalance,
            "total_volume": total_volume,
            "yes_volume": yes_volume,
            "no_volume": no_volume
        }
