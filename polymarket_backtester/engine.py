"""
Core backtesting engine for Polymarket strategies.
"""
import uuid
from datetime import datetime, timedelta
from typing import Protocol, Optional, TYPE_CHECKING

from .models import (
    BacktestConfig, BacktestResult, Market, MarketSnapshot, MarketStatus,
    Order, OrderStatus, OrderType, Position, PortfolioState, Side,
    StrategySignal, Trade, PricePoint
)

if TYPE_CHECKING:
    from .strategies.llm_auditor import LLMAuditorBase, AuditLogger


class Strategy(Protocol):
    """Protocol for trading strategies."""

    name: str

    def generate_signals(
        self,
        snapshots: list[MarketSnapshot],
        portfolio: PortfolioState,
        history: list[MarketSnapshot]
    ) -> list[StrategySignal]:
        """Generate trading signals based on current market state."""
        ...


class BacktestEngine:
    """Main backtesting engine that orchestrates strategy execution."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategies: list[Strategy] = []
        self.market_data: dict[str, list[MarketSnapshot]] = {}  # market_id -> snapshots
        self.portfolio: PortfolioState = None
        self.trades: list[Trade] = []
        self.signals: list[StrategySignal] = []
        self.portfolio_history: list[PortfolioState] = []

        # LLM auditor (optional)
        self.auditor: Optional["LLMAuditorBase"] = None
        self.audit_logger: Optional["AuditLogger"] = None
        self.audit_stats = {"approved": 0, "rejected": 0, "modified": 0}

    def add_strategy(self, strategy: Strategy) -> None:
        """Register a strategy with the engine."""
        self.strategies.append(strategy)

    def set_auditor(
        self,
        auditor: "LLMAuditorBase",
        logger: Optional["AuditLogger"] = None
    ) -> None:
        """
        Set an LLM auditor to review trades before execution.

        Args:
            auditor: LLM auditor instance
            logger: Optional audit logger for tracking decisions
        """
        self.auditor = auditor
        self.audit_logger = logger

    def load_market_data(self, market_id: str, snapshots: list[MarketSnapshot]) -> None:
        """Load historical market data for backtesting."""
        # Sort by timestamp
        sorted_snapshots = sorted(snapshots, key=lambda s: s.price.timestamp)
        self.market_data[market_id] = sorted_snapshots

    def _initialize_portfolio(self) -> None:
        """Initialize portfolio state."""
        self.portfolio = PortfolioState(
            timestamp=self.config.start_date,
            cash=self.config.initial_capital,
            positions={},
            total_value=self.config.initial_capital,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )

    def _get_snapshots_at_time(self, timestamp: datetime) -> list[MarketSnapshot]:
        """Get latest market snapshots at or before the given timestamp."""
        snapshots = []
        for market_id, market_snapshots in self.market_data.items():
            # Find the most recent snapshot at or before timestamp
            latest = None
            for snapshot in market_snapshots:
                if snapshot.price.timestamp <= timestamp:
                    latest = snapshot
                else:
                    break
            if latest and latest.market.status == MarketStatus.OPEN:
                snapshots.append(latest)
        return snapshots

    def _get_history_for_market(
        self, market_id: str, before: datetime, lookback_days: int = 30
    ) -> list[MarketSnapshot]:
        """Get historical snapshots for a market."""
        if market_id not in self.market_data:
            return []

        cutoff = before - timedelta(days=lookback_days)
        return [
            s for s in self.market_data[market_id]
            if cutoff <= s.price.timestamp < before
        ]

    def _execute_order(
        self, order: Order, snapshot: MarketSnapshot, timestamp: datetime
    ) -> Trade | None:
        """Execute an order against current market state."""
        price_point = snapshot.price

        # Determine execution price
        if order.side == Side.YES:
            base_price = price_point.ask_yes
        else:
            base_price = price_point.ask_no

        # Apply slippage
        exec_price = base_price * (1 + self.config.slippage_pct)

        # For limit orders, check if price is acceptable
        if order.order_type == OrderType.LIMIT:
            if exec_price > order.limit_price:
                return None  # Price too high, don't fill

        # Calculate fees
        fees = order.quantity * exec_price * self.config.transaction_fee_pct

        # Check if we have enough cash
        total_cost = order.quantity * exec_price + fees
        if total_cost > self.portfolio.cash:
            # Reduce order size to fit available cash
            max_quantity = (self.portfolio.cash - fees) / (exec_price * (1 + self.config.transaction_fee_pct))
            if max_quantity <= 0:
                return None
            order.quantity = max_quantity
            fees = order.quantity * exec_price * self.config.transaction_fee_pct
            total_cost = order.quantity * exec_price + fees

        # Create trade
        trade = Trade(
            id=str(uuid.uuid4()),
            order_id=order.id,
            market_id=order.market_id,
            side=order.side,
            quantity=order.quantity,
            price=exec_price,
            timestamp=timestamp,
            fees=fees
        )

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_at = timestamp
        order.filled_price = exec_price
        order.filled_quantity = order.quantity

        return trade

    def _update_portfolio(self, trade: Trade) -> None:
        """Update portfolio state after a trade."""
        market_id = trade.market_id
        total_cost = trade.total_cost

        # Deduct cash
        self.portfolio.cash -= total_cost

        # Update or create position
        if market_id in self.portfolio.positions:
            pos = self.portfolio.positions[market_id]
            if pos.side == trade.side:
                # Adding to position
                new_quantity = pos.quantity + trade.quantity
                new_avg_price = (
                    (pos.quantity * pos.average_entry_price + trade.quantity * trade.price)
                    / new_quantity
                )
                pos.quantity = new_quantity
                pos.average_entry_price = new_avg_price
            else:
                # Reducing or closing position
                if trade.quantity >= pos.quantity:
                    # Close position and possibly open opposite
                    realized = pos.quantity * (trade.price - pos.average_entry_price)
                    if pos.side == Side.NO:
                        realized = -realized  # Flip sign for NO positions
                    pos.realized_pnl += realized
                    self.portfolio.realized_pnl += realized

                    remaining = trade.quantity - pos.quantity
                    if remaining > 0:
                        # Open opposite position
                        self.portfolio.positions[market_id] = Position(
                            market_id=market_id,
                            side=trade.side,
                            quantity=remaining,
                            average_entry_price=trade.price
                        )
                    else:
                        del self.portfolio.positions[market_id]
                else:
                    # Partial close
                    realized = trade.quantity * (trade.price - pos.average_entry_price)
                    if pos.side == Side.NO:
                        realized = -realized
                    pos.realized_pnl += realized
                    self.portfolio.realized_pnl += realized
                    pos.quantity -= trade.quantity
        else:
            # New position
            self.portfolio.positions[market_id] = Position(
                market_id=market_id,
                side=trade.side,
                quantity=trade.quantity,
                average_entry_price=trade.price
            )

    def _update_unrealized_pnl(self, snapshots: list[MarketSnapshot]) -> None:
        """Update unrealized P&L based on current prices."""
        snapshot_map = {s.market.id: s for s in snapshots}
        total_unrealized = 0.0
        position_value = 0.0

        for market_id, pos in self.portfolio.positions.items():
            if market_id in snapshot_map:
                snapshot = snapshot_map[market_id]
                if pos.side == Side.YES:
                    current_price = snapshot.price.bid_yes
                else:
                    current_price = snapshot.price.bid_no

                pos.unrealized_pnl = pos.quantity * (current_price - pos.average_entry_price)
                total_unrealized += pos.unrealized_pnl
                position_value += pos.quantity * current_price

        self.portfolio.unrealized_pnl = total_unrealized
        self.portfolio.total_value = self.portfolio.cash + position_value

    def _process_resolutions(self, timestamp: datetime) -> None:
        """Handle market resolutions and settle positions."""
        for market_id, snapshots in self.market_data.items():
            if market_id not in self.portfolio.positions:
                continue

            # Check if market resolved
            for snapshot in snapshots:
                if (snapshot.price.timestamp <= timestamp and
                    snapshot.market.status == MarketStatus.RESOLVED):

                    pos = self.portfolio.positions[market_id]
                    outcome = snapshot.market.resolved_outcome

                    # Settlement: winning side pays $1, losing side pays $0
                    if pos.side == outcome:
                        settlement_price = 1.0
                    else:
                        settlement_price = 0.0

                    realized = pos.quantity * (settlement_price - pos.average_entry_price)
                    self.portfolio.realized_pnl += realized
                    self.portfolio.cash += pos.quantity * settlement_price

                    del self.portfolio.positions[market_id]
                    break

    def _aggregate_signals(self, signals: list[StrategySignal]) -> list[Order]:
        """Combine signals from multiple strategies into orders."""
        # Group signals by market
        market_signals: dict[str, list[StrategySignal]] = {}
        for signal in signals:
            if signal.market_id not in market_signals:
                market_signals[signal.market_id] = []
            market_signals[signal.market_id].append(signal)

        orders = []
        for market_id, sigs in market_signals.items():
            # Weight signals by strength
            weighted_yes = sum(
                s.strength * s.suggested_size
                for s in sigs if s.side == Side.YES
            )
            weighted_no = sum(
                s.strength * s.suggested_size
                for s in sigs if s.side == Side.NO
            )

            # Determine net direction
            if weighted_yes > weighted_no and weighted_yes > 0.1:
                side = Side.YES
                strength = weighted_yes
            elif weighted_no > weighted_yes and weighted_no > 0.1:
                side = Side.NO
                strength = weighted_no
            else:
                continue  # No clear signal

            # Calculate position size
            max_size = self.portfolio.total_value * self.config.max_position_size
            target_size = max_size * min(strength, 1.0)

            # Check existing position
            if market_id in self.portfolio.positions:
                pos = self.portfolio.positions[market_id]
                if pos.side == side:
                    # Already have position in same direction
                    continue
                # Could add logic to close opposite positions here

            # Create order
            order = Order(
                id=str(uuid.uuid4()),
                market_id=market_id,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=target_size,  # Will be adjusted by execution
                limit_price=0.95 if side == Side.YES else 0.95,
                strategy_name=",".join(s.strategy_name for s in sigs),
                signal_strength=strength
            )
            orders.append(order)

        return orders

    def run(self) -> BacktestResult:
        """Execute the backtest."""
        self._initialize_portfolio()

        # Determine time steps
        current_time = self.config.start_date
        if self.config.rebalance_frequency == "daily":
            step = timedelta(days=1)
        elif self.config.rebalance_frequency == "hourly":
            step = timedelta(hours=1)
        else:
            step = timedelta(days=1)

        while current_time <= self.config.end_date:
            # Get current market state
            snapshots = self._get_snapshots_at_time(current_time)

            if not snapshots:
                current_time += step
                continue

            # Process any market resolutions
            self._process_resolutions(current_time)

            # Generate signals from all strategies
            all_signals = []
            for strategy in self.strategies:
                history = {}
                for s in snapshots:
                    history[s.market.id] = self._get_history_for_market(
                        s.market.id, current_time
                    )

                signals = strategy.generate_signals(
                    snapshots, self.portfolio,
                    [h for hist in history.values() for h in hist]
                )
                all_signals.extend(signals)
                self.signals.extend(signals)

            # Convert signals to orders
            orders = self._aggregate_signals(all_signals)

            # Execute orders (with optional LLM audit)
            snapshot_map = {s.market.id: s for s in snapshots}
            for order in orders:
                if order.market_id not in snapshot_map:
                    continue

                snapshot = snapshot_map[order.market_id]

                # Find the signal(s) that generated this order
                order_signals = [s for s in all_signals if s.market_id == order.market_id]
                primary_signal = order_signals[0] if order_signals else None

                # LLM audit if enabled
                should_execute = True
                if self.auditor and primary_signal:
                    from .strategies.llm_auditor import AuditDecision

                    # Get recent signals for context
                    recent_signals = self.signals[-20:] if self.signals else []

                    audit_result = self.auditor.audit_trade(
                        signal=primary_signal,
                        snapshot=snapshot,
                        portfolio=self.portfolio,
                        recent_signals=recent_signals
                    )

                    # Process audit decision
                    if audit_result.decision == AuditDecision.APPROVE:
                        should_execute = True
                        self.audit_stats["approved"] += 1
                        final_action = "executed"
                    elif audit_result.decision == AuditDecision.MODIFY:
                        should_execute = True
                        self.audit_stats["modified"] += 1
                        final_action = "modified"
                        # Apply modifications
                        if audit_result.suggested_modifications:
                            new_size = audit_result.suggested_modifications.get("new_size")
                            if new_size:
                                order.quantity = self.portfolio.total_value * new_size
                    elif audit_result.decision == AuditDecision.REJECT:
                        should_execute = False
                        self.audit_stats["rejected"] += 1
                        final_action = "skipped"
                    else:  # DEFER
                        should_execute = False
                        self.audit_stats["rejected"] += 1
                        final_action = "skipped"

                    # Log audit decision
                    if self.audit_logger:
                        self.audit_logger.log(primary_signal, audit_result, final_action)

                # Execute if approved
                if should_execute:
                    trade = self._execute_order(order, snapshot, current_time)
                    if trade:
                        self.trades.append(trade)
                        self._update_portfolio(trade)

            # Update portfolio valuation
            self._update_unrealized_pnl(snapshots)
            self.portfolio.timestamp = current_time

            # Record portfolio state
            self.portfolio_history.append(PortfolioState(
                timestamp=current_time,
                cash=self.portfolio.cash,
                positions=dict(self.portfolio.positions),
                total_value=self.portfolio.total_value,
                unrealized_pnl=self.portfolio.unrealized_pnl,
                realized_pnl=self.portfolio.realized_pnl
            ))

            current_time += step

        # Calculate final metrics
        return self._calculate_results()

    def _calculate_results(self) -> BacktestResult:
        """Calculate performance metrics."""
        if not self.portfolio_history:
            return BacktestResult(
                config=self.config,
                portfolio_history=[],
                trades=[],
                signals=[]
            )

        # Total return
        initial = self.config.initial_capital
        final = self.portfolio_history[-1].total_value
        total_return = (final - initial) / initial

        # Calculate daily returns for Sharpe ratio
        daily_returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_value = self.portfolio_history[i-1].total_value
            curr_value = self.portfolio_history[i].total_value
            if prev_value > 0:
                daily_returns.append((curr_value - prev_value) / prev_value)

        # Sharpe ratio (assuming 0% risk-free rate)
        if daily_returns:
            import statistics
            mean_return = statistics.mean(daily_returns)
            std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 1
            sharpe_ratio = (mean_return * 252**0.5) / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Max drawdown
        peak = initial
        max_drawdown = 0
        for state in self.portfolio_history:
            if state.total_value > peak:
                peak = state.total_value
            drawdown = (peak - state.total_value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Win rate and profit factor
        winning_trades = 0
        gross_profit = 0
        gross_loss = 0

        for trade in self.trades:
            # Simplified: check if trade was profitable based on subsequent price
            # In reality, would need to track to resolution
            if trade.price < 0.5 and trade.side == Side.YES:
                winning_trades += 1
                gross_profit += trade.quantity * (0.5 - trade.price)
            elif trade.price > 0.5 and trade.side == Side.NO:
                winning_trades += 1
                gross_profit += trade.quantity * (trade.price - 0.5)
            else:
                gross_loss += trade.quantity * 0.1  # Estimated loss

        win_rate = winning_trades / len(self.trades) if self.trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Strategy breakdown
        strategy_perf = {}
        for signal in self.signals:
            if signal.strategy_name not in strategy_perf:
                strategy_perf[signal.strategy_name] = {
                    "signals": 0,
                    "avg_strength": 0
                }
            strategy_perf[signal.strategy_name]["signals"] += 1
            strategy_perf[signal.strategy_name]["avg_strength"] += signal.strength

        for name, perf in strategy_perf.items():
            if perf["signals"] > 0:
                perf["avg_strength"] /= perf["signals"]

        return BacktestResult(
            config=self.config,
            portfolio_history=self.portfolio_history,
            trades=self.trades,
            signals=self.signals,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            strategy_performance=strategy_perf
        )
