"""
Performance metrics and reporting for backtest results.
"""
from datetime import datetime, timedelta
from typing import Optional
import statistics
import json

from ..models import BacktestResult, PortfolioState, Trade, StrategySignal, Side


class PerformanceAnalyzer:
    """
    Calculates detailed performance metrics from backtest results.
    """

    def __init__(self, result: BacktestResult):
        self.result = result
        self._metrics_cache: dict = {}

    def calculate_all_metrics(self) -> dict:
        """Calculate comprehensive performance metrics."""
        if self._metrics_cache:
            return self._metrics_cache

        metrics = {
            "summary": self._calculate_summary_metrics(),
            "returns": self._calculate_return_metrics(),
            "risk": self._calculate_risk_metrics(),
            "trading": self._calculate_trading_metrics(),
            "strategy_breakdown": self._calculate_strategy_breakdown(),
            "time_analysis": self._calculate_time_analysis()
        }

        self._metrics_cache = metrics
        return metrics

    def _calculate_summary_metrics(self) -> dict:
        """High-level summary statistics."""
        history = self.result.portfolio_history
        if not history:
            return {}

        initial = self.result.config.initial_capital
        final = history[-1].total_value

        return {
            "initial_capital": initial,
            "final_value": final,
            "total_return": (final - initial) / initial,
            "total_return_pct": ((final - initial) / initial) * 100,
            "total_pnl": final - initial,
            "realized_pnl": history[-1].realized_pnl,
            "unrealized_pnl": history[-1].unrealized_pnl,
            "total_trades": len(self.result.trades),
            "total_signals": len(self.result.signals),
            "backtest_days": (self.result.config.end_date - self.result.config.start_date).days
        }

    def _calculate_return_metrics(self) -> dict:
        """Detailed return analysis."""
        history = self.result.portfolio_history
        if len(history) < 2:
            return {}

        # Daily returns
        daily_returns = []
        for i in range(1, len(history)):
            prev = history[i-1].total_value
            curr = history[i].total_value
            if prev > 0:
                daily_returns.append((curr - prev) / prev)

        if not daily_returns:
            return {}

        # Annualization factor (assuming daily data)
        annual_factor = 252

        mean_daily = statistics.mean(daily_returns)
        std_daily = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0

        # Calculate percentiles
        sorted_returns = sorted(daily_returns)
        n = len(sorted_returns)

        return {
            "mean_daily_return": mean_daily,
            "std_daily_return": std_daily,
            "annualized_return": mean_daily * annual_factor,
            "annualized_volatility": std_daily * (annual_factor ** 0.5),
            "best_day": max(daily_returns),
            "worst_day": min(daily_returns),
            "median_daily_return": sorted_returns[n // 2],
            "percentile_5": sorted_returns[int(n * 0.05)] if n >= 20 else None,
            "percentile_95": sorted_returns[int(n * 0.95)] if n >= 20 else None,
            "positive_days": sum(1 for r in daily_returns if r > 0),
            "negative_days": sum(1 for r in daily_returns if r < 0),
            "positive_day_pct": sum(1 for r in daily_returns if r > 0) / len(daily_returns)
        }

    def _calculate_risk_metrics(self) -> dict:
        """Risk-adjusted performance metrics."""
        history = self.result.portfolio_history
        if len(history) < 2:
            return {}

        # Daily returns for calculations
        daily_returns = []
        values = []
        for i in range(1, len(history)):
            prev = history[i-1].total_value
            curr = history[i].total_value
            values.append(curr)
            if prev > 0:
                daily_returns.append((curr - prev) / prev)

        if not daily_returns:
            return {}

        mean_return = statistics.mean(daily_returns)
        std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 1

        # Sharpe Ratio (assuming 0% risk-free rate)
        sharpe = (mean_return * 252**0.5) / std_return if std_return > 0 else 0

        # Sortino Ratio (downside deviation)
        negative_returns = [r for r in daily_returns if r < 0]
        downside_std = statistics.stdev(negative_returns) if len(negative_returns) > 1 else std_return
        sortino = (mean_return * 252**0.5) / downside_std if downside_std > 0 else 0

        # Maximum Drawdown
        peak = self.result.config.initial_capital
        max_dd = 0
        max_dd_duration = 0
        current_dd_start = None

        for i, state in enumerate(history):
            if state.total_value > peak:
                peak = state.total_value
                if current_dd_start is not None:
                    dd_duration = i - current_dd_start
                    max_dd_duration = max(max_dd_duration, dd_duration)
                current_dd_start = None
            else:
                if current_dd_start is None:
                    current_dd_start = i
                dd = (peak - state.total_value) / peak
                max_dd = max(max_dd, dd)

        # Calmar Ratio
        annual_return = mean_return * 252
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # Value at Risk (95%)
        sorted_returns = sorted(daily_returns)
        var_95 = sorted_returns[int(len(sorted_returns) * 0.05)] if len(sorted_returns) >= 20 else min(daily_returns)

        # Expected Shortfall (CVaR)
        var_index = int(len(sorted_returns) * 0.05)
        cvar_95 = statistics.mean(sorted_returns[:var_index + 1]) if var_index > 0 else var_95

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd * 100,
            "max_drawdown_duration_days": max_dd_duration,
            "var_95_daily": var_95,
            "cvar_95_daily": cvar_95,
            "downside_deviation": downside_std
        }

    def _calculate_trading_metrics(self) -> dict:
        """Trade-level performance analysis."""
        trades = self.result.trades
        if not trades:
            return {}

        # Separate by outcome (simplified - based on entry price)
        winning_trades = []
        losing_trades = []

        for trade in trades:
            # Simple heuristic: if bought below 0.5 on YES, likely winner
            # This is imperfect without actual resolution data
            if trade.side == Side.YES and trade.price < 0.5:
                winning_trades.append(trade)
            elif trade.side == Side.NO and trade.price < 0.5:
                winning_trades.append(trade)
            else:
                losing_trades.append(trade)

        total_fees = sum(t.fees for t in trades)
        total_volume = sum(t.quantity * t.price for t in trades)

        avg_trade_size = statistics.mean([t.quantity * t.price for t in trades])

        # Trade frequency
        if len(trades) >= 2:
            trade_times = sorted([t.timestamp for t in trades])
            time_diffs = [(trade_times[i+1] - trade_times[i]).total_seconds() / 3600
                         for i in range(len(trade_times) - 1)]
            avg_hours_between_trades = statistics.mean(time_diffs) if time_diffs else 0
        else:
            avg_hours_between_trades = 0

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades) if trades else 0,
            "total_fees": total_fees,
            "total_volume": total_volume,
            "avg_trade_size": avg_trade_size,
            "avg_hours_between_trades": avg_hours_between_trades,
            "fees_as_pct_of_volume": (total_fees / total_volume * 100) if total_volume > 0 else 0
        }

    def _calculate_strategy_breakdown(self) -> dict:
        """Performance breakdown by strategy."""
        signals = self.result.signals
        if not signals:
            return {}

        strategy_stats: dict[str, dict] = {}

        for signal in signals:
            name = signal.strategy_name.split(":")[0]  # Base strategy name
            if name not in strategy_stats:
                strategy_stats[name] = {
                    "signals": 0,
                    "total_strength": 0,
                    "yes_signals": 0,
                    "no_signals": 0,
                    "avg_suggested_size": []
                }

            stats = strategy_stats[name]
            stats["signals"] += 1
            stats["total_strength"] += signal.strength
            if signal.side == Side.YES:
                stats["yes_signals"] += 1
            else:
                stats["no_signals"] += 1
            stats["avg_suggested_size"].append(signal.suggested_size)

        # Calculate averages
        for name, stats in strategy_stats.items():
            if stats["signals"] > 0:
                stats["avg_strength"] = stats["total_strength"] / stats["signals"]
                stats["avg_suggested_size"] = statistics.mean(stats["avg_suggested_size"])
                stats["yes_pct"] = stats["yes_signals"] / stats["signals"]
            del stats["total_strength"]

        return strategy_stats

    def _calculate_time_analysis(self) -> dict:
        """Time-based performance analysis."""
        history = self.result.portfolio_history
        if len(history) < 7:
            return {}

        # Weekly returns
        weekly_values = []
        for i in range(0, len(history), 7):
            if i < len(history):
                weekly_values.append(history[i].total_value)

        weekly_returns = []
        for i in range(1, len(weekly_values)):
            if weekly_values[i-1] > 0:
                weekly_returns.append((weekly_values[i] - weekly_values[i-1]) / weekly_values[i-1])

        # Monthly returns (approx 30 days)
        monthly_values = []
        for i in range(0, len(history), 30):
            if i < len(history):
                monthly_values.append(history[i].total_value)

        monthly_returns = []
        for i in range(1, len(monthly_values)):
            if monthly_values[i-1] > 0:
                monthly_returns.append((monthly_values[i] - monthly_values[i-1]) / monthly_values[i-1])

        result = {
            "num_weeks": len(weekly_returns),
            "num_months": len(monthly_returns)
        }

        if weekly_returns:
            result["avg_weekly_return"] = statistics.mean(weekly_returns)
            result["weekly_win_rate"] = sum(1 for r in weekly_returns if r > 0) / len(weekly_returns)

        if monthly_returns:
            result["avg_monthly_return"] = statistics.mean(monthly_returns)
            result["monthly_win_rate"] = sum(1 for r in monthly_returns if r > 0) / len(monthly_returns)
            result["best_month"] = max(monthly_returns)
            result["worst_month"] = min(monthly_returns)

        return result


class ReportGenerator:
    """
    Generates formatted reports from backtest results.
    """

    def __init__(self, result: BacktestResult):
        self.result = result
        self.analyzer = PerformanceAnalyzer(result)

    def generate_text_report(self) -> str:
        """Generate a text-based performance report."""
        metrics = self.analyzer.calculate_all_metrics()

        lines = [
            "=" * 60,
            "POLYMARKET BACKTEST REPORT",
            "=" * 60,
            "",
            "CONFIGURATION",
            "-" * 40,
            f"Period: {self.result.config.start_date.date()} to {self.result.config.end_date.date()}",
            f"Initial Capital: ${self.result.config.initial_capital:,.2f}",
            f"Max Position Size: {self.result.config.max_position_size * 100:.1f}%",
            f"Transaction Fee: {self.result.config.transaction_fee_pct * 100:.2f}%",
            "",
        ]

        # Summary
        summary = metrics.get("summary", {})
        if summary:
            lines.extend([
                "SUMMARY",
                "-" * 40,
                f"Final Value: ${summary.get('final_value', 0):,.2f}",
                f"Total Return: {summary.get('total_return_pct', 0):.2f}%",
                f"Total P&L: ${summary.get('total_pnl', 0):,.2f}",
                f"  - Realized: ${summary.get('realized_pnl', 0):,.2f}",
                f"  - Unrealized: ${summary.get('unrealized_pnl', 0):,.2f}",
                f"Total Trades: {summary.get('total_trades', 0)}",
                f"Total Signals: {summary.get('total_signals', 0)}",
                "",
            ])

        # Returns
        returns = metrics.get("returns", {})
        if returns:
            lines.extend([
                "RETURN ANALYSIS",
                "-" * 40,
                f"Annualized Return: {returns.get('annualized_return', 0) * 100:.2f}%",
                f"Annualized Volatility: {returns.get('annualized_volatility', 0) * 100:.2f}%",
                f"Best Day: {returns.get('best_day', 0) * 100:.2f}%",
                f"Worst Day: {returns.get('worst_day', 0) * 100:.2f}%",
                f"Positive Days: {returns.get('positive_days', 0)} ({returns.get('positive_day_pct', 0) * 100:.1f}%)",
                "",
            ])

        # Risk
        risk = metrics.get("risk", {})
        if risk:
            lines.extend([
                "RISK METRICS",
                "-" * 40,
                f"Sharpe Ratio: {risk.get('sharpe_ratio', 0):.3f}",
                f"Sortino Ratio: {risk.get('sortino_ratio', 0):.3f}",
                f"Calmar Ratio: {risk.get('calmar_ratio', 0):.3f}",
                f"Max Drawdown: {risk.get('max_drawdown_pct', 0):.2f}%",
                f"Max DD Duration: {risk.get('max_drawdown_duration_days', 0)} days",
                f"VaR (95%, daily): {risk.get('var_95_daily', 0) * 100:.2f}%",
                "",
            ])

        # Trading
        trading = metrics.get("trading", {})
        if trading:
            lines.extend([
                "TRADING STATISTICS",
                "-" * 40,
                f"Total Trades: {trading.get('total_trades', 0)}",
                f"Win Rate: {trading.get('win_rate', 0) * 100:.1f}%",
                f"Total Volume: ${trading.get('total_volume', 0):,.2f}",
                f"Total Fees: ${trading.get('total_fees', 0):,.2f}",
                f"Avg Trade Size: ${trading.get('avg_trade_size', 0):,.2f}",
                "",
            ])

        # Strategy breakdown
        strategy = metrics.get("strategy_breakdown", {})
        if strategy:
            lines.extend([
                "STRATEGY BREAKDOWN",
                "-" * 40,
            ])
            for name, stats in strategy.items():
                lines.append(f"  {name}:")
                lines.append(f"    Signals: {stats.get('signals', 0)}")
                lines.append(f"    Avg Strength: {stats.get('avg_strength', 0):.3f}")
                lines.append(f"    YES/NO Split: {stats.get('yes_pct', 0) * 100:.1f}% / {(1 - stats.get('yes_pct', 0)) * 100:.1f}%")
            lines.append("")

        lines.extend([
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])

        return "\n".join(lines)

    def generate_json_report(self) -> str:
        """Generate a JSON-formatted report."""
        metrics = self.analyzer.calculate_all_metrics()

        # Add config info
        report = {
            "config": {
                "start_date": self.result.config.start_date.isoformat(),
                "end_date": self.result.config.end_date.isoformat(),
                "initial_capital": self.result.config.initial_capital,
                "max_position_size": self.result.config.max_position_size,
                "transaction_fee_pct": self.result.config.transaction_fee_pct
            },
            "metrics": metrics
        }

        return json.dumps(report, indent=2, default=str)

    def generate_equity_curve_data(self) -> list[dict]:
        """Generate equity curve data for plotting."""
        return [
            {
                "timestamp": state.timestamp.isoformat(),
                "total_value": state.total_value,
                "cash": state.cash,
                "position_value": state.total_value - state.cash,
                "unrealized_pnl": state.unrealized_pnl,
                "realized_pnl": state.realized_pnl
            }
            for state in self.result.portfolio_history
        ]

    def generate_trade_log(self) -> list[dict]:
        """Generate detailed trade log."""
        return [
            {
                "id": trade.id,
                "timestamp": trade.timestamp.isoformat(),
                "market_id": trade.market_id,
                "side": trade.side.value,
                "quantity": trade.quantity,
                "price": trade.price,
                "fees": trade.fees,
                "total_cost": trade.total_cost
            }
            for trade in self.result.trades
        ]
