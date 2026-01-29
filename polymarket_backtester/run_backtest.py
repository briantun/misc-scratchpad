#!/usr/bin/env python3
"""
Main runner for Polymarket backtesting.

This script demonstrates how to:
1. Generate synthetic market data
2. Configure and run the backtesting engine
3. Apply multiple strategies
4. Analyze and report results

Usage:
    python -m polymarket_backtester.run_backtest [OPTIONS]

Options:
    --scenario    Scenario type: election, sports, crypto, random (default: election)
    --markets     Number of markets to simulate (default: 5)
    --days        Days of historical data (default: 90)
    --capital     Initial capital (default: 10000)
    --seed        Random seed for reproducibility (default: 42)
    --output      Output format: text, json, both (default: text)
"""
import argparse
import sys
from datetime import datetime, timedelta

from .models import BacktestConfig
from .engine import BacktestEngine
from .strategies import (
    InformationEdgeStrategy,
    MarketInefficiencyStrategy,
    TimingLiquidityStrategy
)
from .utils import SyntheticDataGenerator, ReportGenerator


def run_backtest(
    scenario_type: str = "election",
    num_markets: int = 5,
    num_days: int = 90,
    initial_capital: float = 10000.0,
    seed: int = 42,
    output_format: str = "text"
) -> None:
    """
    Run a complete backtest with all three strategies.
    """
    print("=" * 60)
    print("POLYMARKET STRATEGY BACKTESTER")
    print("=" * 60)
    print()

    # Generate synthetic data
    print(f"Generating {scenario_type} scenario with {num_markets} markets...")
    generator = SyntheticDataGenerator(seed=seed)
    market_data = generator.generate_scenario(
        scenario_type=scenario_type,
        num_markets=num_markets,
        num_days=num_days,
        seed=seed
    )
    print(f"Generated {len(market_data)} markets with {num_days} days of data each")
    print()

    # Configure backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        max_position_size=0.10,  # 10% max per position
        transaction_fee_pct=0.02,  # 2% fees
        slippage_pct=0.005,  # 0.5% slippage
        rebalance_frequency="daily"
    )

    # Initialize engine
    print("Initializing backtesting engine...")
    engine = BacktestEngine(config)

    # Add strategies
    print("Loading strategies:")

    # Strategy 1: Information Edge
    info_strategy = InformationEdgeStrategy(
        min_edge_threshold=0.05,
        max_signal_age_hours=48.0,
        confidence_decay_rate=0.05
    )
    engine.add_strategy(info_strategy)
    print(f"  - {info_strategy.name}: Exploits external signals (polls, sentiment, models)")

    # Strategy 2: Market Inefficiency
    inefficiency_strategy = MarketInefficiencyStrategy(
        longshot_threshold=0.15,
        favorite_threshold=0.85,
        spread_arbitrage_threshold=0.02,
        mean_reversion_zscore=2.0
    )
    engine.add_strategy(inefficiency_strategy)
    print(f"  - {inefficiency_strategy.name}: Exploits biases and arbitrage")

    # Strategy 3: Timing & Liquidity
    timing_strategy = TimingLiquidityStrategy(
        min_spread_for_mm=0.03,
        resolution_threshold=0.95,
        pre_event_days=5,
        volume_spike_multiplier=2.0
    )
    engine.add_strategy(timing_strategy)
    print(f"  - {timing_strategy.name}: Timing-based opportunities")
    print()

    # Load market data
    print("Loading market data into engine...")
    for market_id, snapshots in market_data.items():
        engine.load_market_data(market_id, snapshots)
        # Print some info about each market
        if snapshots:
            market = snapshots[0].market
            start_price = snapshots[0].price.mid_yes
            end_price = snapshots[-1].price.mid_yes
            outcome = market.resolved_outcome.value if market.resolved_outcome else "unknown"
            print(f"  - {market_id[:8]}: {market.question[:40]}...")
            print(f"    Price: {start_price:.2f} -> {end_price:.2f}, Outcome: {outcome}")
    print()

    # Run backtest
    print("Running backtest...")
    print("-" * 40)
    result = engine.run()
    print("-" * 40)
    print("Backtest complete!")
    print()

    # Generate report
    reporter = ReportGenerator(result)

    if output_format in ("text", "both"):
        print(reporter.generate_text_report())

    if output_format in ("json", "both"):
        print()
        print("JSON Report:")
        print(reporter.generate_json_report())

    # Print quick summary
    print()
    print("=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"Final Portfolio Value: ${result.portfolio_history[-1].total_value:,.2f}" if result.portfolio_history else "N/A")
    print(f"Total Return: {result.total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {result.max_drawdown * 100:.2f}%")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate * 100:.1f}%")

    # Strategy breakdown
    if result.strategy_performance:
        print()
        print("Signals by Strategy:")
        for name, perf in result.strategy_performance.items():
            print(f"  - {name}: {perf.get('signals', 0)} signals, avg strength: {perf.get('avg_strength', 0):.3f}")

    return result


def run_strategy_comparison(
    num_trials: int = 10,
    num_days: int = 60,
    initial_capital: float = 10000.0
) -> None:
    """
    Run multiple trials to compare strategy performance.
    """
    print("=" * 60)
    print("STRATEGY COMPARISON (Multiple Trials)")
    print("=" * 60)
    print()

    results = {
        "all_strategies": [],
        "info_only": [],
        "inefficiency_only": [],
        "timing_only": []
    }

    for trial in range(num_trials):
        seed = 100 + trial
        generator = SyntheticDataGenerator(seed=seed)
        market_data = generator.generate_scenario(
            scenario_type="election",
            num_markets=5,
            num_days=num_days,
            seed=seed
        )

        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)

        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            max_position_size=0.10,
            transaction_fee_pct=0.02,
            slippage_pct=0.005,
            rebalance_frequency="daily"
        )

        # Test each strategy configuration
        strategy_configs = {
            "all_strategies": [
                InformationEdgeStrategy(),
                MarketInefficiencyStrategy(),
                TimingLiquidityStrategy()
            ],
            "info_only": [InformationEdgeStrategy()],
            "inefficiency_only": [MarketInefficiencyStrategy()],
            "timing_only": [TimingLiquidityStrategy()]
        }

        for config_name, strategies in strategy_configs.items():
            engine = BacktestEngine(config)
            for strategy in strategies:
                engine.add_strategy(strategy)
            for market_id, snapshots in market_data.items():
                engine.load_market_data(market_id, snapshots)

            result = engine.run()
            results[config_name].append({
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "total_trades": result.total_trades
            })

        print(f"Trial {trial + 1}/{num_trials} complete")

    # Summarize results
    print()
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print()

    for config_name, trial_results in results.items():
        returns = [r["total_return"] for r in trial_results]
        sharpes = [r["sharpe_ratio"] for r in trial_results]
        drawdowns = [r["max_drawdown"] for r in trial_results]

        avg_return = sum(returns) / len(returns)
        avg_sharpe = sum(sharpes) / len(sharpes)
        avg_drawdown = sum(drawdowns) / len(drawdowns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)

        print(f"{config_name}:")
        print(f"  Avg Return: {avg_return * 100:.2f}%")
        print(f"  Avg Sharpe: {avg_sharpe:.3f}")
        print(f"  Avg Max DD: {avg_drawdown * 100:.2f}%")
        print(f"  Win Rate:   {win_rate * 100:.1f}%")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Strategy Backtester"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="election",
        choices=["election", "sports", "crypto", "random"],
        help="Type of scenario to simulate"
    )
    parser.add_argument(
        "--markets",
        type=int,
        default=5,
        help="Number of markets to simulate"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Days of historical data"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="text",
        choices=["text", "json", "both"],
        help="Output format"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run strategy comparison mode"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials for comparison mode"
    )

    args = parser.parse_args()

    if args.compare:
        run_strategy_comparison(
            num_trials=args.trials,
            num_days=args.days,
            initial_capital=args.capital
        )
    else:
        run_backtest(
            scenario_type=args.scenario,
            num_markets=args.markets,
            num_days=args.days,
            initial_capital=args.capital,
            seed=args.seed,
            output_format=args.output
        )


if __name__ == "__main__":
    main()
