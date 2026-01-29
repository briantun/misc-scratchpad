#!/usr/bin/env python3
"""
CLI for the Polymarket Market Researcher.

Usage:
    python -m polymarket_backtester.researcher.cli <market_url_or_id>
    python -m polymarket_backtester.researcher.cli --help
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime

from .market_researcher import MarketResearcher, TradeAction


def format_recommendation(result) -> str:
    """Format the research result for display."""
    rec = result.recommendation

    # Action emoji
    action_emoji = {
        TradeAction.STRONG_BUY_YES: "🟢🟢",
        TradeAction.BUY_YES: "🟢",
        TradeAction.LEAN_YES: "🟡➡️🟢",
        TradeAction.HOLD: "⚪",
        TradeAction.LEAN_NO: "🟡➡️🔴",
        TradeAction.BUY_NO: "🔴",
        TradeAction.STRONG_BUY_NO: "🔴🔴",
    }

    # Edge formatting
    edge_pct = rec.edge * 100
    edge_str = f"+{edge_pct:.1f}%" if edge_pct > 0 else f"{edge_pct:.1f}%"

    output = f"""
{'='*70}
POLYMARKET RESEARCH REPORT
{'='*70}

MARKET: {result.market_question}
URL: {result.market_url}
Time: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{'─'*70}
CURRENT MARKET
{'─'*70}
  YES Price: ${result.current_yes_price:.2f} ({result.current_yes_price*100:.1f}%)
  NO Price:  ${result.current_no_price:.2f} ({result.current_no_price*100:.1f}%)
  Volume:    ${result.volume_24h:,.0f}
  Liquidity: ${result.liquidity:,.0f}
  Resolves:  {result.resolution_date.strftime('%Y-%m-%d') if result.resolution_date else 'Unknown'}

{'─'*70}
RECOMMENDATION  {action_emoji.get(rec.action, '')}
{'─'*70}
  Action:           {rec.action.value.upper().replace('_', ' ')}
  Confidence:       {rec.confidence*100:.0f}%

  My Fair Value:    {rec.fair_probability*100:.1f}%
  Market Price:     {rec.market_probability*100:.1f}%
  Edge:             {edge_str}

  Position Size:    {rec.position_size_pct:.1f}% of bankroll
  Expected Value:   {rec.expected_value*100:.1f}% per dollar risked

{'─'*70}
KEY FACTORS
{'─'*70}
"""
    for i, factor in enumerate(rec.key_factors, 1):
        output += f"  {i}. {factor}\n"

    output += f"""
{'─'*70}
RISKS
{'─'*70}
"""
    for i, risk in enumerate(rec.risks, 1):
        output += f"  {i}. {risk}\n"

    output += f"""
{'─'*70}
REASONING
{'─'*70}
{rec.reasoning}

{'─'*70}
RESEARCH SUMMARY
{'─'*70}
{result.research_summary}

{'='*70}
"""
    return output


def format_simple(result) -> str:
    """Format a simple one-line summary."""
    rec = result.recommendation
    edge_pct = rec.edge * 100

    return (
        f"{result.market_question[:50]}... | "
        f"Action: {rec.action.value} | "
        f"Fair: {rec.fair_probability*100:.0f}% vs Market: {rec.market_probability*100:.0f}% | "
        f"Edge: {edge_pct:+.1f}% | "
        f"Confidence: {rec.confidence*100:.0f}%"
    )


async def analyze_single_market(
    market: str,
    output_format: str = "text",
    model: str = "claude-sonnet-4-20250514",
) -> None:
    """Analyze a single market."""

    print(f"\nAnalyzing market: {market}")
    print("This may take a moment...\n")

    async with MarketResearcher(model=model) as researcher:
        try:
            result = await researcher.analyze_market(market)

            if output_format == "json":
                print(json.dumps(result.to_dict(), indent=2))
            elif output_format == "simple":
                print(format_simple(result))
            else:
                print(format_recommendation(result))

        except Exception as e:
            print(f"Error analyzing market: {e}", file=sys.stderr)
            sys.exit(1)


async def analyze_demo_markets(
    market_data_list: list[dict],
    output_format: str = "text",
    model: str = "claude-sonnet-4-20250514",
) -> None:
    """Analyze demo markets using provided data."""

    results = []

    async with MarketResearcher(model=model) as researcher:
        for i, market_data in enumerate(market_data_list, 1):
            print(f"\n[{i}/{len(market_data_list)}] Analyzing: {market_data['question'][:50]}...")

            try:
                result = await researcher.analyze_from_data(market_data)
                results.append(result)

                if output_format == "simple":
                    print(format_simple(result))
                elif output_format == "text":
                    print(format_recommendation(result))

            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

    if output_format == "json":
        print(json.dumps([r.to_dict() for r in results], indent=2))

    # Summary
    if len(results) > 1 and output_format != "json":
        print_summary(results)


def print_summary(results: list) -> None:
    """Print summary of multiple market analyses."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    buy_yes = [r for r in results if r.recommendation.action in [
        TradeAction.STRONG_BUY_YES, TradeAction.BUY_YES, TradeAction.LEAN_YES
    ]]
    buy_no = [r for r in results if r.recommendation.action in [
        TradeAction.STRONG_BUY_NO, TradeAction.BUY_NO, TradeAction.LEAN_NO
    ]]
    holds = [r for r in results if r.recommendation.action == TradeAction.HOLD]

    print(f"\nTotal markets analyzed: {len(results)}")
    print(f"  Buy YES recommendations: {len(buy_yes)}")
    print(f"  Buy NO recommendations:  {len(buy_no)}")
    print(f"  Hold recommendations:    {len(holds)}")

    if buy_yes or buy_no:
        print("\nTop opportunities by edge:")
        all_trades = buy_yes + buy_no
        all_trades.sort(key=lambda r: abs(r.recommendation.edge), reverse=True)
        for r in all_trades[:5]:
            print(f"  {r.recommendation.action.value}: {r.market_question[:40]}... "
                  f"(edge: {r.recommendation.edge*100:+.1f}%)")


async def analyze_multiple_markets(
    markets: list[str],
    output_format: str = "text",
    model: str = "claude-sonnet-4-20250514",
) -> None:
    """Analyze multiple markets."""

    results = []

    async with MarketResearcher(model=model) as researcher:
        for i, market in enumerate(markets, 1):
            print(f"\n[{i}/{len(markets)}] Analyzing: {market}")

            try:
                result = await researcher.analyze_market(market)
                results.append(result)

                if output_format == "simple":
                    print(format_simple(result))
                elif output_format == "text":
                    print(format_recommendation(result))

            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

    if output_format == "json":
        print(json.dumps([r.to_dict() for r in results], indent=2))

    # Summary
    if len(results) > 1 and output_format != "json":
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        buy_yes = [r for r in results if r.recommendation.action in [
            TradeAction.STRONG_BUY_YES, TradeAction.BUY_YES, TradeAction.LEAN_YES
        ]]
        buy_no = [r for r in results if r.recommendation.action in [
            TradeAction.STRONG_BUY_NO, TradeAction.BUY_NO, TradeAction.LEAN_NO
        ]]
        holds = [r for r in results if r.recommendation.action == TradeAction.HOLD]

        print(f"\nTotal markets analyzed: {len(results)}")
        print(f"  Buy YES recommendations: {len(buy_yes)}")
        print(f"  Buy NO recommendations:  {len(buy_no)}")
        print(f"  Hold recommendations:    {len(holds)}")

        if buy_yes or buy_no:
            print("\nTop opportunities by edge:")
            all_trades = buy_yes + buy_no
            all_trades.sort(key=lambda r: abs(r.recommendation.edge), reverse=True)
            for r in all_trades[:5]:
                print(f"  {r.recommendation.action.value}: {r.market_question[:40]}... "
                      f"(edge: {r.recommendation.edge*100:+.1f}%)")


# Demo markets for testing without API access
DEMO_MARKETS = {
    "demo1": {
        "id": "demo1",
        "question": "Will the Federal Reserve cut interest rates in Q1 2026?",
        "description": "This market resolves YES if the Federal Reserve announces a cut to the federal funds rate at any FOMC meeting in Q1 2026 (January, March meetings).",
        "tokens": [
            {"outcome": "Yes", "price": 0.35},
            {"outcome": "No", "price": 0.65},
        ],
        "volume": 1250000,
        "liquidity": 450000,
        "endDate": "2026-03-31",
        "url": "https://polymarket.com/event/fed-rate-cut-q1-2026",
    },
    "demo2": {
        "id": "demo2",
        "question": "Will Bitcoin reach $150,000 before July 2026?",
        "description": "This market resolves YES if Bitcoin (BTC) reaches a price of $150,000 USD on any major exchange before July 1, 2026.",
        "tokens": [
            {"outcome": "Yes", "price": 0.28},
            {"outcome": "No", "price": 0.72},
        ],
        "volume": 3500000,
        "liquidity": 890000,
        "endDate": "2026-07-01",
        "url": "https://polymarket.com/event/btc-150k-h1-2026",
    },
    "demo3": {
        "id": "demo3",
        "question": "Will OpenAI release GPT-5 in 2026?",
        "description": "This market resolves YES if OpenAI publicly releases or announces general availability of a model officially named 'GPT-5' during 2026.",
        "tokens": [
            {"outcome": "Yes", "price": 0.62},
            {"outcome": "No", "price": 0.38},
        ],
        "volume": 890000,
        "liquidity": 234000,
        "endDate": "2026-12-31",
        "url": "https://polymarket.com/event/gpt5-2026",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Polymarket markets using Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a market by URL
  python -m polymarket_backtester.researcher.cli "https://polymarket.com/event/..."

  # Analyze by market ID
  python -m polymarket_backtester.researcher.cli abc123

  # Output as JSON
  python -m polymarket_backtester.researcher.cli --format json "https://polymarket.com/event/..."

  # Analyze multiple markets
  python -m polymarket_backtester.researcher.cli market1 market2 market3

  # Use a specific model
  python -m polymarket_backtester.researcher.cli --model claude-opus-4-20250514 "https://polymarket.com/event/..."
"""
    )

    parser.add_argument(
        "markets",
        nargs="*",
        default=[],
        help="Polymarket URLs or market IDs to analyze"
    )

    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "simple"],
        default="text",
        help="Output format (default: text)"
    )

    parser.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)"
    )

    parser.add_argument(
        "--context", "-c",
        default="",
        help="Additional context to provide to the researcher"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo markets instead of fetching from Polymarket API"
    )

    parser.add_argument(
        "--list-demos",
        action="store_true",
        help="List available demo markets"
    )

    args = parser.parse_args()

    if args.list_demos:
        print("\nAvailable demo markets:")
        print("-" * 60)
        for key, market in DEMO_MARKETS.items():
            yes_price = 0.5
            for t in market.get("tokens", []):
                if "yes" in t.get("outcome", "").lower():
                    yes_price = t.get("price", 0.5)
            print(f"\n  {key}:")
            print(f"    {market['question']}")
            print(f"    YES: {yes_price*100:.0f}%  |  Volume: ${market['volume']:,}")
        print("\nUsage: python -m polymarket_backtester.researcher --demo demo1")
        print("       python -m polymarket_backtester.researcher --demo all")
        print()
        return

    if not args.markets:
        parser.error("Please provide market IDs or URLs, or use --demo with demo market IDs")

    if args.demo:
        # Use demo markets
        markets_to_analyze = []
        for m in args.markets:
            if m in DEMO_MARKETS:
                markets_to_analyze.append(DEMO_MARKETS[m])
            elif m == "all":
                markets_to_analyze.extend(DEMO_MARKETS.values())
            else:
                print(f"Unknown demo market: {m}. Use --list-demos to see available options.")
                sys.exit(1)

        asyncio.run(analyze_demo_markets(
            markets_to_analyze,
            output_format=args.format,
            model=args.model,
        ))
    elif len(args.markets) == 1:
        asyncio.run(analyze_single_market(
            args.markets[0],
            output_format=args.format,
            model=args.model,
        ))
    else:
        asyncio.run(analyze_multiple_markets(
            args.markets,
            output_format=args.format,
            model=args.model,
        ))


if __name__ == "__main__":
    main()
