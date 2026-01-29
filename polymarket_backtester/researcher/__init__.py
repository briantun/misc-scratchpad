"""
Claude-as-Researcher module for Polymarket analysis.

This module provides an alternative to quantitative backtesting by using
Claude to perform qualitative research and analysis on prediction markets.
"""

from .market_researcher import MarketResearcher, ResearchResult, TradeRecommendation

__all__ = ["MarketResearcher", "ResearchResult", "TradeRecommendation"]
