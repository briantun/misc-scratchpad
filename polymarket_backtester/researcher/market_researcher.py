"""
Market Researcher - Claude-powered analysis for Polymarket.

This module uses Claude to perform qualitative research and analysis,
providing trade recommendations with reasoning.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import httpx


class TradeAction(Enum):
    """Recommended trade action."""
    STRONG_BUY_YES = "strong_buy_yes"
    BUY_YES = "buy_yes"
    LEAN_YES = "lean_yes"
    HOLD = "hold"
    LEAN_NO = "lean_no"
    BUY_NO = "buy_no"
    STRONG_BUY_NO = "strong_buy_no"


@dataclass
class TradeRecommendation:
    """A trade recommendation from the researcher."""
    action: TradeAction
    confidence: float  # 0-1
    fair_probability: float  # Researcher's estimate of true probability
    market_probability: float  # Current market price
    edge: float  # fair_probability - market_probability (or inverse for NO)
    position_size_pct: float  # Suggested position size as % of bankroll
    reasoning: str
    key_factors: list[str]
    risks: list[str]

    @property
    def expected_value(self) -> float:
        """Calculate expected value per dollar risked."""
        if self.action in [TradeAction.STRONG_BUY_YES, TradeAction.BUY_YES, TradeAction.LEAN_YES]:
            # Betting YES: win (1-price)/price if YES, lose 1 if NO
            win_payout = (1 - self.market_probability) / self.market_probability
            return self.fair_probability * win_payout - (1 - self.fair_probability)
        elif self.action in [TradeAction.STRONG_BUY_NO, TradeAction.BUY_NO, TradeAction.LEAN_NO]:
            # Betting NO: win price/(1-price) if NO, lose 1 if YES
            win_payout = self.market_probability / (1 - self.market_probability)
            return (1 - self.fair_probability) * win_payout - self.fair_probability
        return 0.0


@dataclass
class ResearchResult:
    """Complete research result for a market."""
    market_id: str
    market_question: str
    market_url: str
    current_yes_price: float
    current_no_price: float
    volume_24h: float
    liquidity: float
    resolution_date: Optional[datetime]

    # Research outputs
    recommendation: TradeRecommendation
    research_summary: str
    sources_consulted: list[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "market_id": self.market_id,
            "market_question": self.market_question,
            "market_url": self.market_url,
            "current_yes_price": self.current_yes_price,
            "current_no_price": self.current_no_price,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "resolution_date": self.resolution_date.isoformat() if self.resolution_date else None,
            "recommendation": {
                "action": self.recommendation.action.value,
                "confidence": self.recommendation.confidence,
                "fair_probability": self.recommendation.fair_probability,
                "market_probability": self.recommendation.market_probability,
                "edge": self.recommendation.edge,
                "position_size_pct": self.recommendation.position_size_pct,
                "expected_value": self.recommendation.expected_value,
                "reasoning": self.recommendation.reasoning,
                "key_factors": self.recommendation.key_factors,
                "risks": self.recommendation.risks,
            },
            "research_summary": self.research_summary,
            "sources_consulted": self.sources_consulted,
            "timestamp": self.timestamp.isoformat(),
        }


class MarketResearcher:
    """
    Claude-powered market researcher for Polymarket.

    Uses web search and Claude's reasoning to analyze markets
    and provide trade recommendations.
    """

    POLYMARKET_API_BASE = "https://gamma-api.polymarket.com"
    CLOB_API_BASE = "https://clob.polymarket.com"

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_searches: int = 5,
    ):
        """
        Initialize the researcher.

        Args:
            anthropic_api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: Claude model to use
            max_searches: Maximum number of web searches to perform
        """
        self.anthropic_api_key = anthropic_api_key
        self.model = model
        self.max_searches = max_searches
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _extract_market_id(self, url_or_id: str) -> str:
        """Extract market ID from URL or return ID directly."""
        # Handle full URLs
        if "polymarket.com" in url_or_id:
            # URL format: https://polymarket.com/event/xxx/market-slug
            # or: https://polymarket.com/markets/xxx
            patterns = [
                r"polymarket\.com/event/([^/\?]+)",
                r"polymarket\.com/markets/([^/\?]+)",
                r"condition_id=([^&]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, url_or_id)
                if match:
                    return match.group(1)

        # Assume it's already an ID
        return url_or_id

    async def fetch_market_data(self, market_id: str) -> dict:
        """Fetch current market data from Polymarket."""
        client = await self._get_client()

        # Try Gamma API first (more detailed)
        try:
            response = await client.get(
                f"{self.POLYMARKET_API_BASE}/markets/{market_id}"
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass

        # Try CLOB API
        try:
            response = await client.get(
                f"{self.CLOB_API_BASE}/markets/{market_id}"
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass

        # Try searching by slug
        try:
            response = await client.get(
                f"{self.POLYMARKET_API_BASE}/markets",
                params={"slug": market_id}
            )
            if response.status_code == 200:
                markets = response.json()
                if markets:
                    return markets[0]
        except Exception:
            pass

        raise ValueError(f"Could not fetch market data for: {market_id}")

    async def search_web(self, query: str) -> list[dict]:
        """
        Perform web search for market research.

        Returns list of search results with title, url, snippet.
        """
        # This is a placeholder - in production you'd use a real search API
        # For now, we'll return empty and rely on Claude's knowledge +
        # any URLs provided
        return []

    async def fetch_url_content(self, url: str) -> str:
        """Fetch and extract text content from a URL."""
        client = await self._get_client()
        try:
            response = await client.get(url, follow_redirects=True)
            if response.status_code == 200:
                # Basic text extraction - in production use proper HTML parsing
                text = response.text
                # Remove script/style tags
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', text)
                # Clean whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:10000]  # Limit length
        except Exception as e:
            return f"Error fetching URL: {e}"
        return ""

    def _build_research_prompt(
        self,
        market_data: dict,
        additional_context: str = "",
    ) -> str:
        """Build the research prompt for Claude."""

        question = market_data.get("question") or market_data.get("title", "Unknown")
        description = market_data.get("description", "")

        # Extract prices
        yes_price = None
        no_price = None

        # Try different formats
        if "outcomePrices" in market_data:
            try:
                prices = json.loads(market_data["outcomePrices"])
                yes_price = float(prices[0])
                no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price
            except (json.JSONDecodeError, IndexError, TypeError):
                pass

        if yes_price is None and "tokens" in market_data:
            for token in market_data.get("tokens", []):
                outcome = token.get("outcome", "").lower()
                price = token.get("price", 0)
                if "yes" in outcome:
                    yes_price = float(price)
                elif "no" in outcome:
                    no_price = float(price)

        if yes_price is None:
            yes_price = 0.5
            no_price = 0.5

        volume = market_data.get("volume", market_data.get("volumeNum", 0))
        liquidity = market_data.get("liquidity", 0)
        end_date = market_data.get("endDate") or market_data.get("end_date", "Unknown")

        prompt = f"""You are a professional prediction market analyst. Your task is to research and analyze this Polymarket market and provide a trade recommendation.

## MARKET DETAILS
**Question:** {question}

**Description:** {description}

**Current Prices:**
- YES: ${yes_price:.2f} ({yes_price*100:.1f}% implied probability)
- NO: ${no_price:.2f} ({no_price*100:.1f}% implied probability)

**Volume:** ${volume:,.0f}
**Liquidity:** ${liquidity:,.0f}
**Resolution Date:** {end_date}

{f"## ADDITIONAL CONTEXT{chr(10)}{additional_context}" if additional_context else ""}

## YOUR TASK

1. **Research Phase**: Think through what information would be most relevant to assess this market. Consider:
   - Recent news and developments
   - Historical base rates for similar events
   - Key stakeholders and their incentives
   - Potential catalysts before resolution
   - What information the market might be missing

2. **Analysis Phase**: Based on your research:
   - Estimate the TRUE probability of YES outcome
   - Compare to the current market price
   - Identify the edge (if any)
   - Assess your confidence level

3. **Recommendation Phase**: Provide your recommendation in this exact JSON format:

```json
{{
    "fair_probability": 0.XX,
    "confidence": 0.XX,
    "action": "ACTION",
    "position_size_pct": X.X,
    "reasoning": "Your detailed reasoning here",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risks": ["risk1", "risk2", "risk3"],
    "research_summary": "Brief summary of key findings"
}}
```

Where:
- `fair_probability`: Your estimate of true YES probability (0.0 to 1.0)
- `confidence`: How confident you are in your estimate (0.0 to 1.0)
- `action`: One of: "strong_buy_yes", "buy_yes", "lean_yes", "hold", "lean_no", "buy_no", "strong_buy_no"
- `position_size_pct`: Suggested position size as % of bankroll (0.0 to 5.0)
- `reasoning`: Detailed explanation of your analysis
- `key_factors`: Top 3-5 factors driving your assessment
- `risks`: Top 3-5 risks to your thesis

## GUIDELINES

- Be intellectually honest. If you're uncertain, say so.
- Don't anchor too heavily on the current market price.
- Consider what information sophisticated traders might have.
- A "hold" recommendation is valid if you see no clear edge.
- Be specific about what would change your view.
- Position sizing should reflect confidence (lower confidence = smaller position).

Now provide your analysis and recommendation:"""

        return prompt, yes_price, no_price, volume, liquidity

    def _parse_recommendation(
        self,
        response_text: str,
        market_probability: float,
    ) -> tuple[TradeRecommendation, str]:
        """Parse Claude's response into a TradeRecommendation."""

        # Extract JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if not json_match:
            # Try finding raw JSON
            json_match = re.search(r'\{[^{}]*"fair_probability"[^{}]*\}', response_text, re.DOTALL)

        if not json_match:
            raise ValueError("Could not parse recommendation JSON from response")

        try:
            data = json.loads(json_match.group(1) if '```' in response_text else json_match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

        fair_prob = float(data.get("fair_probability", 0.5))
        confidence = float(data.get("confidence", 0.5))
        action_str = data.get("action", "hold").lower()
        position_size = float(data.get("position_size_pct", 0.0))

        # Map action string to enum
        action_map = {
            "strong_buy_yes": TradeAction.STRONG_BUY_YES,
            "buy_yes": TradeAction.BUY_YES,
            "lean_yes": TradeAction.LEAN_YES,
            "hold": TradeAction.HOLD,
            "lean_no": TradeAction.LEAN_NO,
            "buy_no": TradeAction.BUY_NO,
            "strong_buy_no": TradeAction.STRONG_BUY_NO,
        }
        action = action_map.get(action_str, TradeAction.HOLD)

        # Calculate edge
        if action in [TradeAction.STRONG_BUY_YES, TradeAction.BUY_YES, TradeAction.LEAN_YES]:
            edge = fair_prob - market_probability
        elif action in [TradeAction.STRONG_BUY_NO, TradeAction.BUY_NO, TradeAction.LEAN_NO]:
            edge = (1 - fair_prob) - (1 - market_probability)
        else:
            edge = 0.0

        recommendation = TradeRecommendation(
            action=action,
            confidence=confidence,
            fair_probability=fair_prob,
            market_probability=market_probability,
            edge=edge,
            position_size_pct=position_size,
            reasoning=data.get("reasoning", ""),
            key_factors=data.get("key_factors", []),
            risks=data.get("risks", []),
        )

        research_summary = data.get("research_summary", "")

        return recommendation, research_summary

    async def analyze_from_data(
        self,
        market_data: dict,
        additional_context: str = "",
    ) -> ResearchResult:
        """
        Analyze a market from provided data (no API fetch needed).

        Args:
            market_data: Dict with keys: question, yes_price, no_price, volume, liquidity, end_date
            additional_context: Any additional context

        Returns:
            ResearchResult with recommendation and analysis
        """
        import anthropic
        import os

        # Build prompt
        prompt, yes_price, no_price, volume, liquidity = self._build_research_prompt(
            market_data, additional_context
        )

        # Call Claude
        api_key = self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text

        # Parse recommendation
        recommendation, research_summary = self._parse_recommendation(
            response_text, yes_price
        )

        # Extract resolution date
        resolution_date = None
        end_date_str = market_data.get("endDate") or market_data.get("end_date")
        if end_date_str:
            try:
                if isinstance(end_date_str, str):
                    resolution_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        question = market_data.get("question") or market_data.get("title", "")
        market_id = market_data.get("id", "manual")

        return ResearchResult(
            market_id=market_id,
            market_question=question,
            market_url=market_data.get("url", f"https://polymarket.com/event/{market_id}"),
            current_yes_price=yes_price,
            current_no_price=no_price,
            volume_24h=volume,
            liquidity=liquidity,
            resolution_date=resolution_date,
            recommendation=recommendation,
            research_summary=research_summary,
            sources_consulted=[],
        )

    async def analyze_market(
        self,
        market_url_or_id: str,
        additional_context: str = "",
    ) -> ResearchResult:
        """
        Analyze a Polymarket market and provide a trade recommendation.

        Args:
            market_url_or_id: Polymarket URL or market ID
            additional_context: Any additional context to provide to the researcher

        Returns:
            ResearchResult with recommendation and analysis
        """
        import anthropic
        import os

        # Extract market ID and fetch data
        market_id = self._extract_market_id(market_url_or_id)
        market_data = await self.fetch_market_data(market_id)

        # Build prompt
        prompt, yes_price, no_price, volume, liquidity = self._build_research_prompt(
            market_data, additional_context
        )

        # Call Claude
        api_key = self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text

        # Parse recommendation
        recommendation, research_summary = self._parse_recommendation(
            response_text, yes_price
        )

        # Extract resolution date
        resolution_date = None
        end_date_str = market_data.get("endDate") or market_data.get("end_date")
        if end_date_str:
            try:
                resolution_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        # Build market URL
        question = market_data.get("question") or market_data.get("title", "")
        slug = market_data.get("slug", market_id)
        market_url = f"https://polymarket.com/event/{slug}"

        return ResearchResult(
            market_id=market_id,
            market_question=question,
            market_url=market_url,
            current_yes_price=yes_price,
            current_no_price=no_price,
            volume_24h=volume,
            liquidity=liquidity,
            resolution_date=resolution_date,
            recommendation=recommendation,
            research_summary=research_summary,
            sources_consulted=[],  # Would be populated with actual search results
        )

    async def analyze_market_with_search(
        self,
        market_url_or_id: str,
        search_queries: Optional[list[str]] = None,
    ) -> ResearchResult:
        """
        Analyze a market with web search for additional context.

        This version performs web searches to gather current information
        before analysis.
        """
        # Fetch market data first
        market_id = self._extract_market_id(market_url_or_id)
        market_data = await self.fetch_market_data(market_id)

        question = market_data.get("question") or market_data.get("title", "")

        # Generate search queries if not provided
        if not search_queries:
            search_queries = [
                question,
                f"{question} latest news",
                f"{question} prediction",
            ]

        # Perform searches and collect context
        context_parts = []
        sources = []

        for query in search_queries[:self.max_searches]:
            results = await self.search_web(query)
            for result in results[:3]:
                sources.append(result.get("url", ""))
                context_parts.append(
                    f"**{result.get('title', 'Unknown')}**\n{result.get('snippet', '')}"
                )

        additional_context = "\n\n".join(context_parts) if context_parts else ""

        # Run analysis with context
        result = await self.analyze_market(market_url_or_id, additional_context)
        result.sources_consulted = [s for s in sources if s]

        return result
