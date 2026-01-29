"""
LLM Trade Auditor

Uses Claude to audit/validate trades before execution, providing:
1. Sanity checks on trade logic
2. Risk assessment
3. Qualitative context analysis
4. Anomaly detection
5. Trade reasoning documentation

This adds a layer of AI judgment on top of algorithmic signals.
"""
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from ..models import (
    MarketSnapshot, Order, PortfolioState, Side, StrategySignal
)


class AuditDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"  # Approve with modifications (e.g., reduce size)
    DEFER = "defer"    # Need more information


@dataclass
class AuditResult:
    """Result of an LLM audit on a proposed trade."""
    decision: AuditDecision
    confidence: float  # 0-1, how confident is the auditor
    reasoning: str
    risk_assessment: str
    suggested_modifications: Optional[dict] = None
    flags: list[str] = None  # Warning flags raised
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.flags is None:
            self.flags = []


class LLMAuditorBase(ABC):
    """
    Base class for LLM-based trade auditors.

    Subclass this and implement _call_llm() to connect to your preferred
    LLM provider (Anthropic, OpenAI, local model, etc.)
    """

    def __init__(
        self,
        max_position_risk: float = 0.15,
        require_reasoning: bool = True,
        confidence_threshold: float = 0.6,
        enable_modifications: bool = True
    ):
        """
        Initialize the auditor.

        Args:
            max_position_risk: Maximum allowed position as fraction of portfolio
            require_reasoning: Whether to require explanation for approvals
            confidence_threshold: Minimum confidence to approve without flags
            enable_modifications: Whether auditor can suggest trade modifications
        """
        self.max_position_risk = max_position_risk
        self.require_reasoning = require_reasoning
        self.confidence_threshold = confidence_threshold
        self.enable_modifications = enable_modifications

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with a prompt and return the response.

        Override this method to implement your LLM provider.
        """
        pass

    def _build_audit_prompt(
        self,
        signal: StrategySignal,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
        recent_signals: list[StrategySignal]
    ) -> str:
        """Build the prompt for trade auditing."""

        # Format market context
        market_context = f"""
MARKET INFORMATION:
- Question: {snapshot.market.question}
- Category: {snapshot.market.category}
- Current YES price: {snapshot.price.mid_yes:.3f}
- Current NO price: {snapshot.price.mid_no:.3f}
- Bid-Ask spread (YES): {snapshot.price.spread_yes:.3f}
- Resolution date: {snapshot.market.resolution_date}
- Market status: {snapshot.market.status.value}
"""

        # Format external signals if available
        if snapshot.external_signals:
            signals_text = "\nEXTERNAL SIGNALS:\n"
            for name, value in snapshot.external_signals.items():
                if isinstance(value, dict):
                    signals_text += f"- {name}: {value.get('value', value)}\n"
                else:
                    signals_text += f"- {name}: {value}\n"
        else:
            signals_text = "\nNo external signals available.\n"

        # Format portfolio state
        portfolio_context = f"""
PORTFOLIO STATE:
- Total value: ${portfolio.total_value:,.2f}
- Cash available: ${portfolio.cash:,.2f}
- Number of positions: {len(portfolio.positions)}
- Unrealized P&L: ${portfolio.unrealized_pnl:,.2f}
- Realized P&L: ${portfolio.realized_pnl:,.2f}
"""

        # Check if we have existing position in this market
        existing_position = portfolio.positions.get(snapshot.market.id)
        if existing_position:
            portfolio_context += f"""
EXISTING POSITION IN THIS MARKET:
- Side: {existing_position.side.value}
- Quantity: {existing_position.quantity:.2f}
- Average entry: {existing_position.average_entry_price:.3f}
- Unrealized P&L: ${existing_position.unrealized_pnl:.2f}
"""

        # Format proposed trade
        trade_context = f"""
PROPOSED TRADE:
- Strategy: {signal.strategy_name}
- Side: {signal.side.value}
- Signal strength: {signal.strength:.3f}
- Suggested size: {signal.suggested_size:.1%} of portfolio
- Estimated cost: ${portfolio.total_value * signal.suggested_size:,.2f}
"""

        if signal.metadata:
            trade_context += f"- Strategy metadata: {json.dumps(signal.metadata, default=str)}\n"

        # Format recent activity
        recent_context = ""
        if recent_signals:
            recent_context = f"\nRECENT SIGNALS ({len(recent_signals)} in last period):\n"
            for s in recent_signals[-5:]:  # Last 5 signals
                recent_context += f"- {s.strategy_name}: {s.side.value} (strength: {s.strength:.3f})\n"

        # Build full prompt
        prompt = f"""You are a trade auditor for a prediction market trading system. Your job is to review proposed trades and either approve, reject, or suggest modifications.

{market_context}
{signals_text}
{portfolio_context}
{trade_context}
{recent_context}

AUDIT CRITERIA:
1. Does the trade make logical sense given the market question and signals?
2. Is the position size appropriate for the signal strength and portfolio?
3. Are there any red flags (unusual prices, timing near resolution, contradictory signals)?
4. Does this trade align with good risk management practices?
5. Are there any qualitative factors that the algorithm might have missed?

Please provide your audit decision in the following JSON format:
{{
    "decision": "approve" | "reject" | "modify",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your decision",
    "risk_assessment": "Low/Medium/High with explanation",
    "flags": ["list", "of", "warning", "flags"],
    "suggested_modifications": {{
        "new_size": 0.05,  // if recommending size change
        "reason": "explanation"
    }}
}}

Respond ONLY with the JSON object, no other text."""

        return prompt

    def _parse_audit_response(self, response: str) -> AuditResult:
        """Parse LLM response into AuditResult."""
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)

            decision_map = {
                "approve": AuditDecision.APPROVE,
                "reject": AuditDecision.REJECT,
                "modify": AuditDecision.MODIFY,
                "defer": AuditDecision.DEFER
            }

            return AuditResult(
                decision=decision_map.get(data.get("decision", "").lower(), AuditDecision.DEFER),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "No reasoning provided"),
                risk_assessment=data.get("risk_assessment", "Unknown"),
                suggested_modifications=data.get("suggested_modifications"),
                flags=data.get("flags", [])
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If parsing fails, default to defer
            return AuditResult(
                decision=AuditDecision.DEFER,
                confidence=0.0,
                reasoning=f"Failed to parse LLM response: {e}",
                risk_assessment="Unknown",
                flags=["parse_error"]
            )

    def audit_trade(
        self,
        signal: StrategySignal,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
        recent_signals: Optional[list[StrategySignal]] = None
    ) -> AuditResult:
        """
        Audit a proposed trade before execution.

        Args:
            signal: The trading signal to audit
            snapshot: Current market state
            portfolio: Current portfolio state
            recent_signals: Recent signals for context

        Returns:
            AuditResult with decision and reasoning
        """
        recent_signals = recent_signals or []

        # Build and send prompt
        prompt = self._build_audit_prompt(signal, snapshot, portfolio, recent_signals)
        response = self._call_llm(prompt)

        # Parse response
        result = self._parse_audit_response(response)

        # Apply additional rule-based checks
        result = self._apply_rule_checks(result, signal, snapshot, portfolio)

        return result

    def _apply_rule_checks(
        self,
        result: AuditResult,
        signal: StrategySignal,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState
    ) -> AuditResult:
        """Apply additional rule-based safety checks."""

        flags = list(result.flags)

        # Check position size limits
        proposed_size = signal.suggested_size
        if proposed_size > self.max_position_risk:
            flags.append(f"position_size_exceeds_limit ({proposed_size:.1%} > {self.max_position_risk:.1%})")
            if result.decision == AuditDecision.APPROVE:
                result.decision = AuditDecision.MODIFY
                result.suggested_modifications = result.suggested_modifications or {}
                result.suggested_modifications["new_size"] = self.max_position_risk
                result.suggested_modifications["reason"] = "Reduced to max position limit"

        # Check for near-resolution trades
        if snapshot.market.resolution_date:
            days_to_resolution = (snapshot.market.resolution_date - datetime.now()).days
            if days_to_resolution < 1:
                flags.append("very_close_to_resolution")

        # Check for extreme prices
        if snapshot.price.mid_yes < 0.02 or snapshot.price.mid_yes > 0.98:
            flags.append("extreme_price_level")

        # Check for wide spreads
        if snapshot.price.spread_yes > 0.10:
            flags.append("wide_spread_warning")

        # Check for conflicting position
        existing = portfolio.positions.get(snapshot.market.id)
        if existing and existing.side != signal.side:
            flags.append("trade_conflicts_with_existing_position")

        # Check available cash
        trade_cost = portfolio.total_value * signal.suggested_size
        if trade_cost > portfolio.cash:
            flags.append("insufficient_cash")
            if result.decision == AuditDecision.APPROVE:
                result.decision = AuditDecision.MODIFY
                max_size = portfolio.cash / portfolio.total_value * 0.95  # 95% of available
                result.suggested_modifications = result.suggested_modifications or {}
                result.suggested_modifications["new_size"] = max_size
                result.suggested_modifications["reason"] = "Reduced due to insufficient cash"

        result.flags = flags
        return result


class MockLLMAuditor(LLMAuditorBase):
    """
    Mock auditor for testing/backtesting that simulates LLM responses.

    Uses rule-based logic to approximate LLM judgment without API calls.
    """

    def __init__(self, approval_rate: float = 0.8, **kwargs):
        """
        Args:
            approval_rate: Base probability of approving trades
        """
        super().__init__(**kwargs)
        self.approval_rate = approval_rate

    def _call_llm(self, prompt: str) -> str:
        """Simulate LLM response with rule-based logic."""
        import random

        # Parse key information from prompt
        signal_strength = 0.5
        if "Signal strength:" in prompt:
            try:
                strength_line = [l for l in prompt.split("\n") if "Signal strength:" in l][0]
                signal_strength = float(strength_line.split(":")[-1].strip())
            except (IndexError, ValueError):
                pass

        # Higher signal strength = higher approval rate
        adjusted_rate = self.approval_rate * (0.5 + signal_strength)

        # Determine decision
        if random.random() < adjusted_rate:
            decision = "approve"
            confidence = 0.6 + random.random() * 0.3
        else:
            if random.random() < 0.7:
                decision = "modify"
                confidence = 0.5 + random.random() * 0.3
            else:
                decision = "reject"
                confidence = 0.4 + random.random() * 0.4

        # Generate mock response
        response = {
            "decision": decision,
            "confidence": round(confidence, 2),
            "reasoning": f"Mock audit: Signal strength {signal_strength:.2f} evaluated",
            "risk_assessment": "Medium" if signal_strength < 0.7 else "Low",
            "flags": [],
            "suggested_modifications": None
        }

        if decision == "modify":
            response["suggested_modifications"] = {
                "new_size": round(random.uniform(0.03, 0.08), 3),
                "reason": "Adjusted position size for risk management"
            }

        return json.dumps(response)


class AnthropicAuditor(LLMAuditorBase):
    """
    Production auditor using Anthropic's Claude API.

    Requires: pip install anthropic
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        **kwargs
    ):
        """
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model to use for auditing
            max_tokens: Maximum response tokens
        """
        super().__init__(**kwargs)
        self.model = model
        self.max_tokens = max_tokens

        # Import here to make it optional
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    def _call_llm(self, prompt: str) -> str:
        """Call Claude API."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text


class AuditLogger:
    """Logs all audit decisions for analysis and review."""

    def __init__(self):
        self.logs: list[dict] = []

    def log(
        self,
        signal: StrategySignal,
        result: AuditResult,
        final_action: str  # "executed", "modified", "skipped"
    ):
        """Log an audit decision."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "market_id": signal.market_id,
            "strategy": signal.strategy_name,
            "signal_side": signal.side.value,
            "signal_strength": signal.strength,
            "audit_decision": result.decision.value,
            "audit_confidence": result.confidence,
            "audit_reasoning": result.reasoning,
            "risk_assessment": result.risk_assessment,
            "flags": result.flags,
            "final_action": final_action
        })

    def get_summary(self) -> dict:
        """Get summary statistics of audit decisions."""
        if not self.logs:
            return {}

        decisions = [l["audit_decision"] for l in self.logs]
        actions = [l["final_action"] for l in self.logs]

        return {
            "total_audits": len(self.logs),
            "approvals": decisions.count("approve"),
            "rejections": decisions.count("reject"),
            "modifications": decisions.count("modify"),
            "executed": actions.count("executed"),
            "modified": actions.count("modified"),
            "skipped": actions.count("skipped"),
            "approval_rate": decisions.count("approve") / len(decisions),
            "execution_rate": actions.count("executed") / len(actions)
        }

    def export(self, filepath: str):
        """Export logs to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.logs, f, indent=2)
