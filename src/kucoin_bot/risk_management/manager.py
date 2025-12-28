"""Risk management system for the trading bot."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog

from kucoin_bot.config import RiskSettings, TradingSettings
from kucoin_bot.models.data_models import (
    Order,
    OrderSide,
    Position,
    TradingSignal,
)

logger = structlog.get_logger()


@dataclass
class RiskMetrics:
    """Current risk metrics for the portfolio."""

    total_exposure: Decimal = Decimal("0")
    max_exposure: Decimal = Decimal("0")
    exposure_ratio: Decimal = Decimal("0")
    open_positions: int = 0
    max_positions: int = 0
    daily_trades: int = 0
    max_daily_trades: int = 0
    current_drawdown: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    portfolio_value: Decimal = Decimal("0")
    peak_value: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    is_trading_allowed: bool = True
    blocked_reasons: list[str] = field(default_factory=list)


@dataclass
class RiskCheck:
    """Result of a risk check."""

    passed: bool
    reason: str
    severity: str = "info"  # info, warning, critical


class RiskManager:
    """Manages risk for the trading bot."""

    def __init__(
        self,
        risk_settings: RiskSettings,
        trading_settings: TradingSettings,
    ) -> None:
        """Initialize risk manager.

        Args:
            risk_settings: Risk configuration
            trading_settings: Trading configuration
        """
        self.risk_settings = risk_settings
        self.trading_settings = trading_settings
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self.daily_trades: int = 0
        self.last_trade_reset: datetime = datetime.now(UTC)
        self.portfolio_value: Decimal = Decimal("0")
        self.peak_value: Decimal = Decimal("0")
        self.daily_pnl: Decimal = Decimal("0")
        self.trade_history: list[dict[str, Any]] = []
        self.logger = logger.bind(component="risk_manager")

    def load_positions(self, positions: dict[str, Position]) -> None:
        """Load positions from persistent state.

        Args:
            positions: Dictionary of positions to load
        """
        self.positions = positions
        self.logger.info("Positions loaded from state", count=len(positions))

    def set_portfolio_value(self, value: Decimal) -> None:
        """Update portfolio value and track peak.

        Args:
            value: Current portfolio value
        """
        self.portfolio_value = value
        if value > self.peak_value:
            self.peak_value = value

    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics.

        Returns:
            Current risk metrics
        """
        self._reset_daily_trades_if_needed()

        total_exposure = sum(
            (p.size * p.current_price for p in self.positions.values()),
            Decimal("0"),
        )
        max_exposure = Decimal(str(self.trading_settings.max_position_size))

        exposure_ratio = Decimal("0")
        if self.portfolio_value > 0:
            exposure_ratio = (total_exposure / self.portfolio_value) * 100

        current_drawdown = Decimal("0")
        if self.peak_value > 0:
            current_drawdown = ((self.peak_value - self.portfolio_value) / self.peak_value) * 100

        is_trading_allowed = True
        blocked_reasons: list[str] = []

        # Check drawdown limit
        if current_drawdown >= Decimal(str(self.risk_settings.max_drawdown_percent)):
            is_trading_allowed = False
            blocked_reasons.append(f"Max drawdown exceeded: {current_drawdown:.2f}%")

        # Check daily trade limit
        if self.daily_trades >= self.trading_settings.max_daily_trades:
            is_trading_allowed = False
            blocked_reasons.append(f"Max daily trades reached: {self.daily_trades}")

        # Check position limit
        if len(self.positions) >= self.risk_settings.max_open_positions:
            blocked_reasons.append(f"Max positions reached: {len(self.positions)}")

        return RiskMetrics(
            total_exposure=total_exposure,
            max_exposure=max_exposure,
            exposure_ratio=exposure_ratio,
            open_positions=len(self.positions),
            max_positions=self.risk_settings.max_open_positions,
            daily_trades=self.daily_trades,
            max_daily_trades=self.trading_settings.max_daily_trades,
            current_drawdown=current_drawdown,
            max_drawdown=Decimal(str(self.risk_settings.max_drawdown_percent)),
            portfolio_value=self.portfolio_value,
            peak_value=self.peak_value,
            daily_pnl=self.daily_pnl,
            is_trading_allowed=is_trading_allowed,
            blocked_reasons=blocked_reasons,
        )

    def _reset_daily_trades_if_needed(self) -> None:
        """Reset daily trade counter if a new day has started."""
        now = datetime.now(UTC)
        if now.date() > self.last_trade_reset.date():
            self.daily_trades = 0
            self.daily_pnl = Decimal("0")
            self.last_trade_reset = now
            self.logger.info("Daily trade counter reset")

    def check_signal(self, signal: TradingSignal) -> list[RiskCheck]:
        """Perform risk checks on a trading signal.

        Args:
            signal: Trading signal to check

        Returns:
            List of risk check results
        """
        checks: list[RiskCheck] = []
        metrics = self.get_metrics()

        # Check if trading is allowed
        if not metrics.is_trading_allowed:
            for reason in metrics.blocked_reasons:
                checks.append(RiskCheck(passed=False, reason=reason, severity="critical"))
            return checks

        # Check confidence threshold
        if signal.confidence < 0.5:
            checks.append(RiskCheck(
                passed=False,
                reason=f"Signal confidence too low: {signal.confidence:.2f}",
                severity="warning",
            ))

        # Check if we already have a position in this symbol
        if signal.symbol in self.positions:
            existing_position = self.positions[signal.symbol]
            if signal.signal_type.value == existing_position.side.value:
                checks.append(RiskCheck(
                    passed=False,
                    reason=f"Already have {existing_position.side.value} position in {signal.symbol}",
                    severity="warning",
                ))

        # Check daily trade limit
        if self.daily_trades >= self.trading_settings.max_daily_trades:
            checks.append(RiskCheck(
                passed=False,
                reason="Daily trade limit reached",
                severity="critical",
            ))

        # Check position limit for new positions
        from kucoin_bot.models.data_models import SignalType
        if (
            signal.signal_type == SignalType.BUY
            and signal.symbol not in self.positions
            and len(self.positions) >= self.risk_settings.max_open_positions
        ):
            checks.append(RiskCheck(
                passed=False,
                reason="Maximum open positions reached",
                severity="critical",
            ))

        # All checks passed if no failures
        if not checks:
            checks.append(RiskCheck(
                passed=True,
                reason="All risk checks passed",
                severity="info",
            ))

        return checks

    def calculate_position_size(
        self,
        signal: TradingSignal,
        available_balance: Decimal,
    ) -> Decimal:
        """Calculate appropriate position size based on risk parameters.

        Args:
            signal: Trading signal
            available_balance: Available balance for trading

        Returns:
            Recommended position size
        """
        # Base order size from settings
        base_size = Decimal(str(self.trading_settings.base_order_size))

        # Adjust based on confidence
        confidence_multiplier = Decimal(str(signal.confidence))
        adjusted_size = base_size * confidence_multiplier

        # Apply maximum position size limit
        max_size = Decimal(str(self.trading_settings.max_position_size))
        adjusted_size = min(adjusted_size, max_size)

        # Ensure we don't exceed available balance
        adjusted_size = min(adjusted_size, available_balance * Decimal("0.95"))  # Keep 5% buffer

        # Apply risk-based sizing (Kelly criterion simplified)
        win_rate = Decimal("0.55")  # Assume 55% win rate
        risk_reward = Decimal("1.5")  # Assume 1.5 risk/reward ratio
        kelly_fraction = (win_rate * risk_reward - (1 - win_rate)) / risk_reward
        kelly_fraction = max(Decimal("0.01"), min(Decimal("0.25"), kelly_fraction))

        # Use half-Kelly for safety
        safe_fraction = kelly_fraction / 2
        kelly_size = self.portfolio_value * safe_fraction

        # Take the minimum of all constraints
        final_size = min(adjusted_size, kelly_size)

        self.logger.debug(
            "Position size calculated",
            base_size=str(base_size),
            adjusted_size=str(adjusted_size),
            kelly_size=str(kelly_size),
            final_size=str(final_size),
        )

        return max(Decimal("0"), final_size)

    def calculate_smart_leverage(self, signal: TradingSignal) -> int:
        """
        Calculate smart leverage based on Confidence and Volatility.
        High Confidence + Low Volatility = High Leverage
        Low Confidence + High Volatility = Low Leverage
        """
        # Base leverage settings
        MAX_LEVERAGE = 20  # Cap at 20x
        TARGET_RISK = 0.02  # Risk 2% of equity per trade
        AGGRESSION_MULTIPLIER = 10  # Multiplier to scale up leverage

        if signal.volatility <= 0:
            return 1

        # Kelly-like sizing for leverage
        # If volatility is 1% (0.01), and we want to risk 2%, raw leverage could be 2.
        # We scale this by signal confidence.

        raw_leverage = (TARGET_RISK / signal.volatility) * signal.confidence

        # Apply limits and aggression multiplier
        smart_leverage = int(round(raw_leverage * AGGRESSION_MULTIPLIER))
        smart_leverage = max(1, min(smart_leverage, MAX_LEVERAGE))

        # Safety clamp for high volatility
        if signal.volatility > 0.05:  # >5% daily moves
            smart_leverage = min(smart_leverage, 3)  # Cap at 3x for volatile assets

        self.logger.info(
            "Smart leverage calculated",
            symbol=signal.symbol,
            volatility=f"{signal.volatility:.2%}",
            confidence=signal.confidence,
            leverage=smart_leverage,
        )
        return smart_leverage

    def calculate_stop_loss(
        self,
        entry_price: Decimal,
        side: OrderSide,
    ) -> Decimal:
        """Calculate stop loss price.

        Args:
            entry_price: Entry price of the position
            side: Order side

        Returns:
            Stop loss price
        """
        stop_percent = Decimal(str(self.risk_settings.stop_loss_percent)) / 100

        if side == OrderSide.BUY:
            return entry_price * (1 - stop_percent)
        else:
            return entry_price * (1 + stop_percent)

    def calculate_dynamic_stop_loss(
        self, entry_price: Decimal, side: OrderSide, volatility: float
    ) -> Decimal:
        """Calculate Stop Loss based on Volatility (ATR) instead of fixed %."""
        # Use 2x Volatility as stop distance (standard practice)
        stop_distance_percent = Decimal(str(volatility)) * 2

        # Fallback to config if volatility is missing
        if stop_distance_percent == 0:
            stop_distance_percent = Decimal(str(self.risk_settings.stop_loss_percent)) / 100

        if side == OrderSide.BUY:
            return entry_price * (1 - stop_distance_percent)
        else:
            return entry_price * (1 + stop_distance_percent)

    def calculate_take_profit(
        self,
        entry_price: Decimal,
        side: OrderSide,
    ) -> Decimal:
        """Calculate take profit price.

        Args:
            entry_price: Entry price of the position
            side: Order side

        Returns:
            Take profit price
        """
        tp_percent = Decimal(str(self.risk_settings.take_profit_percent)) / 100

        if side == OrderSide.BUY:
            return entry_price * (1 + tp_percent)
        else:
            return entry_price * (1 - tp_percent)

    def calculate_dynamic_take_profit(
        self, entry_price: Decimal, side: OrderSide, volatility: float
    ) -> Decimal:
        """Calculate Take Profit striving for 1.5 Risk:Reward Ratio."""
        stop_distance_percent = Decimal(str(volatility)) * 2
        if stop_distance_percent == 0:
            stop_distance_percent = Decimal(str(self.risk_settings.stop_loss_percent)) / 100

        # Target 1.5x the risk
        tp_distance_percent = stop_distance_percent * Decimal("1.5")

        if side == OrderSide.BUY:
            return entry_price * (1 + tp_distance_percent)
        else:
            return entry_price * (1 - tp_distance_percent)

    def add_position(self, position: Position) -> None:
        """Add a new position to tracking.

        Args:
            position: Position to add
        """
        self.positions[position.symbol] = position
        self.daily_trades += 1

        self.logger.info(
            "Position added",
            symbol=position.symbol,
            side=position.side.value,
            size=str(position.size),
            entry_price=str(position.entry_price),
        )

    def update_position(self, symbol: str, current_price: Decimal) -> Position | None:
        """Update position with current price.

        Args:
            symbol: Trading pair symbol
            current_price: Current market price

        Returns:
            Updated position or None if not found
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        position.current_price = current_price

        return position

    def remove_position(self, symbol: str) -> Position | None:
        """Remove a position from tracking.

        Args:
            symbol: Trading pair symbol

        Returns:
            Removed position or None if not found
        """
        if symbol in self.positions:
            position = self.positions.pop(symbol)
            self.logger.info(
                "Position removed",
                symbol=symbol,
                pnl=str(position.unrealized_pnl),
            )
            return position
        return None

    def should_close_position(
        self,
        symbol: str,
        current_price: Decimal,
    ) -> tuple[bool, str]:
        """Check if position should be closed based on risk rules.

        Args:
            symbol: Trading pair symbol
            current_price: Current market price

        Returns:
            Tuple of (should_close, reason)
        """
        if symbol not in self.positions:
            return False, ""

        position = self.positions[symbol]
        position.current_price = current_price

        # Check stop loss
        if position.stop_loss and (
            (position.side == OrderSide.BUY and current_price <= position.stop_loss)
            or (position.side == OrderSide.SELL and current_price >= position.stop_loss)
        ):
            return True, "Stop loss triggered"

        # Check take profit
        if position.take_profit and (
            (position.side == OrderSide.BUY and current_price >= position.take_profit)
            or (position.side == OrderSide.SELL and current_price <= position.take_profit)
        ):
            return True, "Take profit triggered"

        # Check trailing stop (dynamic)
        pnl_percent = position.unrealized_pnl_percent
        if pnl_percent >= Decimal("5"):  # If we're up 5%+
            trailing_stop = Decimal("2")  # Use 2% trailing stop
            if pnl_percent < trailing_stop:
                return True, "Trailing stop triggered"

        return False, ""

    def add_to_history(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: Decimal,
        exit_price: Decimal,
        size: Decimal,
        pnl: Decimal,
    ) -> None:
        """Add a completed trade to history.

        Args:
            symbol: Trading pair symbol
            side: Order side
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            pnl: Realized profit/loss
        """
        self.trade_history.append({
            "timestamp": datetime.now(UTC),
            "symbol": symbol,
            "side": side.value,
            "entry_price": str(entry_price),
            "exit_price": str(exit_price),
            "size": str(size),
            "pnl": str(pnl),
        })

        self.daily_pnl += pnl

        # Keep last 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

    def get_trade_statistics(self) -> dict[str, Any]:
        """Calculate trade statistics.

        Returns:
            Dictionary of trade statistics
        """
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": "0",
                "average_win": "0",
                "average_loss": "0",
                "profit_factor": 0.0,
            }

        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if Decimal(t["pnl"]) > 0)
        losing_trades = sum(1 for t in self.trade_history if Decimal(t["pnl"]) < 0)

        total_pnl = sum(
            (Decimal(t["pnl"]) for t in self.trade_history),
            Decimal("0"),
        )

        wins = [Decimal(t["pnl"]) for t in self.trade_history if Decimal(t["pnl"]) > 0]
        losses = [Decimal(t["pnl"]) for t in self.trade_history if Decimal(t["pnl"]) < 0]

        average_win = sum(wins, Decimal("0")) / len(wins) if wins else Decimal("0")
        average_loss = sum(losses, Decimal("0")) / len(losses) if losses else Decimal("0")

        total_wins = sum(wins, Decimal("0"))
        total_losses = abs(sum(losses, Decimal("0"))) if losses else Decimal("1")
        profit_factor = float(total_wins / total_losses) if total_losses > 0 else 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0.0,
            "total_pnl": str(total_pnl),
            "average_win": str(average_win),
            "average_loss": str(average_loss),
            "profit_factor": profit_factor,
        }
