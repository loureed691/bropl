"""Tests for risk management."""

from decimal import Decimal

import pytest

from kucoin_bot.config import RiskSettings, TradingSettings
from kucoin_bot.models.data_models import (
    OrderSide,
    Position,
    SignalType,
    TradingSignal,
)
from kucoin_bot.risk_management.manager import RiskManager


class TestRiskManager:
    """Tests for risk manager."""

    @pytest.fixture
    def risk_settings(self) -> RiskSettings:
        """Create risk settings."""
        return RiskSettings(
            max_drawdown_percent=5.0,
            stop_loss_percent=2.0,
            take_profit_percent=3.0,
            max_open_positions=5,
        )

    @pytest.fixture
    def trading_settings(self) -> TradingSettings:
        """Create trading settings."""
        return TradingSettings(
            trading_pairs="BTC-USDT",
            base_order_size=100.0,
            max_position_size=1000.0,
            max_daily_trades=50,
        )

    @pytest.fixture
    def risk_manager(
        self,
        risk_settings: RiskSettings,
        trading_settings: TradingSettings,
    ) -> RiskManager:
        """Create risk manager instance."""
        return RiskManager(risk_settings, trading_settings)

    def test_set_portfolio_value(self, risk_manager: RiskManager) -> None:
        """Test setting portfolio value."""
        risk_manager.set_portfolio_value(Decimal("10000"))
        assert risk_manager.portfolio_value == Decimal("10000")
        assert risk_manager.peak_value == Decimal("10000")

    def test_peak_value_tracking(self, risk_manager: RiskManager) -> None:
        """Test peak value is tracked correctly."""
        risk_manager.set_portfolio_value(Decimal("10000"))
        risk_manager.set_portfolio_value(Decimal("12000"))
        assert risk_manager.peak_value == Decimal("12000")

        # Value decreases but peak should remain
        risk_manager.set_portfolio_value(Decimal("11000"))
        assert risk_manager.peak_value == Decimal("12000")

    def test_get_metrics(self, risk_manager: RiskManager) -> None:
        """Test getting risk metrics."""
        risk_manager.set_portfolio_value(Decimal("10000"))
        metrics = risk_manager.get_metrics()

        assert metrics.portfolio_value == Decimal("10000")
        assert metrics.max_positions == 5
        assert metrics.max_daily_trades == 50
        assert metrics.is_trading_allowed is True

    def test_check_signal_passes(self, risk_manager: RiskManager) -> None:
        """Test risk check passes for valid signal."""
        risk_manager.set_portfolio_value(Decimal("10000"))

        signal = TradingSignal(
            symbol="BTC-USDT",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=Decimal("50000"),
        )

        checks = risk_manager.check_signal(signal)
        assert any(check.passed for check in checks)

    def test_check_signal_low_confidence(self, risk_manager: RiskManager) -> None:
        """Test risk check fails for low confidence signal."""
        risk_manager.set_portfolio_value(Decimal("10000"))

        signal = TradingSignal(
            symbol="BTC-USDT",
            signal_type=SignalType.BUY,
            confidence=0.3,
            price=Decimal("50000"),
        )

        checks = risk_manager.check_signal(signal)
        assert any(not check.passed and "confidence" in check.reason.lower() for check in checks)

    def test_check_signal_max_positions(self, risk_manager: RiskManager) -> None:
        """Test risk check fails when max positions reached."""
        risk_manager.set_portfolio_value(Decimal("10000"))

        # Add max positions
        for i in range(5):
            position = Position(
                symbol=f"COIN{i}-USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("100"),
                size=Decimal("1"),
                current_price=Decimal("100"),
            )
            risk_manager.add_position(position)

        signal = TradingSignal(
            symbol="NEW-USDT",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=Decimal("50"),
        )

        checks = risk_manager.check_signal(signal)
        assert any(not check.passed and "positions" in check.reason.lower() for check in checks)

    def test_calculate_position_size(self, risk_manager: RiskManager) -> None:
        """Test position size calculation."""
        risk_manager.set_portfolio_value(Decimal("10000"))

        signal = TradingSignal(
            symbol="BTC-USDT",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=Decimal("50000"),
        )

        size = risk_manager.calculate_position_size(signal, Decimal("5000"))
        assert size > 0
        assert size <= Decimal("1000")  # Max position size

    def test_calculate_stop_loss_long(self, risk_manager: RiskManager) -> None:
        """Test stop loss calculation for long position."""
        stop_loss = risk_manager.calculate_stop_loss(
            Decimal("50000"),
            OrderSide.BUY,
        )
        # 2% below entry
        assert stop_loss == Decimal("49000")

    def test_calculate_stop_loss_short(self, risk_manager: RiskManager) -> None:
        """Test stop loss calculation for short position."""
        stop_loss = risk_manager.calculate_stop_loss(
            Decimal("50000"),
            OrderSide.SELL,
        )
        # 2% above entry
        assert stop_loss == Decimal("51000")

    def test_calculate_take_profit_long(self, risk_manager: RiskManager) -> None:
        """Test take profit calculation for long position."""
        take_profit = risk_manager.calculate_take_profit(
            Decimal("50000"),
            OrderSide.BUY,
        )
        # 3% above entry
        assert take_profit == Decimal("51500")

    def test_add_position(self, risk_manager: RiskManager) -> None:
        """Test adding a position."""
        position = Position(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            size=Decimal("1"),
            current_price=Decimal("50000"),
        )
        risk_manager.add_position(position)

        assert "BTC-USDT" in risk_manager.positions
        assert risk_manager.daily_trades == 1

    def test_update_position(self, risk_manager: RiskManager) -> None:
        """Test updating a position."""
        position = Position(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            size=Decimal("1"),
            current_price=Decimal("50000"),
        )
        risk_manager.add_position(position)

        updated = risk_manager.update_position("BTC-USDT", Decimal("51000"))
        assert updated is not None
        assert updated.current_price == Decimal("51000")

    def test_remove_position(self, risk_manager: RiskManager) -> None:
        """Test removing a position."""
        position = Position(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            size=Decimal("1"),
            current_price=Decimal("50000"),
        )
        risk_manager.add_position(position)

        removed = risk_manager.remove_position("BTC-USDT")
        assert removed is not None
        assert "BTC-USDT" not in risk_manager.positions

    def test_calculate_smart_leverage_low_volatility_high_confidence(
        self, risk_manager: RiskManager
    ) -> None:
        """Test smart leverage with low volatility and high confidence."""
        signal = TradingSignal(
            symbol="BTC-USDT",
            signal_type=SignalType.BUY,
            confidence=0.9,
            price=Decimal("50000"),
            volatility=0.01,  # 1% volatility
        )

        leverage = risk_manager.calculate_smart_leverage(signal)
        # With 0.01 volatility and 0.9 confidence: (0.02 / 0.01) * 0.9 * 10 = 18
        assert leverage > 1
        assert leverage <= 20  # Should not exceed MAX_LEVERAGE

    def test_calculate_smart_leverage_high_volatility(
        self, risk_manager: RiskManager
    ) -> None:
        """Test smart leverage with high volatility caps at 3x."""
        signal = TradingSignal(
            symbol="BTC-USDT",
            signal_type=SignalType.BUY,
            confidence=0.9,
            price=Decimal("50000"),
            volatility=0.08,  # 8% volatility (high)
        )

        leverage = risk_manager.calculate_smart_leverage(signal)
        # High volatility should cap at 3x
        assert leverage <= 3

    def test_calculate_smart_leverage_zero_volatility(
        self, risk_manager: RiskManager
    ) -> None:
        """Test smart leverage with zero volatility returns 1."""
        signal = TradingSignal(
            symbol="BTC-USDT",
            signal_type=SignalType.BUY,
            confidence=0.9,
            price=Decimal("50000"),
            volatility=0.0,
        )

        leverage = risk_manager.calculate_smart_leverage(signal)
        assert leverage == 1

    def test_calculate_dynamic_stop_loss_with_volatility(
        self, risk_manager: RiskManager
    ) -> None:
        """Test dynamic stop loss calculation with volatility."""
        # 2% volatility means 4% stop distance (2x volatility)
        stop_loss = risk_manager.calculate_dynamic_stop_loss(
            Decimal("50000"),
            OrderSide.BUY,
            0.02,
        )
        # 50000 * (1 - 0.04) = 48000
        assert stop_loss == Decimal("48000")

    def test_calculate_dynamic_stop_loss_fallback(
        self, risk_manager: RiskManager
    ) -> None:
        """Test dynamic stop loss falls back to config when volatility is zero."""
        stop_loss = risk_manager.calculate_dynamic_stop_loss(
            Decimal("50000"),
            OrderSide.BUY,
            0.0,
        )
        # Should use config's 2%
        assert stop_loss == Decimal("49000")

    def test_calculate_dynamic_take_profit_with_volatility(
        self, risk_manager: RiskManager
    ) -> None:
        """Test dynamic take profit calculation with volatility."""
        # 2% volatility means 4% stop distance, 6% take profit (1.5x stop)
        take_profit = risk_manager.calculate_dynamic_take_profit(
            Decimal("50000"),
            OrderSide.BUY,
            0.02,
        )
        # 50000 * (1 + 0.06) = 53000
        assert take_profit == Decimal("53000")

    def test_calculate_dynamic_take_profit_fallback(
        self, risk_manager: RiskManager
    ) -> None:
        """Test dynamic take profit falls back to config when volatility is zero."""
        take_profit = risk_manager.calculate_dynamic_take_profit(
            Decimal("50000"),
            OrderSide.BUY,
            0.0,
        )
        # Should use config's 2% stop * 1.5 = 3%
        assert take_profit == Decimal("51500")
        position = Position(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            size=Decimal("1"),
            current_price=Decimal("50000"),
        )
        risk_manager.add_position(position)

        removed = risk_manager.remove_position("BTC-USDT")
        assert removed is not None
        assert "BTC-USDT" not in risk_manager.positions

    def test_should_close_position_stop_loss(self, risk_manager: RiskManager) -> None:
        """Test position should close at stop loss."""
        position = Position(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            size=Decimal("1"),
            current_price=Decimal("50000"),
            stop_loss=Decimal("49000"),
        )
        risk_manager.add_position(position)

        should_close, reason = risk_manager.should_close_position(
            "BTC-USDT",
            Decimal("48500"),
        )
        assert should_close is True
        assert "stop loss" in reason.lower()

    def test_should_close_position_take_profit(self, risk_manager: RiskManager) -> None:
        """Test position should close at take profit."""
        position = Position(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            size=Decimal("1"),
            current_price=Decimal("50000"),
            take_profit=Decimal("52000"),
        )
        risk_manager.add_position(position)

        should_close, reason = risk_manager.should_close_position(
            "BTC-USDT",
            Decimal("52500"),
        )
        assert should_close is True
        assert "take profit" in reason.lower()

    def test_add_to_history(self, risk_manager: RiskManager) -> None:
        """Test adding trade to history."""
        risk_manager.add_to_history(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            exit_price=Decimal("51000"),
            size=Decimal("1"),
            pnl=Decimal("1000"),
        )

        assert len(risk_manager.trade_history) == 1
        assert risk_manager.daily_pnl == Decimal("1000")

    def test_get_trade_statistics(self, risk_manager: RiskManager) -> None:
        """Test trade statistics calculation."""
        # Add winning trades
        for _ in range(3):
            risk_manager.add_to_history(
                symbol="BTC-USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000"),
                size=Decimal("1"),
                pnl=Decimal("1000"),
            )

        # Add losing trade
        risk_manager.add_to_history(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            exit_price=Decimal("49000"),
            size=Decimal("1"),
            pnl=Decimal("-1000"),
        )

        stats = risk_manager.get_trade_statistics()
        assert stats["total_trades"] == 4
        assert stats["winning_trades"] == 3
        assert stats["losing_trades"] == 1
        assert stats["win_rate"] == 0.75
