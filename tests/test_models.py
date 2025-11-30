"""Tests for data models."""

from datetime import datetime
from decimal import Decimal

from kucoin_bot.models.data_models import (
    AccountBalance,
    Candle,
    MarketDepth,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    SignalType,
    Ticker,
    TradingSignal,
)


class TestTicker:
    """Tests for Ticker model."""

    def test_spread_calculation(self) -> None:
        """Test bid-ask spread calculation."""
        ticker = Ticker(
            symbol="BTC-USDT",
            price=Decimal("50000"),
            best_bid=Decimal("49990"),
            best_ask=Decimal("50010"),
            volume_24h=Decimal("1000"),
            change_24h=Decimal("2.5"),
            high_24h=Decimal("51000"),
            low_24h=Decimal("49000"),
            timestamp=datetime.utcnow(),
        )
        assert ticker.spread == Decimal("20")
        assert ticker.spread_percent == Decimal("0.04")


class TestCandle:
    """Tests for Candle model."""

    def test_bullish_candle(self) -> None:
        """Test bullish candle detection."""
        candle = Candle(
            symbol="BTC-USDT",
            timestamp=datetime.utcnow(),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100"),
        )
        assert candle.is_bullish is True
        assert candle.is_bearish is False
        assert candle.body_size == Decimal("500")

    def test_bearish_candle(self) -> None:
        """Test bearish candle detection."""
        candle = Candle(
            symbol="BTC-USDT",
            timestamp=datetime.utcnow(),
            open=Decimal("50500"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50000"),
            volume=Decimal("100"),
        )
        assert candle.is_bullish is False
        assert candle.is_bearish is True


class TestOrder:
    """Tests for Order model."""

    def test_order_fill_rate(self) -> None:
        """Test order fill rate calculation."""
        order = Order(
            client_order_id="test123",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("50000"),
            size=Decimal("1.0"),
            filled_size=Decimal("0.5"),
            status=OrderStatus.PARTIALLY_FILLED,
        )
        assert order.fill_rate == Decimal("50")
        assert order.remaining_size == Decimal("0.5")
        assert order.is_filled is False

    def test_filled_order(self) -> None:
        """Test fully filled order."""
        order = Order(
            client_order_id="test123",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("1.0"),
            filled_size=Decimal("1.0"),
            status=OrderStatus.FILLED,
        )
        assert order.is_filled is True
        assert order.fill_rate == Decimal("100")


class TestPosition:
    """Tests for Position model."""

    def test_long_position_profit(self) -> None:
        """Test long position with profit."""
        position = Position(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            size=Decimal("1.0"),
            current_price=Decimal("52000"),
        )
        assert position.unrealized_pnl == Decimal("2000")
        assert position.unrealized_pnl_percent == Decimal("4")
        assert position.position_value == Decimal("52000")

    def test_long_position_loss(self) -> None:
        """Test long position with loss."""
        position = Position(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            size=Decimal("1.0"),
            current_price=Decimal("48000"),
        )
        assert position.unrealized_pnl == Decimal("-2000")

    def test_short_position_profit(self) -> None:
        """Test short position with profit."""
        position = Position(
            symbol="BTC-USDT",
            side=OrderSide.SELL,
            entry_price=Decimal("50000"),
            size=Decimal("1.0"),
            current_price=Decimal("48000"),
        )
        assert position.unrealized_pnl == Decimal("2000")


class TestTradingSignal:
    """Tests for TradingSignal model."""

    def test_actionable_signal(self) -> None:
        """Test actionable signal detection."""
        buy_signal = TradingSignal(
            symbol="BTC-USDT",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=Decimal("50000"),
        )
        assert buy_signal.is_actionable is True

    def test_hold_signal(self) -> None:
        """Test hold signal is not actionable."""
        hold_signal = TradingSignal(
            symbol="BTC-USDT",
            signal_type=SignalType.HOLD,
            confidence=0.5,
            price=Decimal("50000"),
        )
        assert hold_signal.is_actionable is False


class TestMarketDepth:
    """Tests for MarketDepth model."""

    def test_market_depth_calculations(self) -> None:
        """Test market depth calculations."""
        depth = MarketDepth(
            symbol="BTC-USDT",
            bids=[
                (Decimal("49990"), Decimal("1.0")),
                (Decimal("49980"), Decimal("2.0")),
            ],
            asks=[
                (Decimal("50010"), Decimal("1.5")),
                (Decimal("50020"), Decimal("2.5")),
            ],
            timestamp=datetime.utcnow(),
        )
        assert depth.best_bid == Decimal("49990")
        assert depth.best_ask == Decimal("50010")
        assert depth.mid_price == Decimal("50000")
        assert depth.get_bid_depth(2) == Decimal("3.0")
        assert depth.get_ask_depth(2) == Decimal("4.0")


class TestAccountBalance:
    """Tests for AccountBalance model."""

    def test_usage_percentage(self) -> None:
        """Test balance usage calculation."""
        balance = AccountBalance(
            currency="USDT",
            available=Decimal("8000"),
            holds=Decimal("2000"),
            total=Decimal("10000"),
        )
        assert balance.usage_percent == Decimal("20")
