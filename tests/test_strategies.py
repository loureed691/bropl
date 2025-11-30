"""Tests for trading strategies."""

from datetime import datetime
from decimal import Decimal

import pytest

from kucoin_bot.config import StrategySettings
from kucoin_bot.models.data_models import Candle, SignalType
from kucoin_bot.strategies.base import (
    DCAStrategy,
    GridStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    ScalpingStrategy,
    StrategyState,
    create_strategy,
)


class TestStrategyState:
    """Tests for strategy state management."""

    def test_add_candle(self) -> None:
        """Test adding candles to state."""
        state = StrategyState(symbol="BTC-USDT", max_candles=10)

        for i in range(15):
            candle = Candle(
                symbol="BTC-USDT",
                timestamp=datetime.utcnow(),
                open=Decimal(str(100 + i)),
                high=Decimal(str(105 + i)),
                low=Decimal(str(95 + i)),
                close=Decimal(str(102 + i)),
                volume=Decimal("100"),
            )
            state.add_candle(candle)

        # Should only keep max_candles
        assert len(state.candles) == 10

    def test_get_closes(self) -> None:
        """Test extracting close prices."""
        state = StrategyState(symbol="BTC-USDT")

        for i in range(5):
            candle = Candle(
                symbol="BTC-USDT",
                timestamp=datetime.utcnow(),
                open=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("95"),
                close=Decimal(str(100 + i)),
                volume=Decimal("100"),
            )
            state.add_candle(candle)

        closes = state.get_closes()
        assert closes == [100.0, 101.0, 102.0, 103.0, 104.0]


class TestMomentumStrategy:
    """Tests for momentum strategy."""

    @pytest.fixture
    def settings(self) -> StrategySettings:
        """Create strategy settings."""
        return StrategySettings()

    @pytest.fixture
    def strategy(self, settings: StrategySettings) -> MomentumStrategy:
        """Create momentum strategy instance."""
        return MomentumStrategy(settings)

    def test_strategy_name(self, strategy: MomentumStrategy) -> None:
        """Test strategy name."""
        assert strategy.name == "momentum"

    def test_min_candles(self, strategy: MomentumStrategy) -> None:
        """Test minimum candles requirement."""
        assert strategy.min_candles == 50

    def test_update_with_insufficient_data(
        self,
        strategy: MomentumStrategy,
    ) -> None:
        """Test strategy with insufficient data."""
        candle = Candle(
            symbol="BTC-USDT",
            timestamp=datetime.utcnow(),
            open=Decimal("50000"),
            high=Decimal("50500"),
            low=Decimal("49500"),
            close=Decimal("50200"),
            volume=Decimal("100"),
        )

        signal = strategy.update("BTC-USDT", candle)
        assert signal.signal_type == SignalType.HOLD
        assert signal.reason == "Insufficient data"

    def test_update_with_sufficient_data(
        self,
        strategy: MomentumStrategy,
    ) -> None:
        """Test strategy with sufficient data."""
        # Add enough candles for strategy to work
        for i in range(60):
            candle = Candle(
                symbol="BTC-USDT",
                timestamp=datetime.utcnow(),
                open=Decimal(str(50000 + i * 10)),
                high=Decimal(str(50100 + i * 10)),
                low=Decimal(str(49900 + i * 10)),
                close=Decimal(str(50050 + i * 10)),
                volume=Decimal("100"),
            )
            signal = strategy.update("BTC-USDT", candle)

        # Should get a real signal (not "Insufficient data")
        assert signal.reason != "Insufficient data"


class TestMeanReversionStrategy:
    """Tests for mean reversion strategy."""

    @pytest.fixture
    def strategy(self) -> MeanReversionStrategy:
        """Create mean reversion strategy instance."""
        return MeanReversionStrategy(StrategySettings())

    def test_strategy_name(self, strategy: MeanReversionStrategy) -> None:
        """Test strategy name."""
        assert strategy.name == "mean_reversion"


class TestGridStrategy:
    """Tests for grid trading strategy."""

    @pytest.fixture
    def strategy(self) -> GridStrategy:
        """Create grid strategy instance."""
        return GridStrategy(StrategySettings(), grid_levels=10, grid_spacing_percent=1.0)

    def test_strategy_name(self, strategy: GridStrategy) -> None:
        """Test strategy name."""
        assert strategy.name == "grid"

    def test_setup_grid(self, strategy: GridStrategy) -> None:
        """Test grid level setup."""
        strategy.setup_grid("BTC-USDT", Decimal("50000"))

        assert "BTC-USDT" in strategy.grid_prices
        assert len(strategy.grid_prices["BTC-USDT"]) > 0


class TestScalpingStrategy:
    """Tests for scalping strategy."""

    @pytest.fixture
    def strategy(self) -> ScalpingStrategy:
        """Create scalping strategy instance."""
        return ScalpingStrategy(StrategySettings())

    def test_strategy_name(self, strategy: ScalpingStrategy) -> None:
        """Test strategy name."""
        assert strategy.name == "scalping"

    def test_min_candles(self, strategy: ScalpingStrategy) -> None:
        """Test minimum candles (should be lower for scalping)."""
        assert strategy.min_candles == 20


class TestDCAStrategy:
    """Tests for DCA strategy."""

    @pytest.fixture
    def strategy(self) -> DCAStrategy:
        """Create DCA strategy instance."""
        return DCAStrategy(
            StrategySettings(),
            buy_interval_candles=24,
            dip_threshold_percent=3.0,
        )

    def test_strategy_name(self, strategy: DCAStrategy) -> None:
        """Test strategy name."""
        assert strategy.name == "dca"


class TestCreateStrategy:
    """Tests for strategy factory function."""

    def test_create_momentum_strategy(self) -> None:
        """Test creating momentum strategy."""
        from kucoin_bot.config import StrategyName

        settings = StrategySettings(strategy_name=StrategyName.MOMENTUM)
        strategy = create_strategy(settings)
        assert isinstance(strategy, MomentumStrategy)

    def test_create_mean_reversion_strategy(self) -> None:
        """Test creating mean reversion strategy."""
        from kucoin_bot.config import StrategyName

        settings = StrategySettings(strategy_name=StrategyName.MEAN_REVERSION)
        strategy = create_strategy(settings)
        assert isinstance(strategy, MeanReversionStrategy)

    def test_create_grid_strategy(self) -> None:
        """Test creating grid strategy."""
        from kucoin_bot.config import StrategyName

        settings = StrategySettings(strategy_name=StrategyName.GRID)
        strategy = create_strategy(settings)
        assert isinstance(strategy, GridStrategy)
