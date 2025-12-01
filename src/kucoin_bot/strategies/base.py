"""Base strategy interface and common strategy implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
import structlog

from kucoin_bot.config import StrategySettings
from kucoin_bot.indicators.technical import SignalGenerator, TechnicalIndicators
from kucoin_bot.models.data_models import (
    Candle,
    OrderSide,
    SignalType,
    TradingSignal,
)

logger = structlog.get_logger()


@dataclass
class StrategyState:
    """Maintains state for a trading strategy."""

    symbol: str
    last_signal: TradingSignal | None = None
    position_side: OrderSide | None = None
    entry_price: Decimal | None = None
    candles: list[Candle] = field(default_factory=list)
    max_candles: int = 500
    indicators: dict[str, float] = field(default_factory=dict)

    def add_candle(self, candle: Candle) -> None:
        """Add a candle to the history."""
        self.candles.append(candle)
        if len(self.candles) > self.max_candles:
            self.candles = self.candles[-self.max_candles :]

    def get_closes(self) -> list[float]:
        """Get close prices from candle history."""
        return [float(c.close) for c in self.candles]

    def get_highs(self) -> list[float]:
        """Get high prices from candle history."""
        return [float(c.high) for c in self.candles]

    def get_lows(self) -> list[float]:
        """Get low prices from candle history."""
        return [float(c.low) for c in self.candles]

    def get_volumes(self) -> list[float]:
        """Get volumes from candle history."""
        return [float(c.volume) for c in self.candles]


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, settings: StrategySettings) -> None:
        """Initialize strategy with settings.

        Args:
            settings: Strategy configuration settings
        """
        self.settings = settings
        self.states: dict[str, StrategyState] = {}
        self.indicators = TechnicalIndicators()
        self.signal_generator = SignalGenerator()
        self.logger = logger.bind(strategy=self.name)

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name identifier."""
        ...

    @abstractmethod
    def analyze(self, state: StrategyState) -> TradingSignal:
        """Analyze market data and generate trading signal.

        Args:
            state: Current strategy state with market data

        Returns:
            Trading signal
        """
        ...

    def get_state(self, symbol: str) -> StrategyState:
        """Get or create state for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Strategy state for the symbol
        """
        if symbol not in self.states:
            self.states[symbol] = StrategyState(symbol=symbol)
        return self.states[symbol]

    def update(self, symbol: str, candle: Candle) -> TradingSignal:
        """Update strategy with new candle data.

        Args:
            symbol: Trading pair symbol
            candle: New candle data

        Returns:
            Trading signal
        """
        state = self.get_state(symbol)
        state.add_candle(candle)

        if len(state.candles) < self.min_candles:
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0,
                price=candle.close,
                reason="Insufficient data",
            )

        signal = self.analyze(state)
        state.last_signal = signal

        self.logger.info(
            "Signal generated",
            symbol=symbol,
            signal=signal.signal_type.value,
            confidence=signal.confidence,
            reason=signal.reason,
        )

        return signal

    @property
    def min_candles(self) -> int:
        """Minimum number of candles required for analysis."""
        return 50

    def reset(self, symbol: str | None = None) -> None:
        """Reset strategy state.

        Args:
            symbol: Optional symbol to reset, or None to reset all
        """
        if symbol:
            if symbol in self.states:
                del self.states[symbol]
        else:
            self.states.clear()


class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy using RSI and EMA crossovers."""

    @property
    def name(self) -> str:
        return "momentum"

    def analyze(self, state: StrategyState) -> TradingSignal:
        """Analyze using momentum indicators."""
        closes = state.get_closes()
        current_price = Decimal(str(closes[-1]))

        # Calculate RSI
        rsi_signal = self.signal_generator.rsi_signal(
            closes,
            period=self.settings.rsi_period,
            overbought=self.settings.rsi_overbought,
            oversold=self.settings.rsi_oversold,
        )

        # Calculate trend from EMA crossover
        trend_signal = self.signal_generator.trend_signal(
            closes,
            short_period=self.settings.ema_short_period,
            long_period=self.settings.ema_long_period,
        )

        # Calculate MACD
        macd_signal = self.signal_generator.macd_signal(closes)

        # Store indicators
        state.indicators = {
            "rsi": rsi_signal.value,
            "trend": trend_signal.value,
            "macd": macd_signal.value,
        }

        # Decision logic
        bullish_count = sum(
            1 for s in [rsi_signal, trend_signal, macd_signal] if s.signal == "bullish"
        )
        bearish_count = sum(
            1 for s in [rsi_signal, trend_signal, macd_signal] if s.signal == "bearish"
        )

        # Calculate confidence
        signals = [rsi_signal, trend_signal, macd_signal]
        avg_strength = sum(s.strength for s in signals) / len(signals)

        if bullish_count >= 2 and trend_signal.signal == "bullish":
            signal_type = SignalType.BUY
            confidence = avg_strength
            reason = f"Momentum bullish: RSI={rsi_signal.value:.1f}, Trend={trend_signal.signal}"
        elif bearish_count >= 2 and trend_signal.signal == "bearish":
            signal_type = SignalType.SELL
            confidence = avg_strength
            reason = f"Momentum bearish: RSI={rsi_signal.value:.1f}, Trend={trend_signal.signal}"
        else:
            signal_type = SignalType.HOLD
            confidence = 0.5
            reason = "No clear momentum signal"

        return TradingSignal(
            symbol=state.symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            indicators=state.indicators,
            reason=reason,
        )


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands."""

    @property
    def name(self) -> str:
        return "mean_reversion"

    def analyze(self, state: StrategyState) -> TradingSignal:
        """Analyze using mean reversion indicators."""
        closes = state.get_closes()
        current_price = Decimal(str(closes[-1]))

        # Calculate Bollinger Bands
        bb_signal = self.signal_generator.bollinger_signal(closes)

        # Calculate RSI for confirmation
        rsi_signal = self.signal_generator.rsi_signal(
            closes,
            period=self.settings.rsi_period,
            overbought=self.settings.rsi_overbought,
            oversold=self.settings.rsi_oversold,
        )

        state.indicators = {
            "bb_position": bb_signal.value,
            "rsi": rsi_signal.value,
        }

        # Buy when price is at lower band and RSI confirms oversold
        if bb_signal.signal == "bullish" and rsi_signal.signal == "bullish":
            signal_type = SignalType.BUY
            confidence = (bb_signal.strength + rsi_signal.strength) / 2
            reason = f"Mean reversion buy: BB position={bb_signal.value:.2f}, RSI={rsi_signal.value:.1f}"
        # Sell when price is at upper band and RSI confirms overbought
        elif bb_signal.signal == "bearish" and rsi_signal.signal == "bearish":
            signal_type = SignalType.SELL
            confidence = (bb_signal.strength + rsi_signal.strength) / 2
            reason = f"Mean reversion sell: BB position={bb_signal.value:.2f}, RSI={rsi_signal.value:.1f}"
        else:
            signal_type = SignalType.HOLD
            confidence = 0.5
            reason = "No mean reversion opportunity"

        return TradingSignal(
            symbol=state.symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            indicators=state.indicators,
            reason=reason,
        )


class GridStrategy(BaseStrategy):
    """Grid trading strategy for ranging markets."""

    def __init__(
        self,
        settings: StrategySettings,
        grid_levels: int = 10,
        grid_spacing_percent: float = 1.0,
    ) -> None:
        """Initialize grid strategy.

        Args:
            settings: Strategy settings
            grid_levels: Number of grid levels
            grid_spacing_percent: Spacing between levels as percentage
        """
        super().__init__(settings)
        self.grid_levels = grid_levels
        self.grid_spacing_percent = grid_spacing_percent
        self.grid_prices: dict[str, list[Decimal]] = {}
        self.active_grids: dict[str, set[int]] = {}

    @property
    def name(self) -> str:
        return "grid"

    def setup_grid(self, symbol: str, center_price: Decimal) -> None:
        """Setup grid levels around a center price.

        Args:
            symbol: Trading pair symbol
            center_price: Price to center the grid around
        """
        levels = []
        for i in range(-self.grid_levels // 2, self.grid_levels // 2 + 1):
            multiplier = Decimal(str(1 + (i * self.grid_spacing_percent / 100)))
            levels.append(center_price * multiplier)
        self.grid_prices[symbol] = sorted(levels)
        self.active_grids[symbol] = set()

    def analyze(self, state: StrategyState) -> TradingSignal:
        """Analyze for grid trading opportunities."""
        closes = state.get_closes()
        current_price = Decimal(str(closes[-1]))

        # Initialize grid if not exists
        if state.symbol not in self.grid_prices:
            self.setup_grid(state.symbol, current_price)

        grid_levels = self.grid_prices[state.symbol]
        active = self.active_grids[state.symbol]

        # Find current grid level
        current_level = 0
        for i, level in enumerate(grid_levels):
            if current_price >= level:
                current_level = i

        state.indicators = {
            "current_level": current_level,
            "grid_levels": len(grid_levels),
        }

        # Buy when price drops to a new level
        if current_level not in active and current_level < len(grid_levels) // 2:
            active.add(current_level)
            return TradingSignal(
                symbol=state.symbol,
                signal_type=SignalType.BUY,
                confidence=0.7,
                price=current_price,
                indicators=state.indicators,
                reason=f"Grid buy at level {current_level}",
            )
        # Sell when price rises to a new level
        elif current_level in active and current_level > len(grid_levels) // 2:
            active.discard(current_level)
            return TradingSignal(
                symbol=state.symbol,
                signal_type=SignalType.SELL,
                confidence=0.7,
                price=current_price,
                indicators=state.indicators,
                reason=f"Grid sell at level {current_level}",
            )

        return TradingSignal(
            symbol=state.symbol,
            signal_type=SignalType.HOLD,
            confidence=0.5,
            price=current_price,
            indicators=state.indicators,
            reason="Waiting for grid level",
        )


class ScalpingStrategy(BaseStrategy):
    """High-frequency scalping strategy for quick profits."""

    @property
    def name(self) -> str:
        return "scalping"

    @property
    def min_candles(self) -> int:
        return 20  # Scalping needs less historical data

    def analyze(self, state: StrategyState) -> TradingSignal:
        """Analyze for scalping opportunities."""
        closes = state.get_closes()
        highs = state.get_highs()
        lows = state.get_lows()
        current_price = Decimal(str(closes[-1]))

        # Calculate short-term indicators
        rsi = self.indicators.rsi(closes, period=7)  # Fast RSI
        stoch_k, stoch_d = self.indicators.stochastic(highs, lows, closes, k_period=5, d_period=3)

        current_rsi = rsi[-1]
        current_stoch_k = stoch_k[-1]
        current_stoch_d = stoch_d[-1]

        if np.isnan(current_rsi) or np.isnan(current_stoch_k):
            return TradingSignal(
                symbol=state.symbol,
                signal_type=SignalType.HOLD,
                confidence=0,
                price=current_price,
                reason="Insufficient data",
            )

        state.indicators = {
            "fast_rsi": float(current_rsi),
            "stoch_k": float(current_stoch_k),
            "stoch_d": float(current_stoch_d),
        }

        # Scalp buy: Both RSI and Stochastic oversold
        if current_rsi < 25 and current_stoch_k < 20 and current_stoch_k > current_stoch_d:
            return TradingSignal(
                symbol=state.symbol,
                signal_type=SignalType.BUY,
                confidence=0.8,
                price=current_price,
                indicators=state.indicators,
                reason=f"Scalp buy: RSI={current_rsi:.1f}, Stoch={current_stoch_k:.1f}",
            )
        # Scalp sell: Both RSI and Stochastic overbought
        elif current_rsi > 75 and current_stoch_k > 80 and current_stoch_k < current_stoch_d:
            return TradingSignal(
                symbol=state.symbol,
                signal_type=SignalType.SELL,
                confidence=0.8,
                price=current_price,
                indicators=state.indicators,
                reason=f"Scalp sell: RSI={current_rsi:.1f}, Stoch={current_stoch_k:.1f}",
            )

        return TradingSignal(
            symbol=state.symbol,
            signal_type=SignalType.HOLD,
            confidence=0.5,
            price=current_price,
            indicators=state.indicators,
            reason="No scalping opportunity",
        )


class DCAStrategy(BaseStrategy):
    """Dollar Cost Averaging strategy for long-term accumulation."""

    def __init__(
        self,
        settings: StrategySettings,
        buy_interval_candles: int = 24,  # Buy every 24 candles
        dip_threshold_percent: float = 3.0,  # Extra buy on 3% dips
    ) -> None:
        """Initialize DCA strategy.

        Args:
            settings: Strategy settings
            buy_interval_candles: Regular buy interval
            dip_threshold_percent: Percentage drop to trigger extra buy
        """
        super().__init__(settings)
        self.buy_interval = buy_interval_candles
        self.dip_threshold = dip_threshold_percent
        self.last_buy_candle: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "dca"

    def analyze(self, state: StrategyState) -> TradingSignal:
        """Analyze for DCA opportunities."""
        closes = state.get_closes()
        current_price = Decimal(str(closes[-1]))
        candle_count = len(state.candles)

        # Track last buy
        if state.symbol not in self.last_buy_candle:
            self.last_buy_candle[state.symbol] = 0

        candles_since_buy = candle_count - self.last_buy_candle[state.symbol]

        # Calculate recent price change
        if len(closes) >= 24:
            recent_high = max(closes[-24:])
            price_change_percent = ((closes[-1] - recent_high) / recent_high) * 100
        else:
            price_change_percent = 0

        state.indicators = {
            "candles_since_buy": candles_since_buy,
            "price_change_24": price_change_percent,
        }

        # Regular DCA buy
        if candles_since_buy >= self.buy_interval:
            self.last_buy_candle[state.symbol] = candle_count
            return TradingSignal(
                symbol=state.symbol,
                signal_type=SignalType.BUY,
                confidence=0.6,
                price=current_price,
                indicators=state.indicators,
                reason=f"Regular DCA buy after {candles_since_buy} candles",
            )

        # Dip buy opportunity
        if price_change_percent <= -self.dip_threshold:
            self.last_buy_candle[state.symbol] = candle_count
            return TradingSignal(
                symbol=state.symbol,
                signal_type=SignalType.BUY,
                confidence=0.8,
                price=current_price,
                indicators=state.indicators,
                reason=f"DCA dip buy: {price_change_percent:.1f}% drop",
            )

        return TradingSignal(
            symbol=state.symbol,
            signal_type=SignalType.HOLD,
            confidence=0.5,
            price=current_price,
            indicators=state.indicators,
            reason=f"DCA waiting: {self.buy_interval - candles_since_buy} candles until next buy",
        )


def create_strategy(settings: StrategySettings) -> BaseStrategy:
    """Factory function to create strategy based on settings.

    Args:
        settings: Strategy configuration

    Returns:
        Configured strategy instance
    """
    strategy_map: dict[str, type[BaseStrategy]] = {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "grid": GridStrategy,
        "scalping": ScalpingStrategy,
        "dca": DCAStrategy,
    }

    strategy_class = strategy_map.get(settings.strategy_name.value)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {settings.strategy_name}")

    return strategy_class(settings)
