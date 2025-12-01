"""Technical analysis indicators for trading strategies."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class IndicatorResult:
    """Container for indicator calculation results."""

    name: str
    value: float
    signal: str = "neutral"  # bullish, bearish, neutral
    strength: float = 0.0  # 0.0 to 1.0


class TechnicalIndicators:
    """Collection of technical analysis indicators."""

    @staticmethod
    def sma(data: Sequence[float], period: int) -> NDArray[Any]:
        """Calculate Simple Moving Average.

        Args:
            data: Price data
            period: SMA period

        Returns:
            SMA values
        """
        series = pd.Series(data)
        return series.rolling(window=period).mean().to_numpy()

    @staticmethod
    def ema(data: Sequence[float], period: int) -> NDArray[Any]:
        """Calculate Exponential Moving Average.

        Args:
            data: Price data
            period: EMA period

        Returns:
            EMA values
        """
        series = pd.Series(data)
        return series.ewm(span=period, adjust=False).mean().to_numpy()

    @staticmethod
    def rsi(data: Sequence[float], period: int = 14) -> NDArray[Any]:
        """Calculate Relative Strength Index.

        Args:
            data: Price data
            period: RSI period

        Returns:
            RSI values (0-100)
        """
        series = pd.Series(data)
        delta = series.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.to_numpy()

    @staticmethod
    def macd(
        data: Sequence[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Calculate MACD (Moving Average Convergence Divergence).

        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        series = pd.Series(data)
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line.to_numpy(), signal_line.to_numpy(), histogram.to_numpy()

    @staticmethod
    def bollinger_bands(
        data: Sequence[float],
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Calculate Bollinger Bands.

        Args:
            data: Price data
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        series = pd.Series(data)
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper.to_numpy(), middle.to_numpy(), lower.to_numpy()

    @staticmethod
    def atr(
        high: Sequence[float],
        low: Sequence[float],
        close: Sequence[float],
        period: int = 14,
    ) -> NDArray[Any]:
        """Calculate Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR values
        """
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)

        tr1 = high_s - low_s
        tr2 = abs(high_s - close_s.shift())
        tr3 = abs(low_s - close_s.shift())

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr.to_numpy()

    @staticmethod
    def stochastic(
        high: Sequence[float],
        low: Sequence[float],
        close: Sequence[float],
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Calculate Stochastic Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D period

        Returns:
            Tuple of (%K, %D)
        """
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)

        lowest_low = low_s.rolling(window=k_period).min()
        highest_high = high_s.rolling(window=k_period).max()

        k = 100 * ((close_s - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()

        return k.to_numpy(), d.to_numpy()

    @staticmethod
    def adx(
        high: Sequence[float],
        low: Sequence[float],
        close: Sequence[float],
        period: int = 14,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Calculate Average Directional Index.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period

        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)

        tr1 = high_s - low_s
        tr2 = abs(high_s - close_s.shift())
        tr3 = abs(low_s - close_s.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high_s.diff()
        minus_dm = -low_s.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr_smooth = true_range.rolling(window=period).sum()
        plus_dm_smooth = plus_dm.rolling(window=period).sum()
        minus_dm_smooth = minus_dm.rolling(window=period).sum()

        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx.to_numpy(), plus_di.to_numpy(), minus_di.to_numpy()

    @staticmethod
    def vwap(
        high: Sequence[float],
        low: Sequence[float],
        close: Sequence[float],
        volume: Sequence[float],
    ) -> NDArray[Any]:
        """Calculate Volume Weighted Average Price.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data

        Returns:
            VWAP values
        """
        typical_price = (np.array(high) + np.array(low) + np.array(close)) / 3
        vol = np.array(volume)

        cumulative_tp_vol = np.cumsum(typical_price * vol)
        cumulative_vol = np.cumsum(vol)

        result: NDArray[Any] = cumulative_tp_vol / cumulative_vol
        return result

    @staticmethod
    def obv(close: Sequence[float], volume: Sequence[float]) -> NDArray[Any]:
        """Calculate On-Balance Volume.

        Args:
            close: Close prices
            volume: Volume data

        Returns:
            OBV values
        """
        close_s = pd.Series(close)
        volume_s = pd.Series(volume)

        direction = close_s.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * volume_s).cumsum()

        return obv.to_numpy()

    @staticmethod
    def fibonacci_levels(
        high: float,
        low: float,
    ) -> dict[str, float]:
        """Calculate Fibonacci retracement levels.

        Args:
            high: Highest price
            low: Lowest price

        Returns:
            Dictionary of Fibonacci levels
        """
        diff = high - low
        return {
            "0.0": high,
            "0.236": high - (diff * 0.236),
            "0.382": high - (diff * 0.382),
            "0.5": high - (diff * 0.5),
            "0.618": high - (diff * 0.618),
            "0.786": high - (diff * 0.786),
            "1.0": low,
        }

    @staticmethod
    def support_resistance(
        high: Sequence[float],
        low: Sequence[float],
        _: Sequence[float],
        period: int = 20,
    ) -> tuple[list[float], list[float]]:
        """Identify support and resistance levels.

        Args:
            high: High prices
            low: Low prices
            _: Close prices (unused, kept for interface consistency)
            period: Lookback period

        Returns:
            Tuple of (support levels, resistance levels)
        """
        high_arr = np.array(high)
        low_arr = np.array(low)

        # Find local maxima for resistance
        resistance = []
        for i in range(period, len(high_arr) - period):
            if high_arr[i] == max(high_arr[i - period : i + period + 1]):
                resistance.append(float(high_arr[i]))

        # Find local minima for support
        support = []
        for i in range(period, len(low_arr) - period):
            if low_arr[i] == min(low_arr[i - period : i + period + 1]):
                support.append(float(low_arr[i]))

        return support, resistance

    @staticmethod
    def calculate_trend(
        data: Sequence[float],
        short_period: int = 9,
        long_period: int = 21,
    ) -> str:
        """Determine overall trend direction.

        Args:
            data: Price data
            short_period: Short EMA period
            long_period: Long EMA period

        Returns:
            Trend direction: "bullish", "bearish", or "neutral"
        """
        if len(data) < long_period:
            return "neutral"

        short_ema = TechnicalIndicators.ema(data, short_period)
        long_ema = TechnicalIndicators.ema(data, long_period)

        # Check current crossover state
        current_short = short_ema[-1]
        current_long = long_ema[-1]

        if np.isnan(current_short) or np.isnan(current_long):
            return "neutral"

        if current_short > current_long:
            return "bullish"
        elif current_short < current_long:
            return "bearish"
        return "neutral"


class SignalGenerator:
    """Generate trading signals from technical indicators."""

    def __init__(self) -> None:
        """Initialize signal generator."""
        self.indicators = TechnicalIndicators()

    def rsi_signal(
        self,
        data: Sequence[float],
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
    ) -> IndicatorResult:
        """Generate signal from RSI.

        Args:
            data: Price data
            period: RSI period
            overbought: Overbought threshold
            oversold: Oversold threshold

        Returns:
            Indicator result with signal
        """
        rsi = self.indicators.rsi(data, period)
        current_rsi = rsi[-1]

        if np.isnan(current_rsi):
            return IndicatorResult(name="RSI", value=0, signal="neutral", strength=0)

        if current_rsi < oversold:
            signal = "bullish"
            strength = (oversold - current_rsi) / oversold
        elif current_rsi > overbought:
            signal = "bearish"
            strength = (current_rsi - overbought) / (100 - overbought)
        else:
            signal = "neutral"
            strength = 0

        return IndicatorResult(
            name="RSI",
            value=float(current_rsi),
            signal=signal,
            strength=min(1.0, strength),
        )

    def macd_signal(self, data: Sequence[float]) -> IndicatorResult:
        """Generate signal from MACD crossover.

        Args:
            data: Price data

        Returns:
            Indicator result with signal
        """
        macd_line, signal_line, histogram = self.indicators.macd(data)

        if len(histogram) < 2 or np.isnan(histogram[-1]) or np.isnan(histogram[-2]):
            return IndicatorResult(name="MACD", value=0, signal="neutral", strength=0)

        current_hist = histogram[-1]
        prev_hist = histogram[-2]

        # Detect crossover
        if prev_hist < 0 and current_hist > 0:
            signal = "bullish"
            strength = min(1.0, abs(current_hist) * 10)
        elif prev_hist > 0 and current_hist < 0:
            signal = "bearish"
            strength = min(1.0, abs(current_hist) * 10)
        elif current_hist > 0:
            signal = "bullish"
            strength = min(0.5, abs(current_hist) * 5)
        else:
            signal = "bearish"
            strength = min(0.5, abs(current_hist) * 5)

        return IndicatorResult(
            name="MACD",
            value=float(current_hist),
            signal=signal,
            strength=strength,
        )

    def bollinger_signal(
        self,
        data: Sequence[float],
        period: int = 20,
        std_dev: float = 2.0,
    ) -> IndicatorResult:
        """Generate signal from Bollinger Bands.

        Args:
            data: Price data
            period: BB period
            std_dev: Standard deviation multiplier

        Returns:
            Indicator result with signal
        """
        upper, middle, lower = self.indicators.bollinger_bands(data, period, std_dev)
        current_price = data[-1]

        if np.isnan(upper[-1]) or np.isnan(lower[-1]):
            return IndicatorResult(name="BB", value=0, signal="neutral", strength=0)

        band_width = upper[-1] - lower[-1]
        position = (current_price - lower[-1]) / band_width if band_width > 0 else 0.5

        if position < 0.2:
            signal = "bullish"
            strength = 1 - (position / 0.2)
        elif position > 0.8:
            signal = "bearish"
            strength = (position - 0.8) / 0.2
        else:
            signal = "neutral"
            strength = 0

        return IndicatorResult(
            name="BB",
            value=float(position),
            signal=signal,
            strength=min(1.0, strength),
        )

    def trend_signal(
        self,
        data: Sequence[float],
        short_period: int = 9,
        long_period: int = 21,
    ) -> IndicatorResult:
        """Generate signal from EMA crossover.

        Args:
            data: Price data
            short_period: Short EMA period
            long_period: Long EMA period

        Returns:
            Indicator result with signal
        """
        trend = self.indicators.calculate_trend(data, short_period, long_period)

        short_ema = self.indicators.ema(data, short_period)
        long_ema = self.indicators.ema(data, long_period)

        if np.isnan(short_ema[-1]) or np.isnan(long_ema[-1]):
            return IndicatorResult(name="Trend", value=0, signal="neutral", strength=0)

        diff_percent = (short_ema[-1] - long_ema[-1]) / long_ema[-1] * 100
        strength = min(1.0, abs(diff_percent) / 5)  # 5% difference = full strength

        return IndicatorResult(
            name="Trend",
            value=float(diff_percent),
            signal=trend,
            strength=strength,
        )

    def combined_signal(
        self,
        data: Sequence[float],
        _: Sequence[float] | None = None,
        __: Sequence[float] | None = None,
    ) -> dict[str, IndicatorResult]:
        """Generate combined signals from multiple indicators.

        Args:
            data: Close price data
            _: High prices (optional, reserved for future use)
            __: Low prices (optional, reserved for future use)

        Returns:
            Dictionary of indicator results
        """
        signals = {
            "rsi": self.rsi_signal(data),
            "macd": self.macd_signal(data),
            "bollinger": self.bollinger_signal(data),
            "trend": self.trend_signal(data),
        }

        return signals

    def get_overall_signal(
        self,
        data: Sequence[float],
        weights: dict[str, float] | None = None,
    ) -> tuple[str, float]:
        """Get overall trading signal with confidence.

        Args:
            data: Price data
            weights: Optional weight for each indicator

        Returns:
            Tuple of (signal, confidence)
        """
        if weights is None:
            weights = {
                "rsi": 0.25,
                "macd": 0.30,
                "bollinger": 0.20,
                "trend": 0.25,
            }

        signals = self.combined_signal(data)

        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0

        for name, result in signals.items():
            weight = weights.get(name, 0.25)
            total_weight += weight

            if result.signal == "bullish":
                bullish_score += weight * result.strength
            elif result.signal == "bearish":
                bearish_score += weight * result.strength

        if total_weight > 0:
            bullish_score /= total_weight
            bearish_score /= total_weight

        if bullish_score > bearish_score and bullish_score > 0.3:
            return "bullish", bullish_score
        elif bearish_score > bullish_score and bearish_score > 0.3:
            return "bearish", bearish_score
        else:
            return "neutral", max(0, 1 - abs(bullish_score - bearish_score))
