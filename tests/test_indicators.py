"""Tests for technical indicators."""

import numpy as np
import pytest

from kucoin_bot.indicators.technical import SignalGenerator, TechnicalIndicators


class TestTechnicalIndicators:
    """Tests for technical indicator calculations."""

    @pytest.fixture
    def sample_data(self) -> list[float]:
        """Generate sample price data."""
        return [
            100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
            110, 108, 106, 104, 102, 103, 105, 107, 109, 111,
            113, 115, 114, 112, 110, 108, 110, 112, 114, 116,
        ]

    def test_sma(self, sample_data: list[float]) -> None:
        """Test SMA calculation."""
        sma = TechnicalIndicators.sma(sample_data, period=5)
        assert len(sma) == len(sample_data)
        assert np.isnan(sma[0])  # First values are NaN
        assert not np.isnan(sma[4])  # Fifth value should exist
        # Check calculation for known position
        expected = sum(sample_data[0:5]) / 5
        assert abs(sma[4] - expected) < 0.001

    def test_ema(self, sample_data: list[float]) -> None:
        """Test EMA calculation."""
        ema = TechnicalIndicators.ema(sample_data, period=5)
        assert len(ema) == len(sample_data)
        assert not np.isnan(ema[-1])

    def test_rsi(self, sample_data: list[float]) -> None:
        """Test RSI calculation."""
        rsi = TechnicalIndicators.rsi(sample_data, period=14)
        assert len(rsi) == len(sample_data)
        # RSI should be between 0 and 100
        valid_rsi = [r for r in rsi if not np.isnan(r)]
        assert all(0 <= r <= 100 for r in valid_rsi)

    def test_macd(self, sample_data: list[float]) -> None:
        """Test MACD calculation."""
        macd_line, signal_line, histogram = TechnicalIndicators.macd(sample_data)
        assert len(macd_line) == len(sample_data)
        assert len(signal_line) == len(sample_data)
        assert len(histogram) == len(sample_data)

    def test_bollinger_bands(self, sample_data: list[float]) -> None:
        """Test Bollinger Bands calculation."""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(sample_data, period=10)
        assert len(upper) == len(sample_data)
        assert len(middle) == len(sample_data)
        assert len(lower) == len(sample_data)
        # Upper should always be > middle > lower
        valid_indices = [i for i in range(len(upper)) if not np.isnan(upper[i])]
        for i in valid_indices:
            assert upper[i] >= middle[i] >= lower[i]

    def test_stochastic(self) -> None:
        """Test Stochastic oscillator calculation."""
        high = [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
        low = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
        close = [103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]

        k, d = TechnicalIndicators.stochastic(high, low, close)
        assert len(k) == len(close)
        assert len(d) == len(close)

    def test_calculate_trend(self, sample_data: list[float]) -> None:
        """Test trend calculation."""
        trend = TechnicalIndicators.calculate_trend(sample_data)
        assert trend in ["bullish", "bearish", "neutral"]

    def test_fibonacci_levels(self) -> None:
        """Test Fibonacci level calculation."""
        levels = TechnicalIndicators.fibonacci_levels(high=100.0, low=80.0)
        assert levels["0.0"] == 100.0
        assert levels["1.0"] == 80.0
        assert levels["0.5"] == 90.0
        assert abs(levels["0.618"] - 87.64) < 0.01


class TestSignalGenerator:
    """Tests for signal generation."""

    @pytest.fixture
    def generator(self) -> SignalGenerator:
        """Create signal generator instance."""
        return SignalGenerator()

    @pytest.fixture
    def trending_up_data(self) -> list[float]:
        """Generate uptrending price data."""
        return [float(i) for i in range(50, 100)]

    @pytest.fixture
    def trending_down_data(self) -> list[float]:
        """Generate downtrending price data."""
        return [float(i) for i in range(100, 50, -1)]

    def test_rsi_signal_oversold(self, generator: SignalGenerator) -> None:
        """Test RSI signal in oversold condition."""
        # Create oversold condition - prices dropping rapidly
        data = [100 - i * 2 for i in range(30)]
        result = generator.rsi_signal(data)
        assert result.name == "RSI"
        assert result.signal == "bullish"

    def test_rsi_signal_overbought(self, generator: SignalGenerator) -> None:
        """Test RSI signal in overbought condition."""
        # Create overbought condition - prices rising rapidly
        data = [100 + i * 2 for i in range(30)]
        result = generator.rsi_signal(data)
        assert result.name == "RSI"
        assert result.signal == "bearish"

    def test_trend_signal(
        self,
        generator: SignalGenerator,
        trending_up_data: list[float],
    ) -> None:
        """Test trend signal generation."""
        result = generator.trend_signal(trending_up_data)
        assert result.name == "Trend"
        assert result.signal == "bullish"

    def test_combined_signal(
        self,
        generator: SignalGenerator,
        trending_up_data: list[float],
    ) -> None:
        """Test combined signal generation."""
        signals = generator.combined_signal(trending_up_data)
        assert "rsi" in signals
        assert "macd" in signals
        assert "bollinger" in signals
        assert "trend" in signals

    def test_overall_signal(
        self,
        generator: SignalGenerator,
        trending_up_data: list[float],
    ) -> None:
        """Test overall signal calculation."""
        signal, confidence = generator.get_overall_signal(trending_up_data)
        assert signal in ["bullish", "bearish", "neutral"]
        assert 0 <= confidence <= 1
