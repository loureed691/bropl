"""Tests for pair selector functionality."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from kucoin_bot.pair_selector.selector import PairScore, PairSelector


class TestPairScore:
    """Tests for PairScore dataclass."""

    def test_pair_score_creation(self) -> None:
        """Test creating a PairScore instance."""
        score = PairScore(
            symbol="BTC-USDT",
            signal_type="bullish",
            signal_strength=0.8,
            volume_24h=Decimal("1000000"),
            price_change_24h=2.5,
            volatility=0.03,
        )

        assert score.symbol == "BTC-USDT"
        assert score.signal_type == "bullish"
        assert score.signal_strength == 0.8
        assert score.volume_24h == Decimal("1000000")
        assert score.price_change_24h == 2.5
        assert score.volatility == 0.03

    def test_composite_score_calculation(self) -> None:
        """Test composite score calculation."""
        score = PairScore(
            symbol="BTC-USDT",
            signal_type="bullish",
            signal_strength=0.8,
            volume_24h=Decimal("1000000"),  # High volume
            price_change_24h=2.5,
            volatility=0.03,  # Optimal volatility
        )

        # Composite score should be weighted combination
        composite = score.composite_score
        assert 0 <= composite <= 1
        assert composite > 0.5  # Strong signal should result in high score

    def test_composite_score_low_volume(self) -> None:
        """Test composite score with low volume."""
        score = PairScore(
            symbol="BTC-USDT",
            signal_type="bullish",
            signal_strength=0.8,
            volume_24h=Decimal("10000"),  # Low volume
            price_change_24h=2.5,
            volatility=0.03,
        )

        composite = score.composite_score
        assert composite < 0.8  # Should be lower due to low volume

    def test_composite_score_high_volatility(self) -> None:
        """Test composite score with high volatility."""
        score = PairScore(
            symbol="BTC-USDT",
            signal_type="bullish",
            signal_strength=0.8,
            volume_24h=Decimal("1000000"),
            price_change_24h=2.5,
            volatility=0.10,  # Very high volatility
        )

        composite = score.composite_score
        # High volatility should reduce the score
        assert composite < 0.85


class TestPairSelector:
    """Tests for PairSelector class."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock KuCoin client."""
        client = MagicMock()
        client.get_symbols = AsyncMock(
            return_value=[
                {"symbol": "BTC-USDT", "quoteCurrency": "USDT", "enableTrading": True},
                {"symbol": "ETH-USDT", "quoteCurrency": "USDT", "enableTrading": True},
                {"symbol": "XRP-USDT", "quoteCurrency": "USDT", "enableTrading": True},
                {"symbol": "BTC-BTC", "quoteCurrency": "BTC", "enableTrading": True},
            ]
        )
        client.get_24h_stats = AsyncMock(
            return_value={"volValue": "500000", "changeRate": "0.025"}
        )
        client.get_candles = AsyncMock(return_value=[])
        return client

    @pytest.fixture
    def selector(self, mock_client: MagicMock) -> PairSelector:
        """Create a PairSelector instance with mock client."""
        return PairSelector(
            client=mock_client,
            min_volume_24h=Decimal("100000"),
            min_signal_strength=0.5,
            top_pairs_count=5,
        )

    def test_selector_initialization(self, selector: PairSelector) -> None:
        """Test PairSelector initialization."""
        assert selector.min_volume_24h == Decimal("100000")
        assert selector.min_signal_strength == 0.5
        assert selector.top_pairs_count == 5

    async def test_get_usdt_pairs(self, selector: PairSelector) -> None:
        """Test getting USDT trading pairs."""
        pairs = await selector.get_usdt_pairs()

        assert len(pairs) == 3
        assert "BTC-USDT" in pairs
        assert "ETH-USDT" in pairs
        assert "XRP-USDT" in pairs
        assert "BTC-BTC" not in pairs  # Not a USDT pair

    async def test_get_usdt_pairs_error(
        self, selector: PairSelector, mock_client: MagicMock
    ) -> None:
        """Test handling errors when getting USDT pairs."""
        mock_client.get_symbols.side_effect = Exception("API Error")
        pairs = await selector.get_usdt_pairs()
        assert pairs == []

    async def test_analyze_pair_low_volume(
        self, selector: PairSelector, mock_client: MagicMock
    ) -> None:
        """Test analyzing a pair with insufficient volume."""
        mock_client.get_24h_stats.return_value = {
            "volValue": "1000",  # Below minimum
            "changeRate": "0.025",
        }

        score = await selector.analyze_pair("BTC-USDT")
        assert score is None  # Filtered out due to low volume

    async def test_get_cached_scores(self, selector: PairSelector) -> None:
        """Test getting cached scores."""
        # Initially empty
        assert selector.get_cached_scores() == []

    def test_get_pair_details_not_found(self, selector: PairSelector) -> None:
        """Test getting details for a pair not in cache."""
        details = selector.get_pair_details("UNKNOWN-USDT")
        assert details is None

    def test_get_scan_summary_empty(self, selector: PairSelector) -> None:
        """Test scan summary with no cached scores."""
        summary = selector.get_scan_summary()

        assert summary["last_scan"] is None
        assert summary["total_pairs"] == 0
        assert summary["bullish_count"] == 0
        assert summary["bearish_count"] == 0
        assert summary["neutral_count"] == 0
        assert summary["top_pairs"] == []

    async def test_scan_all_pairs_with_rate_limiting(
        self, selector: PairSelector, mock_client: MagicMock
    ) -> None:
        """Test that scan_all_pairs respects rate limiting."""
        # Mock to return valid data
        from decimal import Decimal

        mock_candles = [
            MagicMock(close=Decimal(str(100 + i)), high=Decimal(str(105 + i)), low=Decimal(str(95 + i)))
            for i in range(60)
        ]
        mock_client.get_candles.return_value = mock_candles
        mock_client.get_24h_stats.return_value = {"volValue": "500000", "changeRate": "0.025"}

        _ = await selector.scan_all_pairs(max_concurrent=2)

        # Should have called get_24h_stats for each USDT pair
        assert mock_client.get_24h_stats.call_count == 3

    async def test_analyze_pair_insufficient_candles(
        self, selector: PairSelector, mock_client: MagicMock
    ) -> None:
        """Test analyzing a pair with insufficient historical data."""
        # Return less than 50 candles
        mock_candles = [
            MagicMock(close=Decimal(str(100 + i)), high=Decimal(str(105 + i)), low=Decimal(str(95 + i)))
            for i in range(20)
        ]
        mock_client.get_candles.return_value = mock_candles
        mock_client.get_24h_stats.return_value = {"volValue": "500000", "changeRate": "0.025"}

        score = await selector.analyze_pair("BTC-USDT")
        assert score is None  # Filtered out due to insufficient candles

    async def test_get_bullish_pairs(
        self, selector: PairSelector, mock_client: MagicMock
    ) -> None:
        """Test getting bullish pairs filter."""
        # Mock to return empty candles so no pairs qualify
        mock_client.get_candles.return_value = []

        pairs = await selector.get_bullish_pairs(count=3)
        # With empty candle data, no pairs should qualify
        assert isinstance(pairs, list)

    async def test_get_bearish_pairs(
        self, selector: PairSelector, mock_client: MagicMock
    ) -> None:
        """Test getting bearish pairs filter."""
        # Mock to return empty candles so no pairs qualify
        mock_client.get_candles.return_value = []

        pairs = await selector.get_bearish_pairs(count=3)
        # With empty candle data, no pairs should qualify
        assert isinstance(pairs, list)


class TestTradingSettingsAutoSelect:
    """Tests for auto-selection settings in TradingSettings."""

    def test_default_auto_select_settings(self) -> None:
        """Test default auto-selection settings."""
        from kucoin_bot.config import TradingSettings

        settings = TradingSettings()

        assert settings.auto_select_pairs is False
        assert settings.auto_select_count == 5
        assert settings.auto_select_interval == 3600
        assert settings.auto_select_min_volume == 100000.0
        assert settings.auto_select_min_signal == 0.5
        assert settings.auto_select_signal_type == "any"

    def test_auto_select_signal_type_validation(self) -> None:
        """Test validation of auto_select_signal_type."""
        from pydantic import ValidationError

        from kucoin_bot.config import TradingSettings

        # Valid values
        for valid_type in ["any", "bullish", "bearish"]:
            settings = TradingSettings(auto_select_signal_type=valid_type)
            assert settings.auto_select_signal_type == valid_type

        # Invalid value
        with pytest.raises(ValidationError):
            TradingSettings(auto_select_signal_type="invalid")

    def test_auto_select_enabled(self) -> None:
        """Test enabling auto-selection."""
        from kucoin_bot.config import TradingSettings

        settings = TradingSettings(
            auto_select_pairs=True,
            auto_select_count=10,
            auto_select_signal_type="bullish",
        )

        assert settings.auto_select_pairs is True
        assert settings.auto_select_count == 10
        assert settings.auto_select_signal_type == "bullish"

    def test_auto_select_count_limits(self) -> None:
        """Test auto_select_count limits."""
        from pydantic import ValidationError

        from kucoin_bot.config import TradingSettings

        # Valid counts
        settings = TradingSettings(auto_select_count=1)
        assert settings.auto_select_count == 1

        settings = TradingSettings(auto_select_count=20)
        assert settings.auto_select_count == 20

        # Invalid counts
        with pytest.raises(ValidationError):
            TradingSettings(auto_select_count=0)

        with pytest.raises(ValidationError):
            TradingSettings(auto_select_count=21)

    def test_auto_select_interval_limits(self) -> None:
        """Test auto_select_interval limits."""
        from pydantic import ValidationError

        from kucoin_bot.config import TradingSettings

        # Valid intervals
        settings = TradingSettings(auto_select_interval=61)
        assert settings.auto_select_interval == 61

        settings = TradingSettings(auto_select_interval=86400)  # 24 hours max
        assert settings.auto_select_interval == 86400

        # Invalid intervals
        with pytest.raises(ValidationError):
            TradingSettings(auto_select_interval=60)  # Must be > 60

        with pytest.raises(ValidationError):
            TradingSettings(auto_select_interval=86401)  # Must be <= 86400


class TestPairScoreRanking:
    """Tests for pair score ranking and comparison."""

    def test_scores_sorting(self) -> None:
        """Test that scores can be sorted by composite score."""
        scores = [
            PairScore(
                symbol="LOW-USDT",
                signal_type="neutral",
                signal_strength=0.3,
                volume_24h=Decimal("100000"),
                price_change_24h=0.5,
                volatility=0.01,
            ),
            PairScore(
                symbol="HIGH-USDT",
                signal_type="bullish",
                signal_strength=0.9,
                volume_24h=Decimal("1000000"),
                price_change_24h=5.0,
                volatility=0.03,
            ),
            PairScore(
                symbol="MID-USDT",
                signal_type="bullish",
                signal_strength=0.6,
                volume_24h=Decimal("500000"),
                price_change_24h=2.0,
                volatility=0.02,
            ),
        ]

        sorted_scores = sorted(scores, key=lambda x: x.composite_score, reverse=True)

        assert sorted_scores[0].symbol == "HIGH-USDT"
        assert sorted_scores[2].symbol == "LOW-USDT"

    def test_timestamp_default(self) -> None:
        """Test that timestamp is set automatically."""
        before = datetime.now(UTC)
        score = PairScore(
            symbol="BTC-USDT",
            signal_type="bullish",
            signal_strength=0.8,
            volume_24h=Decimal("1000000"),
            price_change_24h=2.5,
            volatility=0.03,
        )
        after = datetime.now(UTC)

        assert before <= score.timestamp <= after
