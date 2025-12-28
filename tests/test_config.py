"""Tests for configuration settings."""

import pytest
from pydantic import ValidationError

from kucoin_bot.config import (
    AppSettings,
    Environment,
    KuCoinSettings,
    RiskSettings,
    Settings,
    StrategyName,
    StrategySettings,
    TradingSettings,
)


class TestKuCoinSettings:
    """Tests for KuCoin API settings."""

    def test_default_settings(self) -> None:
        """Test default KuCoin settings."""
        settings = KuCoinSettings()
        assert settings.api_key.get_secret_value() == ""
        assert settings.api_secret.get_secret_value() == ""
        assert settings.api_passphrase.get_secret_value() == ""


class TestTradingSettings:
    """Tests for trading settings."""

    def test_default_settings(self) -> None:
        """Test default trading settings."""
        settings = TradingSettings()
        assert settings.trading_pairs == "BTC-USDT"
        assert settings.base_order_size == 100.0
        assert settings.max_position_size == 1000.0

    def test_get_pairs_list(self) -> None:
        """Test getting pairs as list."""
        settings = TradingSettings(trading_pairs="BTC-USDT,ETH-USDT")
        pairs = settings.get_pairs_list()
        assert pairs == ["BTC-USDT", "ETH-USDT"]

    def test_invalid_pair_format(self) -> None:
        """Test validation of trading pair format."""
        with pytest.raises(ValidationError):
            TradingSettings(trading_pairs="INVALID")


class TestRiskSettings:
    """Tests for risk management settings."""

    def test_default_settings(self) -> None:
        """Test default risk settings."""
        settings = RiskSettings()
        assert settings.max_drawdown_percent == 5.0
        assert settings.stop_loss_percent == 2.0
        assert settings.take_profit_percent == 3.0
        assert settings.max_open_positions == 5

    def test_invalid_drawdown(self) -> None:
        """Test validation of drawdown percentage."""
        with pytest.raises(ValidationError):
            RiskSettings(max_drawdown_percent=150)


class TestStrategySettings:
    """Tests for strategy settings."""

    def test_default_settings(self) -> None:
        """Test default strategy settings."""
        settings = StrategySettings()
        assert settings.strategy_name == StrategyName.MOMENTUM
        assert settings.rsi_period == 14
        assert settings.rsi_overbought == 70.0
        assert settings.rsi_oversold == 30.0

    def test_invalid_rsi_overbought(self) -> None:
        """Test RSI overbought must be > 50."""
        with pytest.raises(ValidationError):
            StrategySettings(rsi_overbought=40)


class TestAppSettings:
    """Tests for application settings."""

    def test_default_settings(self) -> None:
        """Test default app settings."""
        settings = AppSettings()
        assert settings.environment == Environment.SANDBOX
        assert settings.use_sandbox is True


class TestSettings:
    """Tests for main settings container."""

    def test_default_settings(self) -> None:
        """Test default combined settings."""
        settings = Settings()
        assert settings.is_production is False
        assert "sandbox" in settings.base_url
        assert "sandbox" in settings.ws_url

    def test_production_urls(self) -> None:
        """Test production API URLs."""
        settings = Settings()
        settings.app.use_sandbox = False
        assert "sandbox" not in settings.base_url
        assert "sandbox" not in settings.ws_url

def test_weights_validation_sum_to_one() -> None:
    """Test that scoring weights are validated to sum to 1.0."""
    from kucoin_bot.config import TradingSettings
    
    # Valid weights should work
    settings = TradingSettings(
        pair_score_signal_weight=0.6,
        pair_score_volume_weight=0.25,
        pair_score_volatility_weight=0.15,
    )
    assert settings.pair_score_signal_weight == 0.6
    
    # Invalid weights should fail
    import pytest
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError) as exc_info:
        TradingSettings(
            pair_score_signal_weight=0.5,
            pair_score_volume_weight=0.3,
            pair_score_volatility_weight=0.3,  # Sum = 1.1
        )
    
    assert "must sum to 1.0" in str(exc_info.value)
