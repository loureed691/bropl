"""Configuration settings for the KuCoin trading bot using Pydantic Settings."""

from enum import Enum
from typing import Annotated

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Environment types for the trading bot."""

    PRODUCTION = "production"
    SANDBOX = "sandbox"
    DEVELOPMENT = "development"


class StrategyName(str, Enum):
    """Available trading strategies."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    GRID = "grid"
    DCA = "dca"
    SCALPING = "scalping"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class KuCoinSettings(BaseSettings):
    """KuCoin API configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="KUCOIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr = Field(default=SecretStr(""), description="KuCoin API key")
    api_secret: SecretStr = Field(default=SecretStr(""), description="KuCoin API secret")
    api_passphrase: SecretStr = Field(default=SecretStr(""), description="KuCoin API passphrase")


class TradingSettings(BaseSettings):
    """Trading configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    trading_pairs: Annotated[str, Field(default="BTC-USDT", description="Comma-separated trading pairs")]
    base_order_size: Annotated[float, Field(default=100.0, gt=0, description="Base order size in quote currency")]
    max_position_size: Annotated[float, Field(default=1000.0, gt=0, description="Maximum position size")]
    max_daily_trades: Annotated[int, Field(default=50, gt=0, description="Maximum daily trades")]

    @field_validator("trading_pairs")
    @classmethod
    def validate_trading_pairs(cls, v: str) -> str:
        """Validate trading pairs format."""
        pairs = [p.strip().upper() for p in v.split(",")]
        for pair in pairs:
            if "-" not in pair:
                raise ValueError(f"Invalid trading pair format: {pair}. Expected format: BASE-QUOTE")
        return ",".join(pairs)

    def get_pairs_list(self) -> list[str]:
        """Return trading pairs as a list."""
        return [p.strip() for p in self.trading_pairs.split(",")]


class RiskSettings(BaseSettings):
    """Risk management settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_drawdown_percent: Annotated[float, Field(default=5.0, gt=0, le=100, description="Maximum drawdown percentage")]
    stop_loss_percent: Annotated[float, Field(default=2.0, gt=0, le=100, description="Stop loss percentage")]
    take_profit_percent: Annotated[float, Field(default=3.0, gt=0, le=100, description="Take profit percentage")]
    max_open_positions: Annotated[int, Field(default=5, gt=0, description="Maximum open positions")]


class StrategySettings(BaseSettings):
    """Strategy configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    strategy_name: Annotated[StrategyName, Field(default=StrategyName.MOMENTUM)]
    rsi_period: Annotated[int, Field(default=14, gt=0, description="RSI calculation period")]
    rsi_overbought: Annotated[float, Field(default=70.0, gt=50, le=100, description="RSI overbought threshold")]
    rsi_oversold: Annotated[float, Field(default=30.0, ge=0, lt=50, description="RSI oversold threshold")]
    ema_short_period: Annotated[int, Field(default=9, gt=0, description="Short EMA period")]
    ema_long_period: Annotated[int, Field(default=21, gt=0, description="Long EMA period")]

    @field_validator("ema_long_period")
    @classmethod
    def validate_ema_periods(cls, v: int, info: ValidationInfo) -> int:
        """Validate EMA periods - long must be greater than short."""
        if info.data and "ema_short_period" in info.data and v <= info.data["ema_short_period"]:
            raise ValueError("ema_long_period must be greater than ema_short_period")
        return v


class AppSettings(BaseSettings):
    """Application-wide settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Annotated[Environment, Field(default=Environment.SANDBOX)]
    use_sandbox: Annotated[bool, Field(default=True, description="Use KuCoin sandbox environment")]
    log_level: Annotated[LogLevel, Field(default=LogLevel.INFO)]
    log_format: Annotated[str, Field(default="json", description="Log format: json or text")]


class Settings(BaseSettings):
    """Main settings container combining all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    kucoin: KuCoinSettings = Field(default_factory=KuCoinSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)  # type: ignore[arg-type]
    risk: RiskSettings = Field(default_factory=RiskSettings)  # type: ignore[arg-type]
    strategy: StrategySettings = Field(default_factory=StrategySettings)  # type: ignore[arg-type]
    app: AppSettings = Field(default_factory=AppSettings)  # type: ignore[arg-type]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app.environment == Environment.PRODUCTION

    @property
    def base_url(self) -> str:
        """Get the appropriate KuCoin API base URL."""
        if self.app.use_sandbox:
            return "https://openapi-sandbox.kucoin.com"
        return "https://api.kucoin.com"

    @property
    def ws_url(self) -> str:
        """Get the appropriate KuCoin WebSocket URL."""
        if self.app.use_sandbox:
            return "wss://push-sandbox.kucoin.com"
        return "wss://push-private.kucoin.com"


def get_settings() -> Settings:
    """Factory function to create settings instance."""
    return Settings()
