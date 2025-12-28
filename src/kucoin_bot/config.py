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

    # Auto-selection settings
    auto_select_pairs: Annotated[bool, Field(default=False, description="Enable auto-selection of best trading pairs")]
    auto_select_count: Annotated[int, Field(default=5, gt=0, le=20, description="Number of top pairs to auto-select")]
    auto_select_interval: Annotated[int, Field(default=3600, gt=60, le=86400, description="Pair scan interval in seconds (60s - 24h)")]
    auto_select_min_volume: Annotated[float, Field(default=100000.0, gt=0, description="Minimum 24h volume for pair selection")]
    auto_select_min_signal: Annotated[float, Field(default=0.5, ge=0, le=1, description="Minimum signal strength for pair selection")]
    auto_select_signal_type: Annotated[str, Field(default="any", description="Signal type filter: any, bullish, or bearish")]
    auto_select_strategy: Annotated[bool, Field(default=False, description="Enable automatic strategy selection based on market conditions")]

    # Pair scoring weights for composite score calculation
    pair_score_signal_weight: Annotated[float, Field(default=0.6, ge=0, le=1, description="Weight for signal strength in composite score")]
    pair_score_volume_weight: Annotated[float, Field(default=0.25, ge=0, le=1, description="Weight for volume in composite score")]
    pair_score_volatility_weight: Annotated[float, Field(default=0.15, ge=0, le=1, description="Weight for volatility in composite score")]
    pair_score_volume_threshold: Annotated[float, Field(default=1000000.0, gt=0, description="Volume baseline for scoring normalization (USDT)")]

    @field_validator("trading_pairs")
    @classmethod
    def validate_trading_pairs(cls, v: str) -> str:
        """Validate trading pairs format."""
        pairs = [p.strip().upper() for p in v.split(",")]
        for pair in pairs:
            if "-" not in pair:
                raise ValueError(f"Invalid trading pair format: {pair}. Expected format: BASE-QUOTE")
        return ",".join(pairs)

    @field_validator("auto_select_signal_type")
    @classmethod
    def validate_auto_select_signal_type(cls, v: str) -> str:
        """Validate auto-select signal type."""
        valid_types = ["any", "bullish", "bearish"]
        v = v.lower().strip()
        if v not in valid_types:
            raise ValueError(f"Invalid signal type: {v}. Must be one of: {valid_types}")
        return v

    @field_validator("pair_score_volatility_weight")
    @classmethod
    def validate_weights_sum(cls, v: float, info: ValidationInfo) -> float:
        """Validate that scoring weights sum to approximately 1.0."""
        if info.data:
            signal_weight = info.data.get("pair_score_signal_weight", 0.6)
            volume_weight = info.data.get("pair_score_volume_weight", 0.25)
            volatility_weight = v
            total = signal_weight + volume_weight + volatility_weight
            if abs(total - 1.0) > 0.01:  # Allow small floating point error
                raise ValueError(
                    f"Scoring weights must sum to 1.0. Current sum: {total:.2f} "
                    f"(signal={signal_weight}, volume={volume_weight}, volatility={volatility_weight})"
                )
        return v

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

    # Smart leverage settings
    max_leverage: Annotated[int, Field(default=20, gt=0, le=100, description="Maximum leverage multiplier")]
    target_risk_percent: Annotated[float, Field(default=2.0, gt=0, le=10, description="Target risk per trade as percentage")]
    leverage_aggression_multiplier: Annotated[int, Field(default=10, gt=0, le=50, description="Aggression multiplier for leverage calculation")]


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
