"""Data models for the KuCoin trading bot using Pydantic."""

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TimeInForce(str, Enum):
    """Time in force options for orders."""

    GTC = "GTC"  # Good Till Cancel
    GTT = "GTT"  # Good Till Time
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


class SignalType(str, Enum):
    """Trading signal types."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


class Ticker(BaseModel):
    """Market ticker data model."""

    symbol: str
    price: Annotated[Decimal, Field(gt=0)]
    best_bid: Annotated[Decimal, Field(gt=0)]
    best_ask: Annotated[Decimal, Field(gt=0)]
    volume_24h: Annotated[Decimal, Field(ge=0)]
    change_24h: Decimal
    high_24h: Annotated[Decimal, Field(gt=0)]
    low_24h: Annotated[Decimal, Field(gt=0)]
    timestamp: datetime

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.best_ask - self.best_bid

    @property
    def spread_percent(self) -> Decimal:
        """Calculate spread as percentage of mid price."""
        mid_price = (self.best_bid + self.best_ask) / 2
        if mid_price == 0:
            return Decimal("0")
        return (self.spread / mid_price) * 100


class Candle(BaseModel):
    """OHLCV candle data model."""

    symbol: str
    timestamp: datetime
    open: Annotated[Decimal, Field(gt=0)]
    high: Annotated[Decimal, Field(gt=0)]
    low: Annotated[Decimal, Field(gt=0)]
    close: Annotated[Decimal, Field(gt=0)]
    volume: Annotated[Decimal, Field(ge=0)]

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish."""
        return self.close < self.open

    @property
    def body_size(self) -> Decimal:
        """Calculate candle body size."""
        return abs(self.close - self.open)

    @property
    def wick_size(self) -> Decimal:
        """Calculate total wick size."""
        upper_wick = self.high - max(self.open, self.close)
        lower_wick = min(self.open, self.close) - self.low
        return upper_wick + lower_wick


class Order(BaseModel):
    """Order data model."""

    id: str | None = None
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: Annotated[Decimal | None, Field(default=None, gt=0)]
    size: Annotated[Decimal, Field(gt=0)]
    stop_price: Annotated[Decimal | None, Field(default=None, gt=0)]
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    filled_size: Annotated[Decimal, Field(default=Decimal("0"), ge=0)]
    filled_price: Annotated[Decimal | None, Field(default=None, ge=0)]
    fee: Annotated[Decimal, Field(default=Decimal("0"), ge=0)]
    fee_currency: str = "USDT"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime | None = None

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def remaining_size(self) -> Decimal:
        """Calculate remaining order size."""
        return self.size - self.filled_size

    @property
    def fill_rate(self) -> Decimal:
        """Calculate fill rate as percentage."""
        if self.size == 0:
            return Decimal("0")
        return (self.filled_size / self.size) * 100


class Position(BaseModel):
    """Trading position data model."""

    symbol: str
    side: OrderSide
    entry_price: Annotated[Decimal, Field(gt=0)]
    size: Annotated[Decimal, Field(gt=0)]
    current_price: Annotated[Decimal, Field(gt=0)]
    stop_loss: Annotated[Decimal | None, Field(default=None, gt=0)]
    take_profit: Annotated[Decimal | None, Field(default=None, gt=0)]
    opened_at: datetime = Field(default_factory=utc_now)
    order_ids: list[str] = Field(default_factory=list)

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L."""
        if self.side == OrderSide.BUY:
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.current_price) * self.size

    @property
    def unrealized_pnl_percent(self) -> Decimal:
        """Calculate unrealized P&L as percentage."""
        if self.entry_price == 0:
            return Decimal("0")
        return (self.unrealized_pnl / (self.entry_price * self.size)) * 100

    @property
    def position_value(self) -> Decimal:
        """Calculate current position value."""
        return self.current_price * self.size


class TradingSignal(BaseModel):
    """Trading signal from strategy."""

    symbol: str
    signal_type: SignalType
    confidence: Annotated[float, Field(ge=0, le=1)]
    price: Annotated[Decimal, Field(gt=0)]
    timestamp: datetime = Field(default_factory=utc_now)
    indicators: dict[str, float] = Field(default_factory=dict)
    reason: str = ""

    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (not HOLD)."""
        return self.signal_type != SignalType.HOLD


class AccountBalance(BaseModel):
    """Account balance data model."""

    currency: str
    available: Annotated[Decimal, Field(ge=0)]
    holds: Annotated[Decimal, Field(ge=0)]
    total: Annotated[Decimal, Field(ge=0)]

    @property
    def usage_percent(self) -> Decimal:
        """Calculate percentage of balance in use."""
        if self.total == 0:
            return Decimal("0")
        return (self.holds / self.total) * 100


class Trade(BaseModel):
    """Completed trade record."""

    id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: Annotated[Decimal, Field(gt=0)]
    size: Annotated[Decimal, Field(gt=0)]
    fee: Annotated[Decimal, Field(ge=0)]
    fee_currency: str
    timestamp: datetime
    is_maker: bool

    @property
    def value(self) -> Decimal:
        """Calculate trade value."""
        return self.price * self.size

    @property
    def net_value(self) -> Decimal:
        """Calculate net value after fees."""
        if self.side == OrderSide.BUY:
            return self.value + self.fee
        return self.value - self.fee


class MarketDepth(BaseModel):
    """Order book depth data."""

    symbol: str
    bids: list[tuple[Decimal, Decimal]]  # (price, size)
    asks: list[tuple[Decimal, Decimal]]  # (price, size)
    timestamp: datetime

    @property
    def best_bid(self) -> Decimal | None:
        """Get best bid price."""
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        """Get best ask price."""
        return self.asks[0][0] if self.asks else None

    @property
    def mid_price(self) -> Decimal | None:
        """Calculate mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    def get_bid_depth(self, levels: int = 5) -> Decimal:
        """Calculate bid depth for given levels."""
        return sum((size for _, size in self.bids[:levels]), Decimal("0"))

    def get_ask_depth(self, levels: int = 5) -> Decimal:
        """Calculate ask depth for given levels."""
        return sum((size for _, size in self.asks[:levels]), Decimal("0"))


class PerformanceMetrics(BaseModel):
    """Trading performance metrics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    average_win: Decimal = Decimal("0")
    average_loss: Decimal = Decimal("0")
    profit_factor: float = 0.0
    start_balance: Decimal = Decimal("0")
    current_balance: Decimal = Decimal("0")
    period_start: datetime | None = None
    period_end: datetime | None = None

    @property
    def return_percent(self) -> Decimal:
        """Calculate return percentage."""
        if self.start_balance == 0:
            return Decimal("0")
        return ((self.current_balance - self.start_balance) / self.start_balance) * 100

    def calculate_win_rate(self) -> None:
        """Calculate and update win rate."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
