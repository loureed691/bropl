# KuCoin Trading Bot

A sophisticated, production-ready cryptocurrency trading bot for KuCoin exchange, built with modern 2025 standards and best practices.

## Features

### 🚀 Core Capabilities

- **Async Architecture**: Built entirely on asyncio for high-performance, non-blocking operations
- **Real-time Order Tracking**: WebSocket-based order execution monitoring with polling fallback
- **Auto-Selection of Best Trading Pairs**: Automatically scans and selects USDT pairs with the strongest signals
- **Multiple Trading Strategies**: 
  - Momentum Trading (RSI + EMA crossovers + MACD)
  - Mean Reversion (Bollinger Bands)
  - Grid Trading (for ranging markets)
  - Scalping (high-frequency short-term trades)
  - DCA (Dollar Cost Averaging for accumulation)

### ⚡ Real-time Order Execution Tracking

The bot uses KuCoin's private WebSocket channel (`/spotMarket/tradeOrders`) for real-time order execution updates:

- **Instant Notifications**: Receive immediate updates when orders are filled, partially filled, or canceled
- **No Timeout Limitations**: Unlike polling-based tracking, WebSocket updates ensure orders are tracked regardless of fill time
- **Graceful Fallback**: Automatically falls back to REST API polling if WebSocket connection is unavailable
- **Dual Connection**: Maintains separate WebSocket connections for public market data and private order updates
- **Reliable Tracking**: Orders remain monitored even if they take longer than 30 seconds to fill

### 🔍 Auto Pair Selection

The bot can automatically scan all available USDT trading pairs and select those with the strongest signals:

- **Volume Filtering**: Only considers pairs with sufficient 24-hour volume
- **Signal Analysis**: Uses RSI, MACD, Bollinger Bands, and trend indicators
- **Composite Scoring**: Ranks pairs based on signal strength, volume, and volatility
- **Periodic Re-scanning**: Automatically updates pair selection at configurable intervals
- **Signal Type Filtering**: Can focus on bullish, bearish, or any signals

### 📊 Technical Analysis

- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- EMA/SMA Moving Averages
- Stochastic Oscillator
- ADX (Average Directional Index)
- ATR (Average True Range)
- VWAP (Volume Weighted Average Price)
- Fibonacci Retracement Levels
- Support/Resistance Detection

### 🛡️ Risk Management

- Maximum drawdown protection
- Per-trade stop-loss and take-profit
- Position sizing based on Kelly Criterion
- Daily trade limits
- Maximum open positions limit
- Portfolio exposure tracking

### 📈 Monitoring & Observability

- Prometheus metrics integration
- Structured JSON logging with structlog
- Real-time portfolio tracking
- Trade performance statistics
- Win rate and profit factor calculation

## Installation

### Prerequisites

- Python 3.11 or higher
- KuCoin API credentials

### Setup

1. Clone the repository:
```bash
git clone https://github.com/loureed691/bropl.git
cd bropl
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your KuCoin API credentials
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KUCOIN_API_KEY` | Your KuCoin API key | Required |
| `KUCOIN_API_SECRET` | Your KuCoin API secret | Required |
| `KUCOIN_API_PASSPHRASE` | Your KuCoin API passphrase | Required |
| `TRADING_PAIRS` | Comma-separated trading pairs | `BTC-USDT` |
| `BASE_ORDER_SIZE` | Base order size in quote currency | `100` |
| `MAX_POSITION_SIZE` | Maximum position size | `1000` |
| `MAX_DAILY_TRADES` | Maximum trades per day | `50` |
| `MAX_DRAWDOWN_PERCENT` | Maximum drawdown percentage | `5.0` |
| `STOP_LOSS_PERCENT` | Stop loss percentage | `2.0` |
| `TAKE_PROFIT_PERCENT` | Take profit percentage | `3.0` |
| `MAX_OPEN_POSITIONS` | Maximum concurrent positions | `5` |
| `STRATEGY_NAME` | Trading strategy to use | `momentum` |
| `USE_SANDBOX` | Use KuCoin sandbox environment | `true` |

### Auto-Selection Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTO_SELECT_PAIRS` | Enable auto-selection of best trading pairs | `false` |
| `AUTO_SELECT_COUNT` | Number of top pairs to auto-select (1-20) | `5` |
| `AUTO_SELECT_INTERVAL` | Pair scan interval in seconds | `3600` |
| `AUTO_SELECT_MIN_VOLUME` | Minimum 24h volume for pair selection | `100000` |
| `AUTO_SELECT_MIN_SIGNAL` | Minimum signal strength (0.0-1.0) | `0.5` |
| `AUTO_SELECT_SIGNAL_TYPE` | Signal type filter: `any`, `bullish`, or `bearish` | `any` |
| `AUTO_SELECT_STRATEGY` | Enable automatic strategy selection per pair | `false` |

When `AUTO_SELECT_PAIRS` is enabled, the bot will:
1. Scan all available USDT trading pairs on startup
2. Analyze each pair using technical indicators
3. Rank pairs by a composite score based on signal strength, volume, and volatility
4. Select the top pairs with the strongest signals
5. Periodically re-scan and update the selected pairs

### Automatic Strategy Selection

When `AUTO_SELECT_STRATEGY` is enabled along with `AUTO_SELECT_PAIRS`, the bot automatically chooses the best strategy for each pair based on market conditions:

> **Note:** `AUTO_SELECT_STRATEGY` only takes effect when `AUTO_SELECT_PAIRS` is also enabled. Without pair auto-selection, the bot uses the configured `STRATEGY_NAME` for all pairs.

| Market Condition | Recommended Strategy |
|-----------------|---------------------|
| Strong trend (bullish/bearish) + moderate volatility | `momentum` |
| Neutral/weak signal + low volatility | `grid` |
| High volume + moderate volatility + no strong trend | `scalping` |
| Bear market or high volatility | `dca` |
| Ranging market (neutral signal) | `mean_reversion` |

The strategy selection logic considers:
- **Signal Type**: Bullish, bearish, or neutral market conditions
- **Signal Strength**: How strong the technical indicators are (0.0-1.0)
- **Volatility**: Price movement range (low <2%, moderate 2-5%, high >5%)
- **Volume**: 24-hour trading volume (high >1M USDT)

### Available Strategies

- `momentum`: Uses RSI, EMA crossovers, and MACD for trend following
- `mean_reversion`: Trades Bollinger Band bounces
- `grid`: Places orders at fixed intervals for ranging markets
- `scalping`: High-frequency trading with fast indicators
- `dca`: Dollar cost averaging with dip buying

## Usage

### Running the Bot

```bash
# Start the trading bot
kucoin-bot

# Or run directly with Python
python -m kucoin_bot.main
```

### Programmatic Usage

```python
import asyncio
from kucoin_bot.main import TradingBot
from kucoin_bot.config import Settings

async def main():
    # Create bot with custom settings
    settings = Settings()
    bot = TradingBot(settings)
    
    # Start trading
    await bot.start()

asyncio.run(main())
```

## Project Structure

```
src/kucoin_bot/
├── api/
│   ├── client.py       # KuCoin REST API client
│   └── websocket.py    # WebSocket manager for real-time data
├── execution/
│   └── engine.py       # Order execution engine
├── indicators/
│   └── technical.py    # Technical analysis indicators
├── models/
│   └── data_models.py  # Pydantic data models
├── pair_selector/
│   └── selector.py     # Auto-selection of best trading pairs
├── risk_management/
│   └── manager.py      # Risk management system
├── strategies/
│   └── base.py         # Trading strategies
├── utils/
│   ├── logging.py      # Logging configuration
│   └── metrics.py      # Prometheus metrics
├── config.py           # Configuration settings
└── main.py             # Main bot orchestrator
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=kucoin_bot

# Run specific test file
pytest tests/test_strategies.py -v
```

## Development

### Code Quality

```bash
# Format code
black src tests

# Lint code
ruff check src tests

# Type checking
mypy src
```

## Metrics & Monitoring

The bot exposes Prometheus metrics on port 8000:

- `trading_bot_trades_total`: Total trades executed
- `trading_bot_portfolio_value_usdt`: Portfolio value
- `trading_bot_open_positions`: Number of open positions
- `trading_bot_unrealized_pnl_usdt`: Unrealized P&L
- `trading_bot_win_rate`: Win rate
- `trading_bot_drawdown_percent`: Current drawdown

## Architecture & Technical Details

### WebSocket-Based Order Tracking

The bot implements a hybrid approach to order execution monitoring:

#### Primary: Real-time WebSocket Updates
- Connects to KuCoin's private WebSocket channel `/spotMarket/tradeOrders`
- Receives instant notifications for all order state changes:
  - `open`: Order enters the order book
  - `match`: Order partially or fully executed
  - `filled`: Order completely filled
  - `canceled`: Order canceled
  - `update`: Order modified
- Eliminates the 30-second timeout limitation of polling-only approaches
- Reduces API rate limit usage

#### Fallback: REST API Polling
- Automatically used when WebSocket connection is unavailable
- Polls order status every 0.5 seconds
- Maintains backward compatibility
- Ensures reliability even during network issues

#### Implementation Details

The `ExecutionEngine` maintains two tracking mechanisms:
```python
# WebSocket event signaling
self._ws_order_events: dict[str, asyncio.Event] = {}

# Order data from WebSocket updates  
self._ws_order_data: dict[str, dict[str, Any]] = {}
```

When tracking an order:
1. Create an asyncio.Event for the order ID
2. Wait for WebSocket update with 1-second timeout
3. If WebSocket update received, process immediately
4. If timeout, fall back to REST API polling
5. Clean up tracking data when order completes

### Dual WebSocket Architecture

The bot maintains two separate WebSocket connections:

1. **Public Connection** (`ws_manager`)
   - Market data (tickers, candles, order book)
   - No authentication required
   - Shared across all trading pairs

2. **Private Connection** (`ws_manager_private`)
   - Order execution updates
   - Requires API key authentication
   - Account-specific data

This separation ensures market data streaming continues uninterrupted even if private channel authentication fails.

### Error Handling & Resilience

- **Graceful Degradation**: If private WebSocket fails, bot continues with polling-only tracking
- **Automatic Reconnection**: WebSocket manager implements exponential backoff (max 10 attempts)
- **State Persistence**: Orders remain in `pending_orders` after timeout for later reconciliation
- **Position Sync**: Regular `sync_positions()` call ensures consistency with exchange state
- `trading_bot_drawdown_percent`: Current drawdown

## ⚠️ Disclaimer

**This software is for educational purposes only. Cryptocurrency trading involves substantial risk of loss and is not suitable for every investor. The valuation of cryptocurrencies may fluctuate, and, as a result, clients may lose more than their original investment. Do not trade with money you cannot afford to lose.**

Always test thoroughly in sandbox mode before using real funds. Past performance is not indicative of future results.

## License

MIT License - see LICENSE file for details