# KuCoin Trading Bot

A sophisticated, production-ready cryptocurrency trading bot for KuCoin exchange, built with modern 2025 standards and best practices.

## Features

### 🚀 Core Capabilities

- **Async Architecture**: Built entirely on asyncio for high-performance, non-blocking operations
- **Multiple Trading Strategies**: 
  - Momentum Trading (RSI + EMA crossovers + MACD)
  - Mean Reversion (Bollinger Bands)
  - Grid Trading (for ranging markets)
  - Scalping (high-frequency short-term trades)
  - DCA (Dollar Cost Averaging for accumulation)

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

## ⚠️ Disclaimer

**This software is for educational purposes only. Cryptocurrency trading involves substantial risk of loss and is not suitable for every investor. The valuation of cryptocurrencies may fluctuate, and, as a result, clients may lose more than their original investment. Do not trade with money you cannot afford to lose.**

Always test thoroughly in sandbox mode before using real funds. Past performance is not indicative of future results.

## License

MIT License - see LICENSE file for details