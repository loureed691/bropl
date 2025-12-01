"""Prometheus metrics for monitoring the trading bot."""

from decimal import Decimal

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server


class TradingMetrics:
    """Prometheus metrics for trading bot monitoring."""

    def __init__(self) -> None:
        """Initialize metrics collectors."""
        # Trading metrics
        self.trades_total = Counter(
            "trading_bot_trades_total",
            "Total number of trades executed",
            ["symbol", "side", "status"],
        )

        self.trade_value = Histogram(
            "trading_bot_trade_value_usdt",
            "Trade value in USDT",
            ["symbol", "side"],
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
        )

        self.trade_pnl = Histogram(
            "trading_bot_trade_pnl_usdt",
            "Trade profit/loss in USDT",
            ["symbol"],
            buckets=[-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000],
        )

        # Position metrics
        self.open_positions = Gauge(
            "trading_bot_open_positions",
            "Number of open positions",
        )

        self.position_value = Gauge(
            "trading_bot_position_value_usdt",
            "Total value of open positions in USDT",
        )

        self.position_pnl = Gauge(
            "trading_bot_unrealized_pnl_usdt",
            "Unrealized profit/loss in USDT",
        )

        # Portfolio metrics
        self.portfolio_value = Gauge(
            "trading_bot_portfolio_value_usdt",
            "Total portfolio value in USDT",
        )

        self.available_balance = Gauge(
            "trading_bot_available_balance_usdt",
            "Available balance in USDT",
        )

        self.drawdown = Gauge(
            "trading_bot_drawdown_percent",
            "Current drawdown percentage",
        )

        # Performance metrics
        self.win_rate = Gauge(
            "trading_bot_win_rate",
            "Win rate as a decimal (0-1)",
        )

        self.profit_factor = Gauge(
            "trading_bot_profit_factor",
            "Profit factor (gross profit / gross loss)",
        )

        self.sharpe_ratio = Gauge(
            "trading_bot_sharpe_ratio",
            "Sharpe ratio",
        )

        # Signal metrics
        self.signals_generated = Counter(
            "trading_bot_signals_total",
            "Total number of signals generated",
            ["symbol", "signal_type"],
        )

        self.signal_confidence = Histogram(
            "trading_bot_signal_confidence",
            "Signal confidence distribution",
            ["symbol", "signal_type"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # API metrics
        self.api_requests = Counter(
            "trading_bot_api_requests_total",
            "Total API requests",
            ["endpoint", "method", "status"],
        )

        self.api_latency = Histogram(
            "trading_bot_api_latency_seconds",
            "API request latency",
            ["endpoint", "method"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        self.api_errors = Counter(
            "trading_bot_api_errors_total",
            "Total API errors",
            ["endpoint", "error_code"],
        )

        # System metrics
        self.bot_info = Info(
            "trading_bot_info",
            "Trading bot information",
        )

        self.bot_uptime = Gauge(
            "trading_bot_uptime_seconds",
            "Bot uptime in seconds",
        )

        self.daily_trades = Gauge(
            "trading_bot_daily_trades",
            "Number of trades today",
        )

    def record_trade(
        self,
        symbol: str,
        side: str,
        status: str,
        value: Decimal,
        pnl: Decimal | None = None,
    ) -> None:
        """Record a trade execution.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            status: Order status
            value: Trade value in USDT
            pnl: Profit/loss if closed (optional)
        """
        self.trades_total.labels(symbol=symbol, side=side, status=status).inc()
        self.trade_value.labels(symbol=symbol, side=side).observe(float(value))

        if pnl is not None:
            self.trade_pnl.labels(symbol=symbol).observe(float(pnl))

    def record_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
    ) -> None:
        """Record a trading signal.

        Args:
            symbol: Trading pair symbol
            signal_type: Signal type (buy/sell/hold)
            confidence: Signal confidence (0-1)
        """
        self.signals_generated.labels(symbol=symbol, signal_type=signal_type).inc()
        self.signal_confidence.labels(symbol=symbol, signal_type=signal_type).observe(confidence)

    def update_positions(
        self,
        count: int,
        total_value: Decimal,
        unrealized_pnl: Decimal,
    ) -> None:
        """Update position metrics.

        Args:
            count: Number of open positions
            total_value: Total position value
            unrealized_pnl: Unrealized profit/loss
        """
        self.open_positions.set(count)
        self.position_value.set(float(total_value))
        self.position_pnl.set(float(unrealized_pnl))

    def update_portfolio(
        self,
        value: Decimal,
        available: Decimal,
        drawdown_percent: Decimal,
    ) -> None:
        """Update portfolio metrics.

        Args:
            value: Total portfolio value
            available: Available balance
            drawdown_percent: Current drawdown percentage
        """
        self.portfolio_value.set(float(value))
        self.available_balance.set(float(available))
        self.drawdown.set(float(drawdown_percent))

    def update_performance(
        self,
        win_rate: float,
        profit_factor: float,
        sharpe: float = 0.0,
    ) -> None:
        """Update performance metrics.

        Args:
            win_rate: Win rate (0-1)
            profit_factor: Profit factor
            sharpe: Sharpe ratio
        """
        self.win_rate.set(win_rate)
        self.profit_factor.set(profit_factor)
        self.sharpe_ratio.set(sharpe)

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status: str,
        latency: float,
    ) -> None:
        """Record an API request.

        Args:
            endpoint: API endpoint
            method: HTTP method
            status: Response status
            latency: Request latency in seconds
        """
        self.api_requests.labels(endpoint=endpoint, method=method, status=status).inc()
        self.api_latency.labels(endpoint=endpoint, method=method).observe(latency)

    def record_api_error(self, endpoint: str, error_code: str) -> None:
        """Record an API error.

        Args:
            endpoint: API endpoint
            error_code: Error code
        """
        self.api_errors.labels(endpoint=endpoint, error_code=error_code).inc()

    def set_bot_info(self, version: str, strategy: str, environment: str) -> None:
        """Set bot information.

        Args:
            version: Bot version
            strategy: Active strategy name
            environment: Running environment
        """
        self.bot_info.info({
            "version": version,
            "strategy": strategy,
            "environment": environment,
        })


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus metrics HTTP server.

    Args:
        port: Port to listen on
    """
    start_http_server(port)


# Global metrics instance
metrics = TradingMetrics()
