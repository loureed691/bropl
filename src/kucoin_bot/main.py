"""Main trading bot orchestrator."""

import asyncio
import signal
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog

from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.api.websocket import WebSocketManager
from kucoin_bot.config import Settings, StrategyName, StrategySettings, get_settings
from kucoin_bot.execution.engine import ExecutionEngine
from kucoin_bot.models.data_models import Candle, Ticker
from kucoin_bot.pair_selector.selector import PairSelector
from kucoin_bot.risk_management.manager import RiskManager
from kucoin_bot.strategies.base import BaseStrategy, create_strategy
from kucoin_bot.utils.logging import setup_logging
from kucoin_bot.utils.metrics import metrics, start_metrics_server


class TradingBot:
    """Main trading bot class that orchestrates all components."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the trading bot.

        Args:
            settings: Optional settings override
        """
        self.settings = settings or get_settings()
        self._running = False
        self._start_time: datetime | None = None

        # Initialize logging
        setup_logging(
            self.settings.app.log_level,
            self.settings.app.log_format,
        )

        self.logger = structlog.get_logger().bind(component="trading_bot")

        # Initialize components
        self.client = KuCoinClient(self.settings)
        self.ws_manager = WebSocketManager(self.settings)  # Public market data
        self.ws_manager_private: WebSocketManager | None = None  # Private order updates
        self.risk_manager = RiskManager(
            self.settings.risk,
            self.settings.trading,
        )
        self.execution_engine = ExecutionEngine(
            self.client,
            self.risk_manager,
            self.settings,
            self.ws_manager,  # Pass WebSocket manager for real-time order tracking
        )
        self.strategy: BaseStrategy = create_strategy(self.settings.strategy)

        # Per-pair strategies for automatic strategy selection
        self.pair_strategies: dict[str, BaseStrategy] = {}
        self.auto_select_strategy_enabled = self.settings.trading.auto_select_strategy

        # Initialize pair selector for auto-selection
        self.pair_selector: PairSelector | None = None
        if self.settings.trading.auto_select_pairs:
            self.pair_selector = PairSelector(
                client=self.client,
                min_volume_24h=Decimal(str(self.settings.trading.auto_select_min_volume)),
                min_signal_strength=self.settings.trading.auto_select_min_signal,
                top_pairs_count=self.settings.trading.auto_select_count,
                signal_weight=self.settings.trading.pair_score_signal_weight,
                volume_weight=self.settings.trading.pair_score_volume_weight,
                volatility_weight=self.settings.trading.pair_score_volatility_weight,
                volume_threshold=self.settings.trading.pair_score_volume_threshold,
            )

        # Trading state
        self.trading_pairs = self.settings.trading.get_pairs_list()
        self.latest_prices: dict[str, Decimal] = {}
        self.candle_buffers: dict[str, list[Candle]] = {}

        # Lock for thread-safe trading pairs updates
        self._pairs_lock = asyncio.Lock()

        # Task management
        self._tasks: list[asyncio.Task[Any]] = []
        self._task_registry: dict[str, Callable[[], Any]] = {}

    def _get_strategy_for_pair(self, symbol: str) -> BaseStrategy:
        """Get the strategy to use for a specific pair.

        When auto_select_strategy is enabled, returns a per-pair strategy.
        Otherwise, returns the global strategy.

        Args:
            symbol: Trading pair symbol

        Returns:
            Strategy instance for the pair
        """
        if self.auto_select_strategy_enabled and symbol in self.pair_strategies:
            return self.pair_strategies[symbol]
        return self.strategy

    def _create_strategy_for_pair(self, symbol: str, strategy_name: str) -> BaseStrategy:
        """Create and cache a strategy for a specific pair.

        Args:
            symbol: Trading pair symbol
            strategy_name: Name of the strategy to create

        Returns:
            Created strategy instance
        """
        try:
            strategy_enum = StrategyName(strategy_name)
        except ValueError:
            self.logger.warning(
                "Invalid strategy name, using default momentum",
                symbol=symbol,
                invalid_strategy=strategy_name,
            )
            strategy_enum = StrategyName.MOMENTUM

        # Create settings with the specified strategy
        strategy_settings = StrategySettings(
            strategy_name=strategy_enum,
            rsi_period=self.settings.strategy.rsi_period,
            rsi_overbought=self.settings.strategy.rsi_overbought,
            rsi_oversold=self.settings.strategy.rsi_oversold,
            ema_short_period=self.settings.strategy.ema_short_period,
            ema_long_period=self.settings.strategy.ema_long_period,
        )
        strategy = create_strategy(strategy_settings)
        self.pair_strategies[symbol] = strategy

        self.logger.info(
            "Created strategy for pair",
            symbol=symbol,
            strategy=strategy_enum.value,
        )

        return strategy

    async def start(self) -> None:
        """Start the trading bot."""
        self.logger.info(
            "Starting trading bot",
            pairs=self.trading_pairs,
            strategy=self.strategy.name,
            environment="sandbox" if self.settings.app.use_sandbox else "production",
            auto_select_enabled=self.settings.trading.auto_select_pairs,
            auto_select_strategy=self.auto_select_strategy_enabled,
        )

        self._running = True
        self._start_time = datetime.now(UTC)

        # Set up signal handlers
        loop = asyncio.get_event_loop()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda _sig=sig: asyncio.create_task(self.stop()))
        except NotImplementedError:
            # Windows ProactorEventLoop does not support add_signal_handler
            self.logger.info("Signal handlers not supported on this platform. Use Ctrl+C to stop.")

        try:
            # Connect to API
            await self.client._create_session()

            # Verify connectivity
            if not await self.client.ping():
                raise ConnectionError("Failed to connect to KuCoin API")

            # Load persistent state
            await self._load_state()

            # Reconcile loaded positions with actual exchange state
            await self.execution_engine.sync_positions()

            # Start background pair selection if enabled (non-blocking)
            # Bot will start with configured pairs and update once scan completes
            if self.pair_selector:
                self.logger.info("Starting background pair selection...")
                asyncio.create_task(self._auto_select_pairs())

            # Get initial balances
            await self._update_portfolio()

            # Start metrics server
            start_metrics_server(port=8000)
            metrics.set_bot_info(
                version="1.0.0",
                strategy=self.strategy.name,
                environment=self.settings.app.environment.value,
            )

            # Load historical data
            await self._load_historical_data()

            # Connect WebSocket for market data (public)
            await self.ws_manager.connect(private=False)

            # Subscribe to market data
            await self._subscribe_market_data()

            # Connect to private WebSocket for order updates
            # Note: KuCoin requires separate connections for public and private channels
            try:
                from kucoin_bot.api.websocket import WebSocketManager

                # Create a separate WebSocket manager for private channels
                self.ws_manager_private = WebSocketManager(self.settings)
                await self.ws_manager_private.connect(private=True)

                # Update execution engine with private WebSocket manager
                self.execution_engine.ws_manager = self.ws_manager_private

                # Set up order tracking via WebSocket
                await self.execution_engine.setup_order_tracking()

                self.logger.info("Private WebSocket connection established for order tracking")
            except Exception as e:
                self.logger.warning(
                    "Failed to establish private WebSocket connection, order tracking will use polling only",
                    error=str(e),
                )

            # Register task factories
            self._task_registry = {
                "trading_loop": self._trading_loop,
                "monitoring_loop": self._monitoring_loop,
                "position_check_loop": self._position_check_loop,
                "state_save_loop": self._state_save_loop,
            }

            # Add pair selection loop if auto-select is enabled
            if self.pair_selector:
                self._task_registry["pair_selection_loop"] = self._pair_selection_loop

            # Start task supervisor
            await self._task_supervisor()

        except Exception as e:
            self.logger.error("Bot startup failed", error=str(e))
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        if not self._running:
            return

        self.logger.info("Stopping trading bot...")
        self._running = False

        # Save state before shutdown
        await self._save_state()

        # Cancel running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Disconnect WebSocket
        await self.ws_manager.disconnect()

        # Disconnect private WebSocket if connected
        if self.ws_manager_private:
            await self.ws_manager_private.disconnect()

        # Close API client
        await self.client.close()

        self.logger.info("Trading bot stopped")

    async def _auto_select_pairs(self) -> None:
        """Auto-select the best trading pairs based on signal strength."""
        if not self.pair_selector:
            return

        self.logger.info("Auto-selecting best trading pairs...")

        try:
            signal_type = self.settings.trading.auto_select_signal_type
            signal_type_filter = None if signal_type == "any" else signal_type

            selected_pairs = await self.pair_selector.get_top_pairs(
                signal_type=signal_type_filter,
                count=self.settings.trading.auto_select_count,
            )

            if selected_pairs:
                async with self._pairs_lock:
                    self.trading_pairs = selected_pairs
                self.logger.info(
                    "Trading pairs auto-selected",
                    pairs=self.trading_pairs,
                    signal_type=signal_type,
                )

                # Log details of selected pairs and assign strategies if enabled
                for pair in selected_pairs:
                    score = self.pair_selector.get_pair_details(pair)
                    if score:
                        self.logger.info(
                            "Selected pair details",
                            symbol=pair,
                            signal_type=score.signal_type,
                            strength=f"{score.signal_strength:.3f}",
                            composite_score=f"{score.composite_score:.3f}",
                            volume_24h=str(score.volume_24h),
                            recommended_strategy=score.recommended_strategy,
                        )

                        # Create per-pair strategy if auto_select_strategy is enabled
                        if self.auto_select_strategy_enabled:
                            strategy = self._create_strategy_for_pair(pair, score.recommended_strategy)
                            # Warm up strategy with existing candle data if available
                            if pair in self.candle_buffers:
                                for candle in self.candle_buffers[pair]:
                                    strategy.update(pair, candle)
            else:
                # No pairs met selection criteria
                fallback = self.settings.trading.get_pairs_list()
                if fallback:
                    async with self._pairs_lock:
                        self.trading_pairs = fallback
                    self.logger.warning(
                        "No pairs met selection criteria, using configured fallback pairs",
                        fallback_pairs=fallback,
                    )
                else:
                    async with self._pairs_lock:
                        self.trading_pairs = []
                    self.logger.error(
                        "No pairs met selection criteria and no fallback configured. "
                        "Bot will not trade until pairs are available."
                    )
        except Exception as e:
            self.logger.error("Error during auto-selection", error=str(e))

    async def _pair_selection_loop(self) -> None:
        """Periodically re-scan and update trading pairs."""
        scan_interval = self.settings.trading.auto_select_interval

        while self._running:
            try:
                await asyncio.sleep(scan_interval)

                if not self._running or not self.pair_selector:
                    break

                self.logger.info("Running periodic pair scan...")

                # Get new top pairs
                signal_type = self.settings.trading.auto_select_signal_type
                signal_type_filter = None if signal_type == "any" else signal_type

                new_pairs = await self.pair_selector.get_top_pairs(
                    signal_type=signal_type_filter,
                    count=self.settings.trading.auto_select_count,
                )

                if new_pairs and new_pairs != self.trading_pairs:
                    # Identify changes
                    async with self._pairs_lock:
                        added = set(new_pairs) - set(self.trading_pairs)
                        removed = set(self.trading_pairs) - set(new_pairs)

                    if added or removed:
                        self.logger.info(
                            "Trading pairs updated",
                            added=list(added),
                            removed=list(removed),
                            new_pairs=new_pairs,
                        )

                        # Check for open positions before removing pairs
                        pairs_with_positions = []
                        for symbol in removed:
                            if symbol in self.risk_manager.positions:
                                self.logger.warning(
                                    "Cannot remove pair with open position, keeping it active",
                                    symbol=symbol,
                                    position_side=self.risk_manager.positions[symbol].side.value,
                                )
                                pairs_with_positions.append(symbol)
                            else:
                                await self._unsubscribe_pair(symbol)

                        # Update removed set to exclude pairs with positions
                        removed = removed - set(pairs_with_positions)

                        # Update trading pairs atomically
                        async with self._pairs_lock:
                            # Keep pairs with open positions
                            self.trading_pairs = list(set(new_pairs) | set(pairs_with_positions))

                        # Subscribe to new pairs and assign strategies if enabled
                        for symbol in added:
                            # Create per-pair strategy if auto_select_strategy is enabled
                            if self.auto_select_strategy_enabled:
                                score = self.pair_selector.get_pair_details(symbol)
                                if score:
                                    strategy = self._create_strategy_for_pair(symbol, score.recommended_strategy)
                                    # Warm up with existing candle buffer data if available
                                    if symbol in self.candle_buffers:
                                        for candle in self.candle_buffers[symbol]:
                                            strategy.update(symbol, candle)

                            await self._subscribe_pair(symbol)

                # Log scan summary
                summary = self.pair_selector.get_scan_summary()
                self.logger.info(
                    "Pair scan summary",
                    total_pairs=summary["total_pairs"],
                    bullish=summary["bullish_count"],
                    bearish=summary["bearish_count"],
                    neutral=summary["neutral_count"],
                )

            except asyncio.CancelledError:
                break
            except (ConnectionError, TimeoutError) as e:
                # Transient network errors - retry after delay
                self.logger.warning("Temporary error in pair selection, will retry", error=str(e))
                await asyncio.sleep(60)
            except Exception as e:
                # Unexpected errors - log and disable auto-selection
                self.logger.error(
                    "Fatal error in pair selection loop, disabling auto-selection",
                    error=str(e),
                )
                self.pair_selector = None
                break

    async def _subscribe_pair(self, symbol: str) -> None:
        """Subscribe to market data for a new pair.

        Args:
            symbol: Trading pair symbol
        """
        try:
            # Load historical data for the new pair
            candles = await self.client.get_candles(
                symbol=symbol,
                interval="1hour",
            )
            self.candle_buffers[symbol] = candles

            # Get the appropriate strategy for this pair
            strategy = self._get_strategy_for_pair(symbol)

            # Warm up strategy
            for candle in candles:
                strategy.update(symbol, candle)

            # Subscribe to real-time data
            await self.ws_manager.subscribe_ticker(
                symbol,
                self._make_ticker_callback(symbol),
            )
            await self.ws_manager.subscribe_candles(
                symbol,
                "1hour",
                self._make_candle_callback(symbol),
            )

            self.logger.info(
                "Subscribed to new pair",
                symbol=symbol,
                strategy=strategy.name,
            )

        except Exception as e:
            self.logger.error("Failed to subscribe to pair", symbol=symbol, error=str(e))

    async def _unsubscribe_pair(self, symbol: str) -> None:
        """Unsubscribe from market data for a pair.

        Args:
            symbol: Trading pair symbol
        """
        try:
            # Unsubscribe from WebSocket topics
            ticker_topic = f"/market/ticker:{symbol}"
            candles_topic = f"/market/candles:{symbol}_1hour"

            await self.ws_manager.unsubscribe(ticker_topic)
            await self.ws_manager.unsubscribe(candles_topic)

            # Clean up candle buffer
            if symbol in self.candle_buffers:
                del self.candle_buffers[symbol]

            # Clean up latest prices
            if symbol in self.latest_prices:
                del self.latest_prices[symbol]

            # Reset strategy state for this symbol
            strategy = self._get_strategy_for_pair(symbol)
            strategy.reset(symbol)

            # Clean up per-pair strategy if exists
            if symbol in self.pair_strategies:
                del self.pair_strategies[symbol]

            self.logger.info("Unsubscribed from pair", symbol=symbol)

        except Exception as e:
            self.logger.error("Failed to unsubscribe from pair", symbol=symbol, error=str(e))

    async def _load_historical_data(self) -> None:
        """Load historical candle data for strategies."""
        self.logger.info("Loading historical data...")

        for symbol in self.trading_pairs:
            try:
                candles = await self.client.get_candles(
                    symbol=symbol,
                    interval="1hour",
                )

                self.candle_buffers[symbol] = candles

                # Get the appropriate strategy for this pair
                strategy = self._get_strategy_for_pair(symbol)

                # Warm up strategy with historical data
                for candle in candles:
                    strategy.update(symbol, candle)

                self.logger.info(
                    "Historical data loaded",
                    symbol=symbol,
                    candles=len(candles),
                    strategy=strategy.name,
                )

            except Exception as e:
                self.logger.error(
                    "Failed to load historical data",
                    symbol=symbol,
                    error=str(e),
                )
    def _make_ticker_callback(self, symbol: str) -> Callable[[Ticker], None]:
        """Create a ticker callback bound to a specific symbol."""
        def callback(ticker: Ticker) -> None:
            self._on_ticker(symbol, ticker)
        return callback

    def _make_candle_callback(self, symbol: str) -> Callable[[Candle], None]:
        """Create a candle callback bound to a specific symbol."""
        def callback(candle: Candle) -> None:
            asyncio.create_task(self._on_candle(symbol, candle))
        return callback

    async def _subscribe_market_data(self) -> None:
        """Subscribe to real-time market data."""
        for symbol in self.trading_pairs:
            # Subscribe to ticker updates
            await self.ws_manager.subscribe_ticker(
                symbol,
                self._make_ticker_callback(symbol),
            )

            # Subscribe to candle updates
            await self.ws_manager.subscribe_candles(
                symbol,
                "1hour",
                self._make_candle_callback(symbol),
            )

            self.logger.info("Subscribed to market data", symbol=symbol)

    def _on_ticker(self, symbol: str, ticker: Ticker) -> None:
        """Handle ticker update.

        Args:
            symbol: Trading pair symbol
            ticker: Ticker data
        """
        self.latest_prices[symbol] = ticker.price

        # Update position prices
        self.risk_manager.update_position(symbol, ticker.price)

    async def _on_candle(self, symbol: str, candle: Candle) -> None:
        """Handle candle update.

        Args:
            symbol: Trading pair symbol
            candle: Candle data
        """
        # Add to buffer
        if symbol not in self.candle_buffers:
            self.candle_buffers[symbol] = []
        self.candle_buffers[symbol].append(candle)

        # Keep buffer size manageable
        max_candles = 500
        if len(self.candle_buffers[symbol]) > max_candles:
            self.candle_buffers[symbol] = self.candle_buffers[symbol][-max_candles:]

        # Get the appropriate strategy for this pair
        strategy = self._get_strategy_for_pair(symbol)

        # Update strategy and check for signals
        trading_signal = strategy.update(symbol, candle)

        # Record signal metrics
        metrics.record_signal(
            symbol,
            trading_signal.signal_type.value,
            trading_signal.confidence,
        )

        # Execute if actionable
        if trading_signal.is_actionable and trading_signal.confidence >= 0.6:
            await self._process_signal(symbol, trading_signal)

    async def _process_signal(self, symbol: str, signal: Any) -> None:
        """Process a trading signal.

        Args:
            symbol: Trading pair symbol
            signal: Trading signal
        """
        # Check risk before executing
        risk_checks = self.risk_manager.check_signal(signal)
        all_passed = all(check.passed for check in risk_checks)

        if not all_passed:
            self.logger.info(
                "Signal blocked by risk checks",
                symbol=symbol,
                signal=signal.signal_type.value,
                reasons=[c.reason for c in risk_checks if not c.passed],
            )
            return

        # Get available balance
        balance = await self.client.get_account_balance("USDT")
        if not balance:
            self.logger.warning("Could not get USDT balance")
            return

        available = balance.available

        # Execute signal
        order = await self.execution_engine.execute_signal(signal, available)

        if order and order.is_filled:
            metrics.record_trade(
                symbol=symbol,
                side=signal.signal_type.value,
                status="filled",
                value=order.filled_size * (order.filled_price or Decimal("0")),
            )

    async def _trading_loop(self) -> None:
        """Main trading loop."""
        while self._running:
            try:
                # Check for position exits
                for symbol in list(self.risk_manager.positions.keys()):
                    if symbol in self.latest_prices:
                        current_price = self.latest_prices[symbol]
                        should_close, reason = self.risk_manager.should_close_position(
                            symbol, current_price
                        )

                        if should_close:
                            await self.execution_engine.close_position(symbol, reason)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in trading loop", error=str(e))
                await asyncio.sleep(5)

    async def _monitoring_loop(self) -> None:
        """Monitoring and metrics update loop."""
        while self._running:
            try:
                # Update portfolio metrics
                await self._update_portfolio()

                # Update uptime
                if self._start_time:
                    uptime = (datetime.now(UTC) - self._start_time).total_seconds()
                    metrics.bot_uptime.set(uptime)

                # Update position metrics
                positions = self.risk_manager.positions
                total_value = sum(
                    (p.position_value for p in positions.values()),
                    Decimal("0"),
                )
                unrealized_pnl = sum(
                    (p.unrealized_pnl for p in positions.values()),
                    Decimal("0"),
                )

                metrics.update_positions(
                    count=len(positions),
                    total_value=total_value,
                    unrealized_pnl=unrealized_pnl,
                )

                # Update performance metrics
                stats = self.risk_manager.get_trade_statistics()
                metrics.update_performance(
                    win_rate=stats["win_rate"],
                    profit_factor=stats["profit_factor"],
                )

                # Log status
                risk_metrics = self.risk_manager.get_metrics()
                self.logger.info(
                    "Bot status",
                    positions=risk_metrics.open_positions,
                    daily_trades=risk_metrics.daily_trades,
                    portfolio_value=str(risk_metrics.portfolio_value),
                    drawdown=f"{risk_metrics.current_drawdown:.2f}%",
                )

                await asyncio.sleep(60)  # Update every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(30)

    async def _position_check_loop(self) -> None:
        """Periodically sync positions with exchange."""
        while self._running:
            try:
                await self.execution_engine.sync_positions()
                await asyncio.sleep(300)  # Sync every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in position check loop", error=str(e))
                await asyncio.sleep(60)

    async def _update_portfolio(self) -> None:
        """Update portfolio value from exchange."""
        try:
            balances = await self.client.get_accounts()
            total_value = Decimal("0")
            available_usdt = Decimal("0")

            for balance in balances:
                if balance.currency == "USDT":
                    total_value += balance.total
                    available_usdt = balance.available
                else:
                    # Convert to USDT using latest price
                    symbol = f"{balance.currency}-USDT"
                    if symbol in self.latest_prices:
                        total_value += balance.total * self.latest_prices[symbol]

            self.risk_manager.set_portfolio_value(total_value)

            # Update metrics
            risk_metrics = self.risk_manager.get_metrics()
            metrics.update_portfolio(
                value=total_value,
                available=available_usdt,
                drawdown_percent=risk_metrics.current_drawdown,
            )

        except Exception as e:
            self.logger.error("Failed to update portfolio", error=str(e))

    async def _load_state(self) -> None:
        """Load persistent state from disk."""
        from kucoin_bot.persistence import StateManager

        try:
            state_manager = StateManager()
            positions, pending_orders = state_manager.load_state()

            # Load positions into risk manager
            self.risk_manager.load_positions(positions)

            # Load pending orders into execution engine
            self.execution_engine.load_pending_orders(pending_orders)

            self.logger.info(
                "State loaded successfully",
                positions=len(positions),
                pending_orders=len(pending_orders),
            )

        except Exception as e:
            self.logger.error("Failed to load state", error=str(e))

    async def _save_state(self) -> None:
        """Save persistent state to disk."""
        from kucoin_bot.persistence import StateManager

        try:
            state_manager = StateManager()
            state_manager.save_state(
                positions=self.risk_manager.positions,
                pending_orders=self.execution_engine.pending_orders,
            )
        except Exception as e:
            self.logger.error("Failed to save state", error=str(e))

    async def _state_save_loop(self) -> None:
        """Periodically save state to disk."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Save every minute
                await self._save_state()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in state save loop", error=str(e))
                await asyncio.sleep(30)

    async def _task_supervisor(self) -> None:
        """Supervise tasks and restart them if they fail."""
        # Start all tasks
        for task_name, task_factory in self._task_registry.items():
            task = asyncio.create_task(task_factory(), name=task_name)
            self._tasks.append(task)

        self.logger.info("All tasks started", task_count=len(self._tasks))

        # Monitor tasks
        while self._running:
            try:
                # Check for completed/failed tasks - collect all done tasks first
                done_tasks = [
                    (i, task)
                    for i, task in enumerate(self._tasks)
                    if task.done()
                ]

                for i, task in done_tasks:
                    task_name = task.get_name()

                    # Check if task failed
                    try:
                        exception = task.exception()
                        if exception:
                            self.logger.error(
                                "Task failed with exception",
                                task_name=task_name,
                                error=str(exception),
                            )
                    except asyncio.CancelledError:
                        self.logger.info("Task was cancelled", task_name=task_name)
                        continue

                    # Restart the task if it's in the registry and bot is still running
                    if task_name in self._task_registry and self._running:
                        self.logger.warning(
                            "Restarting failed task",
                            task_name=task_name,
                        )
                        task_factory = self._task_registry[task_name]
                        new_task = asyncio.create_task(task_factory(), name=task_name)
                        self._tasks[i] = new_task

                await asyncio.sleep(5)  # Check every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in task supervisor", error=str(e))
                await asyncio.sleep(10)

    async def manual_trade(
        self,
        symbol: str,
        side: str,
        size: Decimal,
    ) -> Any:
        """Execute a manual trade.

        Args:
            symbol: Trading pair symbol
            side: Order side ("buy" or "sell")
            size: Order size

        Returns:
            Executed order
        """
        from kucoin_bot.models.data_models import OrderSide

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        return await self.execution_engine._place_market_order(
            symbol=symbol,
            side=order_side,
            size=size,
        )

    def get_status(self) -> dict[str, Any]:
        """Get current bot status.

        Returns:
            Status dictionary
        """
        risk_metrics = self.risk_manager.get_metrics()
        exec_stats = self.execution_engine.get_execution_stats()
        trade_stats = self.risk_manager.get_trade_statistics()

        status = {
            "running": self._running,
            "uptime": str(datetime.now(UTC) - self._start_time) if self._start_time else "0",
            "strategy": self.strategy.name,
            "trading_pairs": self.trading_pairs,
            "environment": self.settings.app.environment.value,
            "auto_select_enabled": self.settings.trading.auto_select_pairs,
            "risk_metrics": {
                "portfolio_value": str(risk_metrics.portfolio_value),
                "open_positions": risk_metrics.open_positions,
                "daily_trades": risk_metrics.daily_trades,
                "current_drawdown": f"{risk_metrics.current_drawdown:.2f}%",
                "is_trading_allowed": risk_metrics.is_trading_allowed,
            },
            "execution_stats": exec_stats,
            "trade_statistics": trade_stats,
        }

        # Add pair selection info if enabled
        if self.pair_selector:
            status["pair_selection"] = self.pair_selector.get_scan_summary()

        # Add per-pair strategy info if auto_select_strategy is enabled
        if self.auto_select_strategy_enabled and self.pair_strategies:
            status["auto_select_strategy"] = True
            status["pair_strategies"] = {
                symbol: strategy.name
                for symbol, strategy in self.pair_strategies.items()
            }
        else:
            status["auto_select_strategy"] = False

        return status


async def run_bot() -> None:
    """Run the trading bot."""
    bot = TradingBot()
    await bot.start()


def main() -> None:
    """Entry point for the trading bot."""
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
