"""Order execution engine for the trading bot."""

import asyncio
import time
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog
from asyncio_throttle import Throttler  # type: ignore[attr-defined]
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from kucoin_bot.api.client import KuCoinAPIError, KuCoinClient
from kucoin_bot.config import Settings
from kucoin_bot.models.data_models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    SignalType,
    TimeInForce,
    TradingSignal,
)
from kucoin_bot.risk_management.manager import RiskManager

if TYPE_CHECKING:
    from kucoin_bot.api.websocket import WebSocketManager

logger = structlog.get_logger()


class ExecutionEngine:
    """Handles order execution and position management."""

    def __init__(
        self,
        client: KuCoinClient,
        risk_manager: RiskManager,
        settings: Settings,
        ws_manager: "WebSocketManager | None" = None,
    ) -> None:
        """Initialize execution engine.

        Args:
            client: KuCoin API client
            risk_manager: Risk management system
            settings: Application settings
            ws_manager: Optional WebSocket manager for real-time order updates
        """
        self.client = client
        self.risk_manager = risk_manager
        self.settings = settings
        self.ws_manager = ws_manager
        self.throttler = Throttler(rate_limit=10, period=1)  # 10 requests per second
        self.pending_orders: dict[str, Order] = {}
        self.executed_orders: list[Order] = []
        self.logger = logger.bind(component="execution_engine")

        # Track orders waiting for WebSocket updates
        self._ws_order_events: dict[str, asyncio.Event] = {}
        self._ws_order_data: dict[str, dict[str, Any]] = {}

    def load_pending_orders(self, orders: dict[str, Order]) -> None:
        """Load pending orders from persistent state.

        Args:
            orders: Dictionary of orders to load
        """
        self.pending_orders = orders
        self.logger.info("Pending orders loaded from state", count=len(orders))

    async def setup_order_tracking(self) -> None:
        """Set up WebSocket-based order tracking if available.

        This subscribes to real-time order execution updates via the private
        WebSocket channel, providing immediate notification of order fills
        instead of relying solely on polling.
        """
        if self.ws_manager and self.ws_manager._is_private:
            try:
                await self.ws_manager.subscribe_order_updates(self._handle_order_update)
                self.logger.info("WebSocket order tracking enabled")
            except Exception as e:
                self.logger.warning(
                    "Failed to setup WebSocket order tracking, will use polling only",
                    error=str(e),
                )
        else:
            self.logger.info("WebSocket order tracking not available, using polling only")

    def _handle_order_update(self, data: dict[str, Any]) -> None:
        """Handle real-time order update from WebSocket.

        Args:
            data: Order update data from WebSocket
        """
        order_id = data.get("orderId", "")
        client_oid = data.get("clientOid", "")
        update_type = data.get("type", "")
        status = data.get("status", "")

        self.logger.debug(
            "Order update received",
            order_id=order_id,
            client_oid=client_oid,
            type=update_type,
            status=status,
        )

        # Store the update data
        if order_id:
            self._ws_order_data[order_id] = data
        if client_oid:
            self._ws_order_data[client_oid] = data

        # Signal any waiting coroutines
        if order_id in self._ws_order_events:
            self._ws_order_events[order_id].set()
        if client_oid in self._ws_order_events:
            self._ws_order_events[client_oid].set()

    async def execute_signal(
        self,
        signal: TradingSignal,
        available_balance: Decimal,
    ) -> Order | None:
        """Execute a trading signal.

        Args:
            signal: Trading signal to execute
            available_balance: Available balance for trading

        Returns:
            Executed order or None if not executed
        """
        # Perform risk checks
        risk_checks = self.risk_manager.check_signal(signal)
        for check in risk_checks:
            if not check.passed:
                self.logger.warning(
                    "Risk check failed",
                    symbol=signal.symbol,
                    reason=check.reason,
                    severity=check.severity,
                )
                if check.severity == "critical":
                    return None

        # 1. Calculate Smart Leverage
        leverage = self.risk_manager.calculate_smart_leverage(signal)

        # 2. Calculate Position Size
        # Note: Leverage is calculated and stored in the position but not applied to position sizing
        # in spot trading. For futures trading, this leverage value would be used when placing orders.
        position_size = self.risk_manager.calculate_position_size(
            signal, available_balance
        )

        if position_size <= 0:
            self.logger.warning(
                "Position size is zero",
                symbol=signal.symbol,
            )
            return None

        # Determine order side
        side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL

        # Create and execute order
        try:
            order = await self._place_market_order(
                symbol=signal.symbol,
                side=side,
                size=position_size,
            )

            if order:
                await self._track_order_fill(order)
                await self._manage_position(signal, order, leverage)

            return order

        except KuCoinAPIError as e:
            self.logger.error(
                "Order execution failed",
                symbol=signal.symbol,
                error=str(e),
            )
            return None

    async def _place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        size: Decimal,
    ) -> Order:
        """Place a market order.

        Args:
            symbol: Trading pair symbol
            side: Order side
            size: Order size

        Returns:
            Created order
        """
        async with self.throttler:
            client_order_id = f"bot_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

            order = await self.client.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                size=size,
                client_order_id=client_order_id,
            )

            self.pending_orders[order.client_order_id] = order
            self.logger.info(
                "Market order placed",
                order_id=order.id,
                symbol=symbol,
                side=side.value,
                size=str(size),
            )

            return order

    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        size: Decimal,
        price: Decimal,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> Order:
        """Place a limit order.

        Args:
            symbol: Trading pair symbol
            side: Order side
            size: Order size
            price: Limit price
            time_in_force: Time in force option

        Returns:
            Created order
        """
        async with self.throttler:
            client_order_id = f"bot_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

            order = await self.client.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                size=size,
                price=price,
                client_order_id=client_order_id,
                time_in_force=time_in_force,
            )

            self.pending_orders[order.client_order_id] = order
            self.logger.info(
                "Limit order placed",
                order_id=order.id,
                symbol=symbol,
                side=side.value,
                size=str(size),
                price=str(price),
            )

            return order

    async def place_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        size: Decimal,
        stop_price: Decimal,
        limit_price: Decimal | None = None,
    ) -> Order:
        """Place a stop order.

        Args:
            symbol: Trading pair symbol
            side: Order side
            size: Order size
            stop_price: Stop trigger price
            limit_price: Optional limit price for stop-limit orders

        Returns:
            Created order
        """
        async with self.throttler:
            client_order_id = f"bot_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

            order_type = OrderType.STOP_LIMIT if limit_price else OrderType.STOP

            order = await self.client.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                size=size,
                price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id,
            )

            self.pending_orders[order.client_order_id] = order
            self.logger.info(
                "Stop order placed",
                order_id=order.id,
                symbol=symbol,
                side=side.value,
                size=str(size),
                stop_price=str(stop_price),
            )

            return order

    async def _track_order_fill(self, order: Order, timeout: int = 30) -> None:
        """Wait for order to be filled using WebSocket updates with polling fallback.

        This method now uses WebSocket order updates as the primary mechanism for
        detecting order fills, with REST API polling as a fallback. This ensures
        orders are tracked reliably even if they take longer than the timeout period.

        Args:
            order: Order to track
            timeout: Maximum wait time in seconds
        """
        start_time = time.time()

        # Set up WebSocket event if available
        use_websocket = self.ws_manager and self.ws_manager._is_private
        if use_websocket and order.id:
            self._ws_order_events[order.id] = asyncio.Event()
            if order.client_order_id:
                self._ws_order_events[order.client_order_id] = asyncio.Event()

        while time.time() - start_time < timeout:
            # Try WebSocket updates first if available
            if use_websocket:
                try:
                    # Wait for WebSocket update with short timeout
                    # Prefer order.id but use client_order_id if id is None
                    order_key = order.id if order.id is not None else order.client_order_id
                    event = self._ws_order_events.get(order_key)
                    if event:
                        await asyncio.wait_for(event.wait(), timeout=1.0)

                        # Get the update data
                        ws_data = self._ws_order_data.get(order_key)
                        if ws_data:
                            update_type = ws_data.get("type", "")
                            ws_status = ws_data.get("status", "")

                            # Handle different update types
                            if update_type == "filled" or ws_status == "done":
                                # Order is fully filled
                                order.status = OrderStatus.FILLED
                                order.filled_size = Decimal(str(ws_data.get("filledSize", "0")))

                                # Get fill price, preferring matchPrice over order price
                                match_price = ws_data.get("matchPrice")
                                price = ws_data.get("price")
                                fill_price = match_price if match_price is not None else price
                                order.filled_price = Decimal(str(fill_price or "0"))

                                order.fee = Decimal("0")  # Fee info may need separate query
                                order.updated_at = datetime.now(UTC)

                                self.executed_orders.append(order)
                                if order.client_order_id in self.pending_orders:
                                    del self.pending_orders[order.client_order_id]

                                # Clean up
                                self._cleanup_order_tracking(order)

                                self.logger.info(
                                    "Order filled (WebSocket)",
                                    order_id=order.id,
                                    symbol=order.symbol,
                                    filled_size=str(order.filled_size),
                                    filled_price=str(order.filled_price),
                                )
                                return

                            elif update_type == "canceled":
                                order.status = OrderStatus.CANCELLED
                                order.updated_at = datetime.now(UTC)

                                if order.client_order_id in self.pending_orders:
                                    del self.pending_orders[order.client_order_id]

                                # Clean up
                                self._cleanup_order_tracking(order)

                                self.logger.warning(
                                    "Order cancelled (WebSocket)",
                                    order_id=order.id,
                                    status=order.status.value,
                                )
                                return

                            # Reset event for next update
                            event.clear()

                except TimeoutError:
                    # No WebSocket update, fall through to polling
                    pass

            # Fallback to REST API polling
            await asyncio.sleep(0.5)

            async with self.throttler:
                updated_order = await self.client.get_order(order.id or "")

            if not updated_order:
                continue

            if updated_order.is_filled:
                order.status = OrderStatus.FILLED
                order.filled_size = updated_order.filled_size
                order.filled_price = updated_order.filled_price
                order.fee = updated_order.fee
                order.updated_at = datetime.now(UTC)

                self.executed_orders.append(order)
                if order.client_order_id in self.pending_orders:
                    del self.pending_orders[order.client_order_id]

                # Clean up
                self._cleanup_order_tracking(order)

                self.logger.info(
                    "Order filled (polling)",
                    order_id=order.id,
                    symbol=order.symbol,
                    filled_size=str(order.filled_size),
                    filled_price=str(order.filled_price),
                    fee=str(order.fee),
                )
                return

            if updated_order.status in [OrderStatus.CANCELLED, OrderStatus.FAILED]:
                order.status = updated_order.status
                order.updated_at = datetime.now(UTC)

                if order.client_order_id in self.pending_orders:
                    del self.pending_orders[order.client_order_id]

                # Clean up
                self._cleanup_order_tracking(order)

                self.logger.warning(
                    "Order not filled (polling)",
                    order_id=order.id,
                    status=order.status.value,
                )
                return

        # Timeout - order might be partially filled or still pending
        # Clean up tracking data but keep order in pending_orders for sync_positions to handle
        self._cleanup_order_tracking(order)

        self.logger.warning(
            "Order tracking timeout - order remains in pending_orders for monitoring",
            order_id=order.id,
            symbol=order.symbol,
        )

    def _cleanup_order_tracking(self, order: Order) -> None:
        """Clean up WebSocket tracking data for an order.

        Args:
            order: Order to clean up tracking for
        """
        if order.id:
            self._ws_order_events.pop(order.id, None)
            self._ws_order_data.pop(order.id, None)
        if order.client_order_id:
            self._ws_order_events.pop(order.client_order_id, None)
            self._ws_order_data.pop(order.client_order_id, None)

    async def _manage_position(
        self, signal: TradingSignal, order: Order, leverage: int = 1
    ) -> None:
        """Manage position after order execution.

        Args:
            signal: Original trading signal
            order: Executed order
            leverage: Leverage used for the position
        """
        if not order.is_filled or not order.filled_price:
            return

        # 3. Calculate Dynamic SL/TP
        stop_loss = self.risk_manager.calculate_dynamic_stop_loss(
            order.filled_price, order.side, signal.volatility
        )
        take_profit = self.risk_manager.calculate_dynamic_take_profit(
            order.filled_price, order.side, signal.volatility
        )

        # Create position
        position = Position(
            symbol=order.symbol,
            side=order.side,
            entry_price=order.filled_price,
            size=order.filled_size,
            current_price=order.filled_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_ids=[order.id or ""],
            leverage=leverage,
        )

        self.risk_manager.add_position(position)

        # Place stop loss order with retry logic
        if stop_loss:
            try:
                await self._place_stop_loss_with_retry(order, stop_loss)
            except KuCoinAPIError as e:
                # All retries failed - log critical alert
                self.logger.critical(
                    "CRITICAL: Failed to place stop loss after all retries - position is UNPROTECTED",
                    symbol=order.symbol,
                    position_side=order.side.value,
                    entry_price=str(order.filled_price),
                    size=str(order.filled_size),
                    stop_loss_price=str(stop_loss),
                    error=str(e),
                )
                # Position remains in risk_manager but without stop loss protection
                # Consider manual intervention or emergency position closure

    @retry(
        retry=retry_if_exception_type(KuCoinAPIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _place_stop_loss_with_retry(self, order: Order, stop_loss: Decimal) -> None:
        """Place stop loss order with automatic retry.

        Args:
            order: Executed entry order
            stop_loss: Stop loss price

        Raises:
            KuCoinAPIError: If all retry attempts fail
        """
        try:
            sl_side = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY
            await self.place_stop_order(
                symbol=order.symbol,
                side=sl_side,
                size=order.filled_size,
                stop_price=stop_loss,
            )
            self.logger.info(
                "Stop loss placed successfully",
                symbol=order.symbol,
                stop_price=str(stop_loss),
            )
        except KuCoinAPIError as e:
            self.logger.error(
                "Failed to place stop loss (will retry)",
                symbol=order.symbol,
                error=str(e),
            )
            raise

    async def close_position(
        self,
        symbol: str,
        reason: str = "Manual close",
    ) -> Order | None:
        """Close an open position.

        Args:
            symbol: Trading pair symbol
            reason: Reason for closing

        Returns:
            Close order or None if no position
        """
        position = self.risk_manager.positions.get(symbol)
        if not position:
            self.logger.warning("No position to close", symbol=symbol)
            return None

        # Place market order to close
        close_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY

        try:
            order = await self._place_market_order(
                symbol=symbol,
                side=close_side,
                size=position.size,
            )

            await self._track_order_fill(order)

            if order.is_filled and order.filled_price:
                # Calculate PnL
                pnl = position.unrealized_pnl

                # Record trade
                self.risk_manager.add_to_history(
                    symbol=symbol,
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=order.filled_price,
                    size=position.size,
                    pnl=pnl,
                )

                # Remove position
                self.risk_manager.remove_position(symbol)

                self.logger.info(
                    "Position closed",
                    symbol=symbol,
                    reason=reason,
                    pnl=str(pnl),
                )

            return order

        except KuCoinAPIError as e:
            self.logger.error(
                "Failed to close position",
                symbol=symbol,
                error=str(e),
            )
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        async with self.throttler:
            result = await self.client.cancel_order(order_id)

        if result:
            # Remove from pending orders
            for client_id, order in list(self.pending_orders.items()):
                if order.id == order_id:
                    del self.pending_orders[client_id]
                    break

        return result

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all pending orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of cancelled orders
        """
        async with self.throttler:
            count = await self.client.cancel_all_orders(symbol)

        # Clear pending orders
        if symbol:
            self.pending_orders = {
                k: v for k, v in self.pending_orders.items()
                if v.symbol != symbol
            }
        else:
            self.pending_orders.clear()

        return count

    async def sync_positions(self) -> None:
        """Sync positions with exchange data.

        Reconciles internal state with actual exchange balances and orders.
        """
        try:
            # Sync pending orders
            open_orders = await self.client.get_open_orders()

            for order in open_orders:
                if order.client_order_id not in self.pending_orders:
                    self.pending_orders[order.client_order_id] = order
                    self.logger.info(
                        "Synced pending order from exchange",
                        order_id=order.id,
                        symbol=order.symbol,
                    )

            # Verify positions still have stop loss protection
            # Build map of symbols to their stop loss orders
            stop_loss_orders_by_symbol: dict[str, list[Order]] = {}
            for order in open_orders:
                if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                    if order.symbol not in stop_loss_orders_by_symbol:
                        stop_loss_orders_by_symbol[order.symbol] = []
                    stop_loss_orders_by_symbol[order.symbol].append(order)

            for symbol in list(self.risk_manager.positions.keys()):
                position = self.risk_manager.positions[symbol]

                # Check if position has a stop loss configured but no stop loss order on exchange
                if position.stop_loss:
                    sl_orders = stop_loss_orders_by_symbol.get(symbol, [])
                    if not sl_orders:
                        self.logger.warning(
                            "Position missing stop loss order on exchange",
                            symbol=symbol,
                            position_side=position.side.value,
                            expected_stop_loss=str(position.stop_loss),
                        )

            self.logger.info(
                "Positions synced",
                pending_orders=len(self.pending_orders),
                open_positions=len(self.risk_manager.positions),
            )

        except KuCoinAPIError as e:
            self.logger.error("Failed to sync positions", error=str(e))

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics.

        Returns:
            Execution statistics
        """
        filled_orders = [o for o in self.executed_orders if o.is_filled]
        total_fees = sum(o.fee for o in filled_orders)

        return {
            "total_executed": len(self.executed_orders),
            "filled_orders": len(filled_orders),
            "pending_orders": len(self.pending_orders),
            "total_fees": str(total_fees),
            "active_positions": len(self.risk_manager.positions),
        }
