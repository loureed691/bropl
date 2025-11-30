"""WebSocket client for real-time market data and private channel updates."""

import asyncio
import contextlib
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import aiohttp
import orjson
import structlog
import websockets
from websockets import WebSocketClientProtocol

from kucoin_bot.config import Settings
from kucoin_bot.models.data_models import Candle, MarketDepth, Ticker, Trade

logger = structlog.get_logger()


class WebSocketManager:
    """Manages WebSocket connections to KuCoin for real-time data."""

    def __init__(self, settings: Settings) -> None:
        """Initialize WebSocket manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.base_url = settings.base_url
        self._ws: WebSocketClientProtocol | None = None
        self._running = False
        self._ping_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._receive_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._callbacks: dict[str, list[Callable[[Any], None]]] = {}
        self._subscriptions: set[str] = set()
        self._connect_id = 0
        self._token: str | None = None
        self._endpoint: str | None = None
        self._ping_interval = 30
        self._reconnect_delay = 5
        self.logger = logger.bind(component="websocket_manager")

    async def connect(self, private: bool = False) -> None:
        """Establish WebSocket connection.

        Args:
            private: Whether to connect to private channel
        """
        # Get WebSocket token and endpoint
        token_data = await self._get_ws_token(private)
        self._token = token_data["token"]
        self._endpoint = token_data["instanceServers"][0]["endpoint"]
        self._ping_interval = token_data["instanceServers"][0].get("pingInterval", 30000) // 1000

        # Build connection URL
        connect_id = int(time.time() * 1000)
        ws_url = f"{self._endpoint}?token={self._token}&connectId={connect_id}"

        self.logger.info("Connecting to WebSocket", endpoint=self._endpoint)

        self._ws = await websockets.connect(ws_url)
        self._running = True

        # Wait for welcome message
        welcome = await self._ws.recv()
        welcome_data = orjson.loads(welcome)
        if welcome_data.get("type") != "welcome":
            raise ConnectionError("Failed to receive welcome message")

        self.logger.info("WebSocket connected successfully")

        # Start background tasks
        self._ping_task = asyncio.create_task(self._ping_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _get_ws_token(self, private: bool = False) -> dict[str, Any]:
        """Get WebSocket connection token.

        Args:
            private: Whether to get private channel token

        Returns:
            Token data with endpoint information
        """
        endpoint = "/api/v1/bullet-private" if private else "/api/v1/bullet-public"

        async with aiohttp.ClientSession() as session, session.post(f"{self.base_url}{endpoint}") as response:
            data = await response.json()
            if data.get("code") != "200000":
                raise ConnectionError(f"Failed to get WS token: {data.get('msg')}")
            return data["data"]

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False

        if self._ping_task:
            self._ping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ping_task

        if self._receive_task:
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._subscriptions.clear()
        self.logger.info("WebSocket disconnected")

    async def _ping_loop(self) -> None:
        """Send periodic ping messages to keep connection alive."""
        while self._running and self._ws:
            try:
                ping_msg = orjson.dumps({
                    "id": str(int(time.time() * 1000)),
                    "type": "ping",
                })
                await self._ws.send(ping_msg)
                await asyncio.sleep(self._ping_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Ping failed", error=str(e))
                break

    async def _receive_loop(self) -> None:
        """Receive and process incoming WebSocket messages."""
        while self._running and self._ws:
            try:
                message = await self._ws.recv()
                await self._process_message(message)
            except asyncio.CancelledError:
                break
            except websockets.ConnectionClosed:
                self.logger.warning("WebSocket connection closed")
                await self._handle_reconnect()
                break
            except Exception as e:
                self.logger.error("Error receiving message", error=str(e))

    async def _handle_reconnect(self) -> None:
        """Handle reconnection after connection loss."""
        if not self._running:
            return

        self.logger.info("Attempting to reconnect", delay=self._reconnect_delay)
        await asyncio.sleep(self._reconnect_delay)

        try:
            await self.connect()
            # Resubscribe to previous topics
            for topic in list(self._subscriptions):
                await self._send_subscription(topic, True)
        except Exception as e:
            self.logger.error("Reconnection failed", error=str(e))
            await self._handle_reconnect()

    async def _process_message(self, message: str | bytes) -> None:
        """Process incoming WebSocket message.

        Args:
            message: Raw message data
        """
        try:
            data = orjson.loads(message)
            msg_type = data.get("type")

            if msg_type == "pong":
                return
            elif msg_type == "ack":
                self.logger.debug("Subscription acknowledged", id=data.get("id"))
                return
            elif msg_type == "message":
                await self._dispatch_message(data)
        except Exception as e:
            self.logger.error("Error processing message", error=str(e))

    async def _dispatch_message(self, data: dict[str, Any]) -> None:
        """Dispatch message to registered callbacks.

        Args:
            data: Parsed message data
        """
        topic = data.get("topic", "")
        subject = data.get("subject", "")
        payload = data.get("data", {})

        # Determine callback key
        callback_key = topic
        if subject:
            callback_key = f"{topic}:{subject}"

        # Call registered callbacks
        callbacks = self._callbacks.get(topic, []) + self._callbacks.get(callback_key, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(payload)
                else:
                    callback(payload)
            except Exception as e:
                self.logger.error("Callback error", error=str(e), topic=topic)

    async def _send_subscription(self, topic: str, subscribe: bool = True) -> None:
        """Send subscription/unsubscription message.

        Args:
            topic: Topic to subscribe/unsubscribe
            subscribe: True to subscribe, False to unsubscribe
        """
        if not self._ws:
            raise ConnectionError("WebSocket not connected")

        self._connect_id += 1
        message = orjson.dumps({
            "id": str(self._connect_id),
            "type": "subscribe" if subscribe else "unsubscribe",
            "topic": topic,
            "privateChannel": False,
            "response": True,
        })

        await self._ws.send(message)

        if subscribe:
            self._subscriptions.add(topic)
        else:
            self._subscriptions.discard(topic)

        self.logger.info(
            "Subscription updated",
            topic=topic,
            action="subscribe" if subscribe else "unsubscribe",
        )

    def on(self, topic: str, callback: Callable[[Any], None]) -> None:
        """Register a callback for a topic.

        Args:
            topic: Topic to listen for
            callback: Function to call when message received
        """
        if topic not in self._callbacks:
            self._callbacks[topic] = []
        self._callbacks[topic].append(callback)

    def off(self, topic: str, callback: Callable[[Any], None] | None = None) -> None:
        """Remove callback(s) for a topic.

        Args:
            topic: Topic to remove callbacks from
            callback: Specific callback to remove, or None to remove all
        """
        if topic in self._callbacks:
            if callback:
                self._callbacks[topic] = [cb for cb in self._callbacks[topic] if cb != callback]
            else:
                del self._callbacks[topic]

    # ==================== Market Data Subscriptions ====================

    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Ticker], None],
    ) -> None:
        """Subscribe to ticker updates.

        Args:
            symbol: Trading pair symbol
            callback: Function to call with ticker updates
        """
        topic = f"/market/ticker:{symbol}"

        def parse_ticker(data: dict[str, Any]) -> None:
            ticker = Ticker(
                symbol=symbol,
                price=Decimal(str(data.get("price", "0"))),
                best_bid=Decimal(str(data.get("bestBid", "0"))),
                best_ask=Decimal(str(data.get("bestAsk", "0"))),
                volume_24h=Decimal(str(data.get("size", "0"))),
                change_24h=Decimal("0"),
                high_24h=Decimal("0"),
                low_24h=Decimal("0"),
                timestamp=data.get("time", 0),
            )
            callback(ticker)

        self.on(topic, parse_ticker)
        await self._send_subscription(topic)

    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable[[Trade], None],
    ) -> None:
        """Subscribe to trade updates.

        Args:
            symbol: Trading pair symbol
            callback: Function to call with trade updates
        """
        topic = f"/market/match:{symbol}"

        def parse_trade(data: dict[str, Any]) -> None:
            from datetime import datetime

            from kucoin_bot.models.data_models import OrderSide

            trade = Trade(
                id=data.get("sequence", ""),
                order_id="",
                symbol=symbol,
                side=OrderSide(data.get("side", "buy")),
                price=Decimal(str(data.get("price", "0"))),
                size=Decimal(str(data.get("size", "0"))),
                fee=Decimal("0"),
                fee_currency="USDT",
                timestamp=datetime.fromtimestamp(data.get("time", 0) / 1000000000),
                is_maker=data.get("makerOrderId") == data.get("takerOrderId"),
            )
            callback(trade)

        self.on(topic, parse_trade)
        await self._send_subscription(topic)

    async def subscribe_orderbook(
        self,
        symbol: str,
        callback: Callable[[MarketDepth], None],
        depth: int = 5,
    ) -> None:
        """Subscribe to order book updates.

        Args:
            symbol: Trading pair symbol
            callback: Function to call with order book updates
            depth: Order book depth level
        """
        topic = f"/spotMarket/level2Depth{depth}:{symbol}"

        def parse_orderbook(data: dict[str, Any]) -> None:
            from datetime import datetime

            bids = [(Decimal(p), Decimal(s)) for p, s in data.get("bids", [])]
            asks = [(Decimal(p), Decimal(s)) for p, s in data.get("asks", [])]

            depth_data = MarketDepth(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.fromtimestamp(data.get("timestamp", 0) / 1000),
            )
            callback(depth_data)

        self.on(topic, parse_orderbook)
        await self._send_subscription(topic)

    async def subscribe_candles(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Candle], None],
    ) -> None:
        """Subscribe to candle/kline updates.

        Args:
            symbol: Trading pair symbol
            interval: Candle interval (1min, 5min, etc.)
            callback: Function to call with candle updates
        """
        topic = f"/market/candles:{symbol}_{interval}"

        def parse_candle(data: dict[str, Any]) -> None:
            from datetime import datetime

            candles = data.get("candles", [])
            if candles:
                candle = Candle(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(int(candles[0])),
                    open=Decimal(candles[1]),
                    close=Decimal(candles[2]),
                    high=Decimal(candles[3]),
                    low=Decimal(candles[4]),
                    volume=Decimal(candles[5]),
                )
                callback(candle)

        self.on(topic, parse_candle)
        await self._send_subscription(topic)

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic.

        Args:
            topic: Topic to unsubscribe from
        """
        await self._send_subscription(topic, subscribe=False)
        self.off(topic)
