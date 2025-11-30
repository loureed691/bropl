"""KuCoin API client with async support and proper authentication."""

import base64
import hashlib
import hmac
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

import aiohttp
import orjson
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from kucoin_bot.config import Settings
from kucoin_bot.models.data_models import (
    AccountBalance,
    Candle,
    MarketDepth,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
    TimeInForce,
    Trade,
)

logger = structlog.get_logger()

# Conversion constant for nanoseconds to seconds
NANOSECONDS_PER_SECOND = 1_000_000_000


class KuCoinAPIError(Exception):
    """Custom exception for KuCoin API errors."""

    def __init__(self, code: str, message: str, status_code: int = 0) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"KuCoin API Error [{code}]: {message}")


class KuCoinClient:
    """Async KuCoin API client with authentication and rate limiting."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the KuCoin API client.

        Args:
            settings: Application settings containing API credentials
        """
        self.settings = settings
        self.base_url = settings.base_url
        self._session: aiohttp.ClientSession | None = None
        self._api_key = settings.kucoin.api_key
        self._api_secret = settings.kucoin.api_secret
        self._api_passphrase = settings.kucoin.api_passphrase
        self.logger = logger.bind(component="kucoin_client")

    async def __aenter__(self) -> "KuCoinClient":
        """Async context manager entry."""
        await self._create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Async context manager exit."""
        await self.close()

    async def _create_session(self) -> None:
        """Create aiohttp session with proper settings."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                json_serialize=lambda x: orjson.dumps(x).decode(),
            )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _generate_signature(
        self,
        timestamp: str,
        method: str,
        endpoint: str,
        body: str = "",
    ) -> tuple[str, str]:
        """Generate API request signature.

        Args:
            timestamp: Request timestamp in milliseconds
            method: HTTP method
            endpoint: API endpoint path
            body: Request body as string

        Returns:
            Tuple of (signature, encrypted_passphrase)
        """
        # Build the signature string
        str_to_sign = f"{timestamp}{method}{endpoint}{body}"

        # Get secret value safely
        secret = self._api_secret.get_secret_value()
        passphrase = self._api_passphrase.get_secret_value()

        # Generate signature
        signature = base64.b64encode(
            hmac.new(
                secret.encode("utf-8"),
                str_to_sign.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        # Encrypt passphrase
        encrypted_passphrase = base64.b64encode(
            hmac.new(
                secret.encode("utf-8"),
                passphrase.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        return signature, encrypted_passphrase

    def _get_headers(
        self,
        method: str,
        endpoint: str,
        body: str = "",
    ) -> dict[str, str]:
        """Generate authenticated headers for API requests.

        Args:
            method: HTTP method
            endpoint: API endpoint path
            body: Request body as string

        Returns:
            Dictionary of HTTP headers
        """
        timestamp = str(int(time.time() * 1000))
        signature, encrypted_passphrase = self._generate_signature(
            timestamp, method, endpoint, body
        )

        return {
            "KC-API-KEY": self._api_key.get_secret_value(),
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": encrypted_passphrase,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        """Make an API request with retry logic.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            authenticated: Whether to include authentication headers

        Returns:
            API response data

        Raises:
            KuCoinAPIError: If API returns an error
        """
        await self._create_session()
        assert self._session is not None

        url = f"{self.base_url}{endpoint}"
        body = orjson.dumps(data).decode() if data else ""

        headers = {}
        if authenticated:
            headers = self._get_headers(method, endpoint, body)

        self.logger.debug(
            "API request",
            method=method,
            endpoint=endpoint,
            params=params,
        )

        async with self._session.request(
            method=method,
            url=url,
            params=params,
            data=body if data else None,
            headers=headers,
        ) as response:
            response_text = await response.text()

            try:
                result = orjson.loads(response_text)
            except orjson.JSONDecodeError as e:
                raise KuCoinAPIError(
                    code="PARSE_ERROR",
                    message=f"Failed to parse response: {response_text}",
                    status_code=response.status,
                ) from e

            if result.get("code") != "200000":
                raise KuCoinAPIError(
                    code=result.get("code", "UNKNOWN"),
                    message=result.get("msg", "Unknown error"),
                    status_code=response.status,
                )

            return result.get("data", {})

    # ==================== Market Data Endpoints ====================

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker data for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC-USDT")

        Returns:
            Ticker data
        """
        data = await self._request(
            "GET",
            f"/api/v1/market/orderbook/level1?symbol={symbol}",
            authenticated=False,
        )

        return Ticker(
            symbol=symbol,
            price=Decimal(str(data.get("price", "0"))),
            best_bid=Decimal(str(data.get("bestBid", "0"))),
            best_ask=Decimal(str(data.get("bestAsk", "0"))),
            volume_24h=Decimal(str(data.get("size", "0"))),
            change_24h=Decimal("0"),  # Not in this endpoint
            high_24h=Decimal("0"),  # Not in this endpoint
            low_24h=Decimal("0"),  # Not in this endpoint
            timestamp=datetime.fromtimestamp(data.get("time", 0) / 1000),
        )

    async def get_24h_stats(self, symbol: str) -> dict[str, Any]:
        """Get 24-hour statistics for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            24-hour statistics
        """
        return await self._request(
            "GET",
            f"/api/v1/market/stats?symbol={symbol}",
            authenticated=False,
        )

    async def get_candles(
        self,
        symbol: str,
        interval: str = "1hour",
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Candle]:
        """Get historical candle data.

        Args:
            symbol: Trading pair symbol
            interval: Candle interval (1min, 5min, 15min, 30min, 1hour, 4hour, 1day)
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds

        Returns:
            List of candle data
        """
        params: dict[str, str] = {"symbol": symbol, "type": interval}
        if start_time:
            params["startAt"] = str(start_time)
        if end_time:
            params["endAt"] = str(end_time)

        endpoint = "/api/v1/market/candles"
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        data = await self._request(
            "GET",
            f"{endpoint}?{query_string}",
            authenticated=False,
        )

        candles = []
        for candle_data in data:
            candles.append(
                Candle(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(int(candle_data[0])),
                    open=Decimal(candle_data[1]),
                    close=Decimal(candle_data[2]),
                    high=Decimal(candle_data[3]),
                    low=Decimal(candle_data[4]),
                    volume=Decimal(candle_data[5]),
                )
            )

        return candles

    async def get_order_book(
        self,
        symbol: str,
        depth: int = 20,
    ) -> MarketDepth:
        """Get order book depth.

        Args:
            symbol: Trading pair symbol
            depth: Number of levels (20 or 100)

        Returns:
            Market depth data
        """
        endpoint = f"/api/v1/market/orderbook/level2_{depth}?symbol={symbol}"
        data = await self._request("GET", endpoint, authenticated=False)

        bids = [(Decimal(price), Decimal(size)) for price, size in data.get("bids", [])]
        asks = [(Decimal(price), Decimal(size)) for price, size in data.get("asks", [])]

        return MarketDepth(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.fromtimestamp(data.get("time", 0) / 1000),
        )

    # ==================== Account Endpoints ====================

    async def get_accounts(self, currency: str | None = None) -> list[AccountBalance]:
        """Get account balances.

        Args:
            currency: Optional currency filter

        Returns:
            List of account balances
        """
        endpoint = "/api/v1/accounts"
        if currency:
            endpoint += f"?currency={currency}"

        data = await self._request("GET", endpoint)

        balances = []
        for account in data:
            if account.get("type") == "trade":
                balances.append(
                    AccountBalance(
                        currency=account["currency"],
                        available=Decimal(account["available"]),
                        holds=Decimal(account["holds"]),
                        total=Decimal(account["balance"]),
                    )
                )

        return balances

    async def get_account_balance(self, currency: str) -> AccountBalance | None:
        """Get balance for a specific currency.

        Args:
            currency: Currency code (e.g., "USDT")

        Returns:
            Account balance or None if not found
        """
        balances = await self.get_accounts(currency)
        return balances[0] if balances else None

    # ==================== Order Endpoints ====================

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        size: Decimal,
        price: Decimal | None = None,
        client_order_id: str | None = None,
        stop_price: Decimal | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> Order:
        """Place a new order.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            order_type: Order type (market/limit)
            size: Order size
            price: Limit price (required for limit orders)
            client_order_id: Client-assigned order ID
            stop_price: Stop price for stop orders
            time_in_force: Time in force option

        Returns:
            Created order

        Raises:
            KuCoinAPIError: If order creation fails
        """
        if client_order_id is None:
            client_order_id = f"bot_{int(time.time() * 1000)}"

        order_data: dict[str, Any] = {
            "clientOid": client_order_id,
            "side": side.value,
            "symbol": symbol,
            "type": order_type.value,
        }

        if order_type == OrderType.LIMIT:
            if price is None:
                raise ValueError("Price is required for limit orders")
            order_data["price"] = str(price)
            order_data["size"] = str(size)
            order_data["timeInForce"] = time_in_force.value
        else:
            # Market orders use funds for buy and size for sell
            if side == OrderSide.BUY:
                order_data["funds"] = str(size)
            else:
                order_data["size"] = str(size)

        if stop_price:
            order_data["stop"] = "loss" if side == OrderSide.SELL else "entry"
            order_data["stopPrice"] = str(stop_price)

        data = await self._request("POST", "/api/v1/orders", data=order_data)

        self.logger.info(
            "Order placed",
            order_id=data.get("orderId"),
            symbol=symbol,
            side=side.value,
            type=order_type.value,
            size=str(size),
        )

        return Order(
            id=data.get("orderId"),
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            size=size,
            stop_price=stop_price,
            time_in_force=time_in_force,
            status=OrderStatus.OPEN,
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        try:
            await self._request("DELETE", f"/api/v1/orders/{order_id}")
            self.logger.info("Order cancelled", order_id=order_id)
            return True
        except KuCoinAPIError as e:
            self.logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            return False

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all orders, optionally filtered by symbol.

        Args:
            symbol: Optional trading pair filter

        Returns:
            Number of cancelled orders
        """
        endpoint = "/api/v1/orders"
        if symbol:
            endpoint += f"?symbol={symbol}"

        data = await self._request("DELETE", endpoint)
        cancelled = len(data.get("cancelledOrderIds", []))
        self.logger.info("Orders cancelled", count=cancelled, symbol=symbol)
        return cancelled

    async def get_order(self, order_id: str) -> Order | None:
        """Get order details by ID.

        Args:
            order_id: Order ID

        Returns:
            Order details or None if not found
        """
        try:
            data = await self._request("GET", f"/api/v1/orders/{order_id}")
            return self._parse_order(data)
        except KuCoinAPIError:
            return None

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get list of open orders.

        Args:
            symbol: Optional trading pair filter

        Returns:
            List of open orders
        """
        endpoint = "/api/v1/orders?status=active"
        if symbol:
            endpoint += f"&symbol={symbol}"

        data = await self._request("GET", endpoint)
        return [self._parse_order(order) for order in data.get("items", [])]

    async def get_order_history(
        self,
        symbol: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Order]:
        """Get order history.

        Args:
            symbol: Optional trading pair filter
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            List of historical orders
        """
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if start_time:
            params.append(f"startAt={start_time}")
        if end_time:
            params.append(f"endAt={end_time}")

        endpoint = "/api/v1/orders"
        if params:
            endpoint += "?" + "&".join(params)

        data = await self._request("GET", endpoint)
        return [self._parse_order(order) for order in data.get("items", [])]

    def _parse_order(self, data: dict[str, Any]) -> Order:
        """Parse order data from API response.

        Args:
            data: Raw order data from API

        Returns:
            Parsed Order object
        """
        status_map = {
            "active": OrderStatus.OPEN,
            "done": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
        }

        order_type = OrderType.MARKET
        if data.get("type") == "limit":
            order_type = OrderType.LIMIT
        elif data.get("stop"):
            order_type = OrderType.STOP

        return Order(
            id=data.get("id"),
            client_order_id=data.get("clientOid", ""),
            symbol=data.get("symbol", ""),
            side=OrderSide(data.get("side", "buy")),
            order_type=order_type,
            price=Decimal(data["price"]) if data.get("price") else None,
            size=Decimal(data.get("size", "0")),
            stop_price=Decimal(data["stopPrice"]) if data.get("stopPrice") else None,
            time_in_force=TimeInForce(data.get("timeInForce", "GTC")),
            status=status_map.get(data.get("isActive") and "active" or "done", OrderStatus.OPEN),
            filled_size=Decimal(data.get("dealSize", "0")),
            filled_price=Decimal(data["dealFunds"]) / Decimal(data["dealSize"])
            if data.get("dealSize") and Decimal(data.get("dealSize", "0")) > 0
            else None,
            fee=Decimal(data.get("fee", "0")),
            fee_currency=data.get("feeCurrency", "USDT"),
            created_at=datetime.fromtimestamp(data.get("createdAt", 0) / 1000),
        )

    # ==================== Trade Endpoints ====================

    async def get_recent_trades(self, symbol: str) -> list[Trade]:
        """Get recent trades for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            List of recent trades
        """
        data = await self._request(
            "GET",
            f"/api/v1/market/histories?symbol={symbol}",
            authenticated=False,
        )

        trades = []
        for trade_data in data:
            trades.append(
                Trade(
                    id=trade_data.get("sequence", ""),
                    order_id="",
                    symbol=symbol,
                    side=OrderSide(trade_data.get("side", "buy")),
                    price=Decimal(trade_data.get("price", "0")),
                    size=Decimal(trade_data.get("size", "0")),
                    fee=Decimal("0"),
                    fee_currency="USDT",
                    timestamp=datetime.fromtimestamp(
                        trade_data.get("time", 0) / NANOSECONDS_PER_SECOND
                    ),
                    is_maker=False,
                )
            )

        return trades

    async def get_my_trades(
        self,
        symbol: str | None = None,
        order_id: str | None = None,
    ) -> list[Trade]:
        """Get user's trade history.

        Args:
            symbol: Optional trading pair filter
            order_id: Optional order ID filter

        Returns:
            List of user's trades
        """
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if order_id:
            params.append(f"orderId={order_id}")

        endpoint = "/api/v1/fills"
        if params:
            endpoint += "?" + "&".join(params)

        data = await self._request("GET", endpoint)

        trades = []
        for trade_data in data.get("items", []):
            trades.append(
                Trade(
                    id=trade_data.get("tradeId", ""),
                    order_id=trade_data.get("orderId", ""),
                    symbol=trade_data.get("symbol", ""),
                    side=OrderSide(trade_data.get("side", "buy")),
                    price=Decimal(trade_data.get("price", "0")),
                    size=Decimal(trade_data.get("size", "0")),
                    fee=Decimal(trade_data.get("fee", "0")),
                    fee_currency=trade_data.get("feeCurrency", "USDT"),
                    timestamp=datetime.fromtimestamp(trade_data.get("createdAt", 0) / 1000),
                    is_maker=trade_data.get("liquidity") == "maker",
                )
            )

        return trades

    # ==================== Utility Methods ====================

    async def get_server_time(self) -> datetime:
        """Get KuCoin server time.

        Returns:
            Server timestamp
        """
        data = await self._request("GET", "/api/v1/timestamp", authenticated=False)
        return datetime.fromtimestamp(data / 1000)

    async def get_symbols(self) -> list[dict[str, Any]]:
        """Get list of available trading symbols.

        Returns:
            List of symbol information
        """
        return await self._request("GET", "/api/v1/symbols", authenticated=False)

    async def ping(self) -> bool:
        """Check API connectivity.

        Returns:
            True if API is reachable
        """
        try:
            await self.get_server_time()
            return True
        except Exception:
            return False
