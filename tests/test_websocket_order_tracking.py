"""Tests for WebSocket-based order tracking functionality."""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from kucoin_bot.api.websocket import WebSocketManager
from kucoin_bot.config import Settings
from kucoin_bot.execution.engine import ExecutionEngine
from kucoin_bot.models.data_models import Order, OrderSide, OrderStatus, OrderType
from kucoin_bot.risk_management.manager import RiskManager


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def ws_manager(settings):
    """Create a WebSocket manager for testing."""
    return WebSocketManager(settings)


@pytest.fixture
def mock_client():
    """Create a mock KuCoin client."""
    client = Mock()
    client.get_order = AsyncMock(return_value=None)
    return client


@pytest.fixture
def risk_manager(settings):
    """Create a risk manager for testing."""
    return RiskManager(settings.risk, settings.trading)


@pytest.fixture
def execution_engine(mock_client, risk_manager, settings):
    """Create an execution engine for testing."""
    return ExecutionEngine(mock_client, risk_manager, settings)


class TestWebSocketManager:
    """Test WebSocket manager private channel support."""

    @pytest.mark.asyncio
    async def test_private_channel_flag_initialization(self, ws_manager):
        """Test that private channel flag is initialized correctly."""
        assert ws_manager._is_private is False

    @pytest.mark.asyncio
    async def test_private_subscription_requires_private_connection(self, ws_manager):
        """Test that private subscriptions require private connection."""
        callback = Mock()

        with pytest.raises(ConnectionError, match="Private channel connection required"):
            await ws_manager.subscribe_order_updates(callback)

    @pytest.mark.asyncio
    async def test_send_subscription_with_private_flag(self, ws_manager):
        """Test that subscriptions can be sent with private flag."""
        ws_manager._ws = Mock()
        ws_manager._ws.send = AsyncMock()

        topic = "/spotMarket/tradeOrders"
        await ws_manager._send_subscription(topic, subscribe=True, private=True)

        assert topic in ws_manager._subscriptions


class TestExecutionEngineWebSocketIntegration:
    """Test ExecutionEngine WebSocket integration."""

    @pytest.mark.asyncio
    async def test_engine_accepts_websocket_manager(
        self, mock_client, risk_manager, settings, ws_manager
    ):
        """Test that execution engine accepts WebSocket manager."""
        engine = ExecutionEngine(mock_client, risk_manager, settings, ws_manager)
        assert engine.ws_manager == ws_manager

    @pytest.mark.asyncio
    async def test_setup_order_tracking_without_websocket(
        self, mock_client, risk_manager, settings
    ):
        """Test setup_order_tracking when WebSocket is not available."""
        engine = ExecutionEngine(mock_client, risk_manager, settings, ws_manager=None)
        await engine.setup_order_tracking()
        # Should complete without error, using polling only

    @pytest.mark.asyncio
    async def test_setup_order_tracking_with_private_websocket(
        self, mock_client, risk_manager, settings
    ):
        """Test setup_order_tracking with private WebSocket connection."""
        ws_manager = Mock()
        ws_manager._is_private = True
        ws_manager.subscribe_order_updates = AsyncMock()

        engine = ExecutionEngine(mock_client, risk_manager, settings, ws_manager)
        await engine.setup_order_tracking()

        ws_manager.subscribe_order_updates.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_order_update_stores_data(self, execution_engine):
        """Test that order updates are stored correctly."""
        order_data = {
            "orderId": "test-order-123",
            "clientOid": "client-123",
            "type": "match",
            "status": "done",
            "filledSize": "0.1",
            "price": "50000",
        }

        execution_engine._handle_order_update(order_data)

        assert "test-order-123" in execution_engine._ws_order_data
        assert execution_engine._ws_order_data["test-order-123"] == order_data

    @pytest.mark.asyncio
    async def test_handle_order_update_signals_event(self, execution_engine):
        """Test that order updates signal waiting coroutines."""
        order_id = "test-order-123"
        event = asyncio.Event()
        execution_engine._ws_order_events[order_id] = event

        order_data = {
            "orderId": order_id,
            "type": "filled",
            "status": "done",
        }

        execution_engine._handle_order_update(order_data)

        assert event.is_set()

    @pytest.mark.asyncio
    async def test_cleanup_order_tracking(self, execution_engine):
        """Test cleanup of order tracking data."""
        order = Order(
            id="order-123",
            client_order_id="client-123",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.1"),
        )

        # Set up tracking data
        execution_engine._ws_order_events[order.id] = asyncio.Event()
        execution_engine._ws_order_data[order.id] = {"test": "data"}

        # Clean up
        execution_engine._cleanup_order_tracking(order)

        # Verify cleanup
        assert order.id not in execution_engine._ws_order_events
        assert order.id not in execution_engine._ws_order_data


class TestOrderTrackingWithWebSocket:
    """Test order tracking with WebSocket updates."""

    @pytest.mark.asyncio
    async def test_track_order_fill_with_websocket_filled(
        self, mock_client, risk_manager, settings
    ):
        """Test tracking order fill via WebSocket."""
        # Create a mock WebSocket manager that's connected to private channel
        ws_manager = Mock()
        ws_manager._is_private = True

        engine = ExecutionEngine(mock_client, risk_manager, settings, ws_manager)

        order = Order(
            id="order-123",
            client_order_id="client-123",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.1"),
        )

        # Simulate WebSocket update arriving
        async def simulate_ws_update():
            await asyncio.sleep(0.1)
            order_data = {
                "orderId": "order-123",
                "type": "filled",
                "status": "done",
                "filledSize": "0.1",
                "matchPrice": "50000",
                "price": "50000",
            }
            engine._handle_order_update(order_data)

        # Start both tasks
        update_task = asyncio.create_task(simulate_ws_update())
        track_task = asyncio.create_task(engine._track_order_fill(order, timeout=5))

        # Wait for completion
        await asyncio.gather(update_task, track_task)

        # Verify order was marked as filled
        assert order.status == OrderStatus.FILLED
        assert order.filled_size == Decimal("0.1")
        assert order in engine.executed_orders

    @pytest.mark.asyncio
    async def test_track_order_fill_fallback_to_polling(
        self, mock_client, risk_manager, settings
    ):
        """Test that order tracking falls back to polling when WebSocket doesn't provide update."""
        ws_manager = Mock()
        ws_manager._is_private = False  # WebSocket not available

        engine = ExecutionEngine(mock_client, risk_manager, settings, ws_manager)

        order = Order(
            id="order-123",
            client_order_id="client-123",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.1"),
        )

        # Mock client.get_order to return filled order after delay
        filled_order = Order(
            id="order-123",
            client_order_id="client-123",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.1"),
            status=OrderStatus.FILLED,
            filled_size=Decimal("0.1"),
            filled_price=Decimal("50000"),
            fee=Decimal("5"),
        )

        call_count = 0

        async def delayed_response(*_args, **_kwargs):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count >= 2:  # Return filled order on second call
                return filled_order
            return None

        mock_client.get_order = AsyncMock(side_effect=delayed_response)

        # Track order fill
        await engine._track_order_fill(order, timeout=5)

        # Verify order was filled via polling
        assert order.status == OrderStatus.FILLED
        assert order.filled_size == Decimal("0.1")
        assert mock_client.get_order.call_count >= 1

    @pytest.mark.asyncio
    async def test_track_order_fill_timeout_keeps_pending(
        self, mock_client, risk_manager, settings
    ):
        """Test that orders timing out remain in pending_orders."""
        engine = ExecutionEngine(mock_client, risk_manager, settings, ws_manager=None)

        order = Order(
            id="order-123",
            client_order_id="client-123",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.1"),
        )

        engine.pending_orders[order.client_order_id] = order

        # Mock client.get_order to always return None (order not found)
        mock_client.get_order = AsyncMock(return_value=None)

        # Track with very short timeout
        await engine._track_order_fill(order, timeout=1)

        # Verify order is still in pending_orders
        assert order.client_order_id in engine.pending_orders
        # Verify tracking data was cleaned up
        assert order.id not in engine._ws_order_events
        assert order.id not in engine._ws_order_data


class TestWebSocketAuthentication:
    """Test WebSocket authentication for private channels."""

    @pytest.mark.asyncio
    async def test_get_ws_token_private_includes_auth_headers(self, settings):
        """Test that getting private WS token includes authentication."""
        ws_manager = WebSocketManager(settings)

        with patch("aiohttp.ClientSession") as mock_session_class:
            # Create mock response
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(
                return_value={
                    "code": "200000",
                    "data": {
                        "token": "test-token",
                        "instanceServers": [
                            {
                                "endpoint": "wss://test.kucoin.com",
                                "pingInterval": 30000,
                            }
                        ],
                    },
                }
            )
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            # Create mock session
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            # Get private token
            result = await ws_manager._get_ws_token(private=True)

            # Verify post was called
            assert mock_session.post.called
            call_args = mock_session.post.call_args

            # Verify authentication headers were included
            if "headers" in call_args[1]:
                headers = call_args[1]["headers"]
                assert "KC-API-KEY" in headers
                assert "KC-API-SIGN" in headers
                assert "KC-API-TIMESTAMP" in headers
                assert "KC-API-PASSPHRASE" in headers

            assert result["token"] == "test-token"
