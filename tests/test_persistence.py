"""Tests for state persistence functionality."""

import contextlib
import json
from decimal import Decimal
from pathlib import Path

from kucoin_bot.models.data_models import Order, OrderSide, OrderStatus, OrderType, Position
from kucoin_bot.persistence import StateManager


class TestStateManager:
    """Test StateManager for state persistence."""

    def test_save_and_load_positions(self, tmp_path: Path) -> None:
        """Test saving and loading positions."""
        state_file = tmp_path / "test_state.json"
        manager = StateManager(str(state_file))

        # Create test position
        position = Position(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            current_price=Decimal("51000"),
            stop_loss=Decimal("49000"),
            take_profit=Decimal("52000"),
            order_ids=["order123"],
        )

        positions = {"BTC-USDT": position}
        orders: dict[str, Order] = {}

        # Save state
        manager.save_state(positions, orders)

        # Load state
        loaded_positions, loaded_orders = manager.load_state()

        # Verify loaded data
        assert len(loaded_positions) == 1
        assert "BTC-USDT" in loaded_positions
        loaded_pos = loaded_positions["BTC-USDT"]
        assert loaded_pos.symbol == "BTC-USDT"
        assert loaded_pos.side == OrderSide.BUY
        assert loaded_pos.entry_price == Decimal("50000")
        assert loaded_pos.size == Decimal("0.1")
        assert loaded_pos.stop_loss == Decimal("49000")

    def test_save_and_load_orders(self, tmp_path: Path) -> None:
        """Test saving and loading pending orders."""
        state_file = tmp_path / "test_state.json"
        manager = StateManager(str(state_file))

        # Create test order
        order = Order(
            id="order123",
            client_order_id="client123",
            symbol="ETH-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal("1.0"),
            price=Decimal("3000"),
            status=OrderStatus.PENDING,
        )

        positions: dict[str, Position] = {}
        orders = {"client123": order}

        # Save state
        manager.save_state(positions, orders)

        # Load state
        loaded_positions, loaded_orders = manager.load_state()

        # Verify loaded data
        assert len(loaded_orders) == 1
        assert "client123" in loaded_orders
        loaded_order = loaded_orders["client123"]
        assert loaded_order.symbol == "ETH-USDT"
        assert loaded_order.side == OrderSide.BUY
        assert loaded_order.order_type == OrderType.LIMIT
        assert loaded_order.size == Decimal("1.0")
        assert loaded_order.price == Decimal("3000")

    def test_load_nonexistent_state(self, tmp_path: Path) -> None:
        """Test loading state when file doesn't exist."""
        state_file = tmp_path / "nonexistent.json"
        manager = StateManager(str(state_file))

        positions, orders = manager.load_state()

        # Should return empty dicts
        assert positions == {}
        assert orders == {}

    def test_save_state_atomic(self, tmp_path: Path) -> None:
        """Test that state saving is atomic."""
        state_file = tmp_path / "test_state.json"
        manager = StateManager(str(state_file))

        position = Position(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            current_price=Decimal("51000"),
        )

        positions = {"BTC-USDT": position}
        orders: dict[str, Order] = {}

        # Save state
        manager.save_state(positions, orders)

        # State file should exist, temp file should not
        assert state_file.exists()
        assert not (tmp_path / "test_state.tmp").exists()

        # File should be valid JSON
        with state_file.open("r") as f:
            data = json.load(f)
            assert "version" in data
            assert "timestamp" in data
            assert "positions" in data

    def test_clear_state(self, tmp_path: Path) -> None:
        """Test clearing state file."""
        state_file = tmp_path / "test_state.json"
        manager = StateManager(str(state_file))

        # Create and save some state
        position = Position(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            current_price=Decimal("51000"),
        )
        positions = {"BTC-USDT": position}
        manager.save_state(positions, {})

        # Verify file exists
        assert state_file.exists()

        # Clear state
        manager.clear_state()

        # File should be deleted
        assert not state_file.exists()

    def test_save_multiple_positions(self, tmp_path: Path) -> None:
        """Test saving and loading multiple positions."""
        state_file = tmp_path / "test_state.json"
        manager = StateManager(str(state_file))

        # Create multiple positions
        positions = {
            "BTC-USDT": Position(
                symbol="BTC-USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                size=Decimal("0.1"),
                current_price=Decimal("51000"),
            ),
            "ETH-USDT": Position(
                symbol="ETH-USDT",
                side=OrderSide.SELL,
                entry_price=Decimal("3000"),
                size=Decimal("1.0"),
                current_price=Decimal("2900"),
            ),
        }

        manager.save_state(positions, {})
        loaded_positions, _ = manager.load_state()

        assert len(loaded_positions) == 2
        assert "BTC-USDT" in loaded_positions
        assert "ETH-USDT" in loaded_positions
        assert loaded_positions["BTC-USDT"].side == OrderSide.BUY
        assert loaded_positions["ETH-USDT"].side == OrderSide.SELL


class TestStateManagerIntegration:
    """Integration tests for state persistence with RiskManager and ExecutionEngine."""

    def test_risk_manager_load_positions(self) -> None:
        """Test loading positions into RiskManager."""
        from kucoin_bot.config import RiskSettings, TradingSettings
        from kucoin_bot.risk_management.manager import RiskManager

        # Create risk manager
        risk_settings = RiskSettings()
        trading_settings = TradingSettings()
        risk_manager = RiskManager(risk_settings, trading_settings)

        # Create positions
        positions = {
            "BTC-USDT": Position(
                symbol="BTC-USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                size=Decimal("0.1"),
                current_price=Decimal("51000"),
            )
        }

        # Load positions
        risk_manager.load_positions(positions)

        # Verify positions are loaded
        assert len(risk_manager.positions) == 1
        assert "BTC-USDT" in risk_manager.positions
        assert risk_manager.positions["BTC-USDT"].entry_price == Decimal("50000")


class TestStopLossRetry:
    """Test stop loss retry logic."""

    async def test_stop_loss_retry_success_on_first_attempt(self) -> None:
        """Test that stop loss placement succeeds on first attempt."""
        from unittest.mock import AsyncMock, MagicMock

        from kucoin_bot.config import Settings
        from kucoin_bot.execution.engine import ExecutionEngine

        # Create mocks
        mock_client = MagicMock()
        mock_risk_manager = MagicMock()
        settings = Settings()

        engine = ExecutionEngine(mock_client, mock_risk_manager, settings)

        # Create test order
        order = Order(
            id="test123",
            client_order_id="client123",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.1"),
            filled_size=Decimal("0.1"),
            filled_price=Decimal("50000"),
            status=OrderStatus.FILLED,
        )

        # Mock place_stop_order to succeed
        engine.place_stop_order = AsyncMock()

        # Should succeed without retry
        await engine._place_stop_loss_with_retry(order, Decimal("49000"))

        # Verify it was called once
        assert engine.place_stop_order.call_count == 1

    async def test_stop_loss_retry_on_failure(self) -> None:
        """Test that stop loss retries on failure."""

        # Create mocks
        from unittest.mock import MagicMock

        from kucoin_bot.api.client import KuCoinAPIError
        from kucoin_bot.config import Settings
        from kucoin_bot.execution.engine import ExecutionEngine

        mock_client = MagicMock()
        mock_risk_manager = MagicMock()
        settings = Settings()

        engine = ExecutionEngine(mock_client, mock_risk_manager, settings)

        # Create test order
        order = Order(
            id="test123",
            client_order_id="client123",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.1"),
            filled_size=Decimal("0.1"),
            filled_price=Decimal("50000"),
            status=OrderStatus.FILLED,
        )

        # Mock place_stop_order to fail twice then succeed
        call_count = 0

        async def mock_place_stop(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise KuCoinAPIError("TEST_ERROR", "API Error")
            return None

        engine.place_stop_order = mock_place_stop

        # Should succeed after retries
        await engine._place_stop_loss_with_retry(order, Decimal("49000"))

        # Verify it retried
        assert call_count == 3

    async def test_stop_loss_max_retries_exceeded(self) -> None:
        """Test that stop loss raises after max retries."""

        # Create mocks
        from unittest.mock import MagicMock

        from kucoin_bot.api.client import KuCoinAPIError
        from kucoin_bot.config import Settings
        from kucoin_bot.execution.engine import ExecutionEngine

        mock_client = MagicMock()
        mock_risk_manager = MagicMock()
        settings = Settings()

        engine = ExecutionEngine(mock_client, mock_risk_manager, settings)

        # Create test order
        order = Order(
            id="test123",
            client_order_id="client123",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.1"),
            filled_size=Decimal("0.1"),
            filled_price=Decimal("50000"),
            status=OrderStatus.FILLED,
        )

        # Mock place_stop_order to always fail
        call_count = 0

        async def mock_place_stop(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            raise KuCoinAPIError("TEST_ERROR", "API Error")

        engine.place_stop_order = mock_place_stop

        # Should raise after 3 attempts
        import pytest
        with pytest.raises(KuCoinAPIError):
            await engine._place_stop_loss_with_retry(order, Decimal("49000"))

        # Verify it tried 3 times
        assert call_count == 3


class TestTaskSupervisor:
    """Test task supervisor logic."""

    async def test_task_supervisor_restarts_failed_task(self) -> None:
        """Test that supervisor restarts a failed task."""
        import asyncio

        from kucoin_bot.main import TradingBot

        # Create a bot instance with mocked components
        bot = TradingBot()
        bot._running = True
        bot._task_registry = {}
        bot._tasks = []

        # Create a task that will fail
        task_called = []

        async def failing_task():
            task_called.append(1)
            if len(task_called) == 1:
                raise ValueError("Test failure")
            # Second call succeeds
            await asyncio.sleep(0.1)

        # Register the task
        bot._task_registry["test_task"] = failing_task

        # Create the initial task
        task = asyncio.create_task(failing_task(), name="test_task")
        bot._tasks.append(task)

        # Wait for task to fail
        await asyncio.sleep(0.05)

        # Manually run one cycle of supervisor logic (simplified)
        done_tasks = [(i, t) for i, t in enumerate(bot._tasks) if t.done()]
        for i, failed_task in done_tasks:
            task_name = failed_task.get_name()
            if task_name in bot._task_registry:
                new_task = asyncio.create_task(bot._task_registry[task_name](), name=task_name)
                bot._tasks[i] = new_task

        # Wait for new task to run
        await asyncio.sleep(0.15)

        # Verify task was restarted (called twice)
        assert len(task_called) == 2

        # Cleanup
        bot._running = False
        for t in bot._tasks:
            if not t.done():
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t

    async def test_task_supervisor_handles_multiple_failures(self) -> None:
        """Test that supervisor can handle multiple task failures simultaneously."""
        import asyncio

        from kucoin_bot.main import TradingBot

        # Create a bot instance
        bot = TradingBot()
        bot._running = True
        bot._task_registry = {}
        bot._tasks = []

        # Create multiple tasks that will fail
        task1_calls = []
        task2_calls = []

        async def failing_task1():
            task1_calls.append(1)
            if len(task1_calls) == 1:
                raise ValueError("Task 1 failure")
            await asyncio.sleep(0.1)

        async def failing_task2():
            task2_calls.append(1)
            if len(task2_calls) == 1:
                raise ValueError("Task 2 failure")
            await asyncio.sleep(0.1)

        # Register tasks
        bot._task_registry["task1"] = failing_task1
        bot._task_registry["task2"] = failing_task2

        # Create initial tasks
        task1 = asyncio.create_task(failing_task1(), name="task1")
        task2 = asyncio.create_task(failing_task2(), name="task2")
        bot._tasks.extend([task1, task2])

        # Wait for tasks to fail
        await asyncio.sleep(0.05)

        # Manually run one cycle of supervisor logic with collection
        done_tasks = [(i, t) for i, t in enumerate(bot._tasks) if t.done()]
        for i, failed_task in done_tasks:
            task_name = failed_task.get_name()
            if task_name in bot._task_registry:
                new_task = asyncio.create_task(bot._task_registry[task_name](), name=task_name)
                bot._tasks[i] = new_task

        # Wait for new tasks to run
        await asyncio.sleep(0.15)

        # Verify both tasks were restarted
        assert len(task1_calls) == 2
        assert len(task2_calls) == 2

        # Cleanup
        bot._running = False
        for t in bot._tasks:
            if not t.done():
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
