"""Tests for state persistence functionality."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

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

    def test_risk_manager_load_positions(self, tmp_path: Path) -> None:
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
