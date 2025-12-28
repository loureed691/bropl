"""State persistence for trading bot positions and orders."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog

from kucoin_bot.models.data_models import Order, OrderSide, OrderStatus, OrderType, Position

logger = structlog.get_logger()


class StateManager:
    """Manages persistent state for positions and orders."""

    def __init__(self, state_file: str = "bot_state.json") -> None:
        """Initialize state manager.

        Args:
            state_file: Path to state file (relative to working directory)
        """
        self.state_file = Path(state_file)
        self.logger = logger.bind(component="state_manager")

    def save_state(
        self,
        positions: dict[str, Position],
        pending_orders: dict[str, Order],
    ) -> None:
        """Save current state to disk.

        Args:
            positions: Dictionary of open positions
            pending_orders: Dictionary of pending orders
        """
        try:
            state = {
                "version": "1.0",
                "timestamp": datetime.now(UTC).isoformat(),
                "positions": self._serialize_positions(positions),
                "pending_orders": self._serialize_orders(pending_orders),
            }

            # Write to temporary file first, then rename for atomicity
            temp_file = self.state_file.with_suffix(".tmp")
            with temp_file.open("w") as f:
                json.dump(state, f, indent=2)

            # Atomic rename
            temp_file.replace(self.state_file)

            self.logger.info(
                "State saved",
                positions=len(positions),
                pending_orders=len(pending_orders),
            )

        except Exception as e:
            self.logger.error("Failed to save state", error=str(e))

    def load_state(self) -> tuple[dict[str, Position], dict[str, Order]]:
        """Load state from disk.

        Returns:
            Tuple of (positions dict, pending_orders dict)
        """
        if not self.state_file.exists():
            self.logger.info("No state file found, starting fresh")
            return {}, {}

        try:
            with self.state_file.open("r") as f:
                state = json.load(f)

            positions = self._deserialize_positions(state.get("positions", {}))
            pending_orders = self._deserialize_orders(state.get("pending_orders", {}))

            self.logger.info(
                "State loaded",
                positions=len(positions),
                pending_orders=len(pending_orders),
                saved_at=state.get("timestamp", "unknown"),
            )

            return positions, pending_orders

        except Exception as e:
            self.logger.error("Failed to load state", error=str(e))
            return {}, {}

    def _serialize_positions(self, positions: dict[str, Position]) -> dict[str, Any]:
        """Serialize positions to JSON-compatible dict.

        Args:
            positions: Dictionary of positions

        Returns:
            JSON-serializable dictionary
        """
        return {
            symbol: {
                "symbol": pos.symbol,
                "side": pos.side.value,
                "entry_price": str(pos.entry_price),
                "size": str(pos.size),
                "current_price": str(pos.current_price),
                "stop_loss": str(pos.stop_loss) if pos.stop_loss else None,
                "take_profit": str(pos.take_profit) if pos.take_profit else None,
                "order_ids": pos.order_ids,
                "opened_at": pos.opened_at.isoformat(),
                "updated_at": pos.updated_at.isoformat() if pos.updated_at else None,
            }
            for symbol, pos in positions.items()
        }

    def _deserialize_positions(self, data: dict[str, Any]) -> dict[str, Position]:
        """Deserialize positions from JSON data.

        Args:
            data: JSON data dictionary

        Returns:
            Dictionary of Position objects
        """
        positions = {}
        for symbol, pos_data in data.items():
            try:
                positions[symbol] = Position(
                    symbol=pos_data["symbol"],
                    side=OrderSide(pos_data["side"]),
                    entry_price=Decimal(pos_data["entry_price"]),
                    size=Decimal(pos_data["size"]),
                    current_price=Decimal(pos_data["current_price"]),
                    stop_loss=Decimal(pos_data["stop_loss"]) if pos_data.get("stop_loss") else None,
                    take_profit=(
                        Decimal(pos_data["take_profit"]) if pos_data.get("take_profit") else None
                    ),
                    order_ids=pos_data.get("order_ids", []),
                    opened_at=datetime.fromisoformat(pos_data["opened_at"]),
                    updated_at=(
                        datetime.fromisoformat(pos_data["updated_at"])
                        if pos_data.get("updated_at")
                        else None
                    ),
                )
            except Exception as e:
                self.logger.warning("Failed to deserialize position", symbol=symbol, error=str(e))

        return positions

    def _serialize_orders(self, orders: dict[str, Order]) -> dict[str, Any]:
        """Serialize orders to JSON-compatible dict.

        Args:
            orders: Dictionary of orders

        Returns:
            JSON-serializable dictionary
        """
        return {
            client_id: {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "size": str(order.size),
                "price": str(order.price) if order.price else None,
                "stop_price": str(order.stop_price) if order.stop_price else None,
                "filled_size": str(order.filled_size),
                "filled_price": str(order.filled_price) if order.filled_price else None,
                "fee": str(order.fee),
                "status": order.status.value,
                "created_at": order.created_at.isoformat(),
                "updated_at": order.updated_at.isoformat() if order.updated_at else None,
            }
            for client_id, order in orders.items()
        }

    def _deserialize_orders(self, data: dict[str, Any]) -> dict[str, Order]:
        """Deserialize orders from JSON data.

        Args:
            data: JSON data dictionary

        Returns:
            Dictionary of Order objects
        """
        orders = {}
        for client_id, order_data in data.items():
            try:
                orders[client_id] = Order(
                    id=order_data.get("id"),
                    client_order_id=order_data["client_order_id"],
                    symbol=order_data["symbol"],
                    side=OrderSide(order_data["side"]),
                    order_type=OrderType(order_data["order_type"]),
                    size=Decimal(order_data["size"]),
                    price=Decimal(order_data["price"]) if order_data.get("price") else None,
                    stop_price=(
                        Decimal(order_data["stop_price"]) if order_data.get("stop_price") else None
                    ),
                    filled_size=Decimal(order_data["filled_size"]),
                    filled_price=(
                        Decimal(order_data["filled_price"])
                        if order_data.get("filled_price")
                        else None
                    ),
                    fee=Decimal(order_data["fee"]),
                    status=OrderStatus(order_data["status"]),
                    created_at=datetime.fromisoformat(order_data["created_at"]),
                    updated_at=(
                        datetime.fromisoformat(order_data["updated_at"])
                        if order_data.get("updated_at")
                        else None
                    ),
                )
            except Exception as e:
                self.logger.warning("Failed to deserialize order", client_id=client_id, error=str(e))

        return orders

    def clear_state(self) -> None:
        """Clear saved state file."""
        if self.state_file.exists():
            try:
                self.state_file.unlink()
                self.logger.info("State file cleared")
            except Exception as e:
                self.logger.error("Failed to clear state", error=str(e))
