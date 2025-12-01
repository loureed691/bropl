"""Logging configuration using structlog."""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from kucoin_bot.config import LogLevel


def setup_logging(log_level: LogLevel, log_format: str = "json") -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Logging level
        log_format: Output format ("json" or "text")
    """
    # Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.value),
    )

    # Common processors
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Format-specific processors
    if log_format == "json":
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None, **kwargs: Any) -> structlog.stdlib.BoundLogger:
    """Get a logger instance.

    Args:
        name: Logger name
        **kwargs: Additional context to bind

    Returns:
        Configured logger instance
    """
    log: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if kwargs:
        log = log.bind(**kwargs)
    return log
