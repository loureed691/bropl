"""Pair selection module for auto-selecting best trading pairs."""

from kucoin_bot.pair_selector.selector import PairScore, PairSelector, select_best_strategy

__all__ = ["PairSelector", "PairScore", "select_best_strategy"]
