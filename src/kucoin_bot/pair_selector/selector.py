"""Pair selector for auto-selecting best futures USDT pairs with strongest signals."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog

from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.indicators.technical import SignalGenerator

logger = structlog.get_logger()


@dataclass
class PairScore:
    """Represents the score of a trading pair based on signal strength."""

    symbol: str
    signal_type: str  # "bullish", "bearish", "neutral"
    signal_strength: float  # 0.0 to 1.0
    volume_24h: Decimal
    price_change_24h: float
    volatility: float
    indicators: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def composite_score(self) -> float:
        """Calculate a composite score for ranking pairs.

        Higher score = better trading opportunity.
        """
        # Base score from signal strength
        base_score = self.signal_strength

        # Volume factor (higher volume = more reliable signals)
        # Normalize to a 0-1 scale assuming volume in millions
        volume_factor = min(1.0, float(self.volume_24h) / 1_000_000)

        # Volatility factor (moderate volatility is preferred)
        # Too low = no movement, too high = risky
        volatility_factor = 1.0 - abs(self.volatility - 0.03) / 0.03 if self.volatility > 0 else 0.5
        volatility_factor = max(0.0, min(1.0, volatility_factor))

        # Weighted composite score
        return (base_score * 0.6) + (volume_factor * 0.25) + (volatility_factor * 0.15)


class PairSelector:
    """Selects the best futures USDT trading pairs based on signal strength."""

    def __init__(
        self,
        client: KuCoinClient,
        min_volume_24h: Decimal = Decimal("100000"),
        min_signal_strength: float = 0.5,
        top_pairs_count: int = 5,
    ) -> None:
        """Initialize pair selector.

        Args:
            client: KuCoin API client
            min_volume_24h: Minimum 24h volume in USDT for filtering pairs
            min_signal_strength: Minimum signal strength to consider (0.0 to 1.0)
            top_pairs_count: Number of top pairs to select
        """
        self.client = client
        self.min_volume_24h = min_volume_24h
        self.min_signal_strength = min_signal_strength
        self.top_pairs_count = top_pairs_count
        self.signal_generator = SignalGenerator()
        self.logger = logger.bind(component="pair_selector")
        self._cached_scores: list[PairScore] = []
        self._last_scan_time: datetime | None = None

    async def get_usdt_pairs(self) -> list[str]:
        """Get all available futures USDT trading pairs.

        Returns:
            List of trading pair symbols (e.g., ["BTC-USDT", "ETH-USDT"])
        """
        try:
            symbols = await self.client.get_symbols()
            usdt_pairs = [
                s["symbol"]
                for s in symbols
                if s.get("quoteCurrency") == "USDT"
                and s.get("enableTrading", False)
            ]
            self.logger.info("Found USDT pairs", count=len(usdt_pairs))
            return usdt_pairs
        except Exception as e:
            self.logger.error("Failed to get USDT pairs", error=str(e))
            return []

    async def analyze_pair(self, symbol: str) -> PairScore | None:
        """Analyze a single trading pair for signal strength.

        Args:
            symbol: Trading pair symbol (e.g., "BTC-USDT")

        Returns:
            PairScore with signal analysis or None if analysis failed
        """
        try:
            # Get 24h stats for volume and price change
            stats = await self.client.get_24h_stats(symbol)
            volume_24h = Decimal(str(stats.get("volValue", "0")))

            # Filter by minimum volume
            if volume_24h < self.min_volume_24h:
                return None

            price_change_24h = float(stats.get("changeRate", 0)) * 100

            # Get historical candles for signal analysis
            candles = await self.client.get_candles(
                symbol=symbol,
                interval="1hour",
            )

            if len(candles) < 50:  # Need minimum candles for proper analysis
                return None

            # Extract close prices
            closes = [float(c.close) for c in candles]
            highs = [float(c.high) for c in candles]
            lows = [float(c.low) for c in candles]

            # Calculate volatility (average true range as percentage)
            if len(highs) > 0 and len(lows) > 0:
                avg_price = sum(closes) / len(closes)
                price_range = sum(
                    high - low for high, low in zip(highs, lows, strict=True)
                ) / len(highs)
                volatility = price_range / avg_price if avg_price > 0 else 0
            else:
                volatility = 0.0

            # Get signal analysis
            signal_type, signal_strength = self.signal_generator.get_overall_signal(closes)

            # Get detailed indicators
            signals = self.signal_generator.combined_signal(closes)
            indicators = {
                name: result.value for name, result in signals.items()
            }

            return PairScore(
                symbol=symbol,
                signal_type=signal_type,
                signal_strength=signal_strength,
                volume_24h=volume_24h,
                price_change_24h=price_change_24h,
                volatility=volatility,
                indicators=indicators,
            )

        except Exception as e:
            self.logger.debug("Failed to analyze pair", symbol=symbol, error=str(e))
            return None

    async def scan_all_pairs(self) -> list[PairScore]:
        """Scan all USDT pairs and return sorted scores.

        Returns:
            List of PairScore sorted by composite score (highest first)
        """
        self.logger.info("Starting pair scan")

        usdt_pairs = await self.get_usdt_pairs()
        if not usdt_pairs:
            self.logger.warning("No USDT pairs found")
            return []

        scores: list[PairScore] = []

        for symbol in usdt_pairs:
            score = await self.analyze_pair(symbol)
            if score and score.signal_strength >= self.min_signal_strength:
                scores.append(score)
                self.logger.debug(
                    "Pair analyzed",
                    symbol=symbol,
                    signal_type=score.signal_type,
                    strength=f"{score.signal_strength:.2f}",
                    composite=f"{score.composite_score:.2f}",
                )

        # Sort by composite score (highest first)
        scores.sort(key=lambda x: x.composite_score, reverse=True)

        self._cached_scores = scores
        self._last_scan_time = datetime.now(UTC)

        self.logger.info(
            "Pair scan completed",
            total_analyzed=len(usdt_pairs),
            qualifying_pairs=len(scores),
        )

        return scores

    async def get_top_pairs(
        self,
        signal_type: str | None = None,
        count: int | None = None,
    ) -> list[str]:
        """Get the top trading pairs with strongest signals.

        Args:
            signal_type: Optional filter by signal type ("bullish" or "bearish")
            count: Number of pairs to return (defaults to top_pairs_count)

        Returns:
            List of trading pair symbols sorted by signal strength
        """
        scores = await self.scan_all_pairs()

        if signal_type:
            scores = [s for s in scores if s.signal_type == signal_type]

        top_count = count or self.top_pairs_count
        top_scores = scores[:top_count]

        pairs = [s.symbol for s in top_scores]

        self.logger.info(
            "Top pairs selected",
            pairs=pairs,
            signal_type=signal_type or "any",
        )

        return pairs

    async def get_bullish_pairs(self, count: int | None = None) -> list[str]:
        """Get the top bullish trading pairs.

        Args:
            count: Number of pairs to return

        Returns:
            List of trading pair symbols with bullish signals
        """
        return await self.get_top_pairs(signal_type="bullish", count=count)

    async def get_bearish_pairs(self, count: int | None = None) -> list[str]:
        """Get the top bearish trading pairs.

        Args:
            count: Number of pairs to return

        Returns:
            List of trading pair symbols with bearish signals
        """
        return await self.get_top_pairs(signal_type="bearish", count=count)

    def get_cached_scores(self) -> list[PairScore]:
        """Get the cached pair scores from the last scan.

        Returns:
            List of PairScore from the last scan
        """
        return self._cached_scores

    def get_pair_details(self, symbol: str) -> PairScore | None:
        """Get detailed score for a specific pair from cache.

        Args:
            symbol: Trading pair symbol

        Returns:
            PairScore for the symbol or None if not found
        """
        for score in self._cached_scores:
            if score.symbol == symbol:
                return score
        return None

    def get_scan_summary(self) -> dict[str, Any]:
        """Get a summary of the last pair scan.

        Returns:
            Dictionary with scan summary information
        """
        if not self._cached_scores:
            return {
                "last_scan": None,
                "total_pairs": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "top_pairs": [],
            }

        bullish = [s for s in self._cached_scores if s.signal_type == "bullish"]
        bearish = [s for s in self._cached_scores if s.signal_type == "bearish"]
        neutral = [s for s in self._cached_scores if s.signal_type == "neutral"]

        return {
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "total_pairs": len(self._cached_scores),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "neutral_count": len(neutral),
            "top_pairs": [
                {
                    "symbol": s.symbol,
                    "signal_type": s.signal_type,
                    "strength": round(s.signal_strength, 3),
                    "composite_score": round(s.composite_score, 3),
                }
                for s in self._cached_scores[: self.top_pairs_count]
            ],
        }
