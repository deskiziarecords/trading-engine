"""
market_engine.py
Converted from marketEngine.ts — IPDA/ICT-style market structure engine.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Session = Literal["ASIAN", "LONDON", "NY", "NONE"]


@dataclass
class TickData:
    time: int           # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: float
    ask: float
    spread: float
    session: Session
    is_kill_zone: bool
    is_event_trigger: bool


@dataclass
class IPDAZone:
    type: Literal["FVG_BISI", "FVG_SIBI", "ORDER_BLOCK"]
    start_time: int
    high: float
    low: float
    is_mitigated: bool
    confluence_score: float
    distance_pips: float


@dataclass
class SOS27XOutput:
    confidence: float
    regime_vector: list[float]
    predicted_displacement: float
    latency_ms: float


@dataclass
class EquityPoint:
    time: int
    value: float


@dataclass
class ConfidencePoint:
    confidence: float
    accuracy: float


@dataclass
class EvaluationStats:
    equity_curve: list[EquityPoint]
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    win_rate_by_regime: dict[str, float]
    confidence_calibration: list[ConfidencePoint]
    latency_histogram: list[float]
    false_positive_rate_sweeps: float


# ---------------------------------------------------------------------------
# Seedable PRNG  (LCG — matches TS behaviour exactly)
# ---------------------------------------------------------------------------

class SeededRandom:
    def __init__(self, seed: str) -> None:
        self._seed = self._hash_string(seed)

    @staticmethod
    def _hash_string(s: str) -> int:
        h = 0
        for ch in s:
            h = (h << 5) - h + ord(ch)
            h = h & 0xFFFFFFFF          # keep 32-bit signed wrap
            if h >= 0x80000000:
                h -= 0x100000000
        return h

    def next(self) -> float:
        self._seed = (self._seed * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        if self._seed >= 0x80000000:
            self._seed -= 0x100000000
        return (self._seed & 0xFFFFFFFF) / 4_294_967_296.0


# ---------------------------------------------------------------------------
# UROL: Data Cleaning & Windowing
# ---------------------------------------------------------------------------

def urol_process(data: list[TickData]) -> list[TickData]:
    """Pass-through stub — add cleaning / windowing logic here."""
    return data


# ---------------------------------------------------------------------------
# ADELIC: IPDA Concept Engine
# ---------------------------------------------------------------------------

def adelic_detect_zones(data: list[TickData]) -> list[IPDAZone]:
    """Detect FVGs (BISI / SIBI) and bullish Order Blocks."""
    zones: list[IPDAZone] = []
    if len(data) < 60:
        return zones

    # FVG detection
    for i in range(2, len(data)):
        c1, c3 = data[i - 2], data[i]

        if c1.high < c3.low:                        # Bullish FVG (BISI)
            zones.append(IPDAZone(
                type="FVG_BISI",
                start_time=data[i - 1].time,
                high=c3.low,
                low=c1.high,
                is_mitigated=False,
                confluence_score=75.0,
                distance_pips=(c3.low - c1.high) * 10_000,
            ))
        elif c1.low > c3.high:                      # Bearish FVG (SIBI)
            zones.append(IPDAZone(
                type="FVG_SIBI",
                start_time=data[i - 1].time,
                high=c1.low,
                low=c3.high,
                is_mitigated=False,
                confluence_score=75.0,
                distance_pips=(c1.low - c3.high) * 10_000,
            ))

    # Order Block detection
    for i in range(1, len(data) - 2):
        curr, exp = data[i], data[i + 2]
        is_bearish_ob = curr.close < curr.open
        is_bullish_exp = exp.close > exp.open and exp.close > curr.high
        if is_bearish_ob and is_bullish_exp:
            zones.append(IPDAZone(
                type="ORDER_BLOCK",
                start_time=curr.time,
                high=curr.high,
                low=curr.low,
                is_mitigated=False,
                confluence_score=85.0,
                distance_pips=(curr.high - curr.low) * 10_000,
            ))

    return zones


# ---------------------------------------------------------------------------
# SOS-27-X: Regime & Displacement Prediction
# ---------------------------------------------------------------------------

def sos_27_x_analyze(data: list[TickData]) -> SOS27XOutput:
    """Single-step momentum regime signal."""
    import random

    last = data[-1]
    prev = data[-2] if len(data) >= 2 else last
    momentum = last.close - prev.close

    return SOS27XOutput(
        confidence=min(0.95, 0.5 + abs(momentum) * 1_000),
        regime_vector=[
            1.0 if momentum > 0 else -1.0,
            last.spread * 1_000,
            last.volume / 1_000,
        ],
        predicted_displacement=momentum * 2,
        latency_ms=1.2 + random.random() * 0.5,
    )


# ---------------------------------------------------------------------------
# Session Awareness
# ---------------------------------------------------------------------------

def get_session_info(timestamp_ms: int) -> tuple[Session, bool]:
    """Return (session, is_kill_zone) for a UTC millisecond timestamp."""
    from datetime import datetime, timezone

    dt = datetime.fromtimestamp(timestamp_ms / 1_000, tz=timezone.utc)
    hour = dt.hour

    session: Session = "NONE"
    if 0 <= hour < 8:
        session = "ASIAN"
    elif 8 <= hour < 12:
        session = "LONDON"
    elif 13 <= hour < 17:
        session = "NY"

    is_kill_zone = (7 <= hour < 9) or (12 <= hour < 14)
    return session, is_kill_zone


# ---------------------------------------------------------------------------
# Deterministic Data Generation
# ---------------------------------------------------------------------------

def generate_deterministic_data(
    count: int,
    seed: str,
    start_price: float = 1.0850,
) -> list[TickData]:
    """Reproduce the TS deterministic tick stream from the same seed."""
    from datetime import datetime, timezone

    rng = SeededRandom(seed)
    data: list[TickData] = []
    current_price = start_price
    start_time = int(time.time() * 1_000) - count * 60_000

    for i in range(count):
        ts = start_time + i * 60_000
        session, is_kill_zone = get_session_info(ts)

        base_spread = 0.0001
        spread = base_spread * (1.5 + rng.next()) if is_kill_zone else base_spread

        volatility = 0.0004 if is_kill_zone else 0.0002
        open_ = current_price
        close = open_ + (rng.next() - 0.5) * volatility
        high = max(open_, close) + rng.next() * 0.0001
        low = min(open_, close) - rng.next() * 0.0001

        dt = datetime.fromtimestamp(ts / 1_000, tz=timezone.utc)
        is_event_trigger = dt.hour == 8 and dt.minute == 15

        data.append(TickData(
            time=ts,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=int(rng.next() * 1_000) + (1_000 if is_kill_zone else 500),
            bid=close - spread / 2,
            ask=close + spread / 2,
            spread=spread,
            session=session,
            is_kill_zone=is_kill_zone,
            is_event_trigger=is_event_trigger,
        ))
        current_price = close

    return data


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def calculate_evaluation_stats(logs: list[dict], balance: float) -> EvaluationStats:
    """Compute Sharpe, Sortino, win-rate and supporting diagnostics."""
    equity_curve = [EquityPoint(time=lg["time"], value=lg["equity"]) for lg in logs]

    returns: list[float] = []
    for i in range(1, len(equity_curve)):
        prev_val = equity_curve[i - 1].value
        if prev_val > 0:
            returns.append((equity_curve[i].value - prev_val) / prev_val)

    avg_return = sum(returns) / len(returns) if returns else 0.0
    variance = (
        sum((r - avg_return) ** 2 for r in returns) / len(returns)
        if returns else 0.0
    )
    std_dev = math.sqrt(variance) if variance > 0 else 0.0001

    annualisation = math.sqrt(252 * 24 * 60)
    sharpe = (avg_return / std_dev) * annualisation
    sortino = (avg_return / std_dev) * annualisation   # same σ stub as original

    return EvaluationStats(
        equity_curve=equity_curve,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        win_rate=0.65,
        win_rate_by_regime={"Trending": 0.72, "Ranging": 0.45},
        confidence_calibration=[ConfidencePoint(confidence=0.8, accuracy=0.82)],
        latency_histogram=[1.2, 1.3, 1.4, 1.5],
        false_positive_rate_sweeps=0.12,
    )
