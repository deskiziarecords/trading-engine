"""
IPDA Core – Multi-Pair Signal Generator
========================================
Supported pairs (all routed independently):
    EUR/USD  GBP/USD  USD/JPY  GBP/JPY
    USD/CHF  AUD/USD  USD/CAD  NZD/USD  EUR/GBP

Per-pair handling:
  • Independent rolling OHLCV buffers (20/40/60-day look-backs)
  • Correct pip size per pair (JPY pairs: 0.01, all others: 0.0001)
  • Correct pip value per standard lot in USD
  • Independent IPDA phase tracking
  • Each signal carries its symbol – AECABI routes accordingly

Pipeline:
  UROL -> clean:ticks (Redis stream, each bar tagged with symbol)
       -> IPDA Core (per-pair phase + size)
       -> jax:signals (Redis stream, consumed by AECABI)
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import redis

# ─────────────────────────── CONFIG ──────────────────────────────────────────
REDIS_HOST       = "127.0.0.1"
REDIS_PORT       = 6379
CLEAN_STREAM     = "clean:ticks"
SIGNAL_STREAM    = "jax:signals"
STATE_KEY        = "trading:global_state"
CONSUMER_GROUP   = "ipda-core"
CONSUMER_NAME    = "ipda-worker-1"

ACCOUNT_SIZE     = 100_000.0    # USD
RISK_PER_TRADE   = 0.01         # 1% per trade per pair

# 5-min bars: 24h x 12 bars/h = 288 bars/day
BARS_PER_DAY     = 288
LOOKBACK_DAYS    = [20, 40, 60]
LOOKBACK_BARS    = [d * BARS_PER_DAY for d in LOOKBACK_DAYS]

# Phase detection thresholds
MANIP_VOL_MULT   = 1.5
MANIP_MOVE_MULT  = 2.0
ATR_PERIOD       = 20

# Kill-zones (UTC hours, inclusive start, exclusive end)
KILL_ZONES_UTC = [
    (0,  3),    # Tokyo    00:00-03:00 UTC
    (7,  10),   # London   07:00-10:00 UTC
    (12, 15),   # NY       12:00-15:00 UTC
]

# ── Per-pair FX metadata ──────────────────────────────────────────────────────
# pip_size      : minimum price move (0.01 for JPY pairs, 0.0001 for others)
# pip_value_usd : USD value of 1 pip per 1 standard lot (100,000 base units)
#   USD-quote pairs (EUR/USD, GBP/USD etc.): pip_value = 0.0001 * 100,000 = $10
#   USD-base pairs (USD/JPY, USD/CHF, USD/CAD): pip_value = pip_size * 100,000 / rate
#     Using approximate mid-market rates for fixed values below.
#     Update pip_value_usd periodically as rates shift significantly.
FX_META = {
    "EUR.USD": {"pip_size": 0.0001, "pip_value_usd": 10.00},
    "GBP.USD": {"pip_size": 0.0001, "pip_value_usd": 10.00},
    "USD.JPY": {"pip_size": 0.01,   "pip_value_usd":  9.10},  # ~$9.10 @ 110 USDJPY
    "GBP.JPY": {"pip_size": 0.01,   "pip_value_usd":  9.10},  # cross – approx
    "USD.CHF": {"pip_size": 0.0001, "pip_value_usd":  9.80},  # ~$9.80 @ 1.02 USDCHF
    "AUD.USD": {"pip_size": 0.0001, "pip_value_usd": 10.00},
    "USD.CAD": {"pip_size": 0.0001, "pip_value_usd":  7.50},  # ~$7.50 @ 1.33 USDCAD
    "NZD.USD": {"pip_size": 0.0001, "pip_value_usd": 10.00},
    "EUR.GBP": {"pip_size": 0.0001, "pip_value_usd": 12.50},  # ~$12.50 @ 0.80 EURGBP
}

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] IPDA – %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ipda.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ipda")


# ─────────────────────── Kill-zone helpers ────────────────────────────────────

def is_kill_zone() -> bool:
    h = datetime.now(timezone.utc).hour
    return any(s <= h < e for s, e in KILL_ZONES_UTC)


def active_session() -> str:
    h = datetime.now(timezone.utc).hour
    if  0 <= h <  3: return "TOKYO"
    if  7 <= h < 10: return "LONDON"
    if 12 <= h < 15: return "NY"
    return "OFF"


# ─────────────────────── ATR ─────────────────────────────────────────────────

def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = ATR_PERIOD) -> float:
    """Wilder's ATR – returns most-recent value."""
    if len(high) < 2:
        return 0.0
    tr = np.maximum(
        np.maximum(high[1:] - low[1:],
                   np.abs(high[1:] - close[:-1])),
        np.abs(low[1:] - close[:-1])
    )
    atr = np.zeros(len(tr))
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return float(atr[-1])


# ─────────────────────── Position sizing ─────────────────────────────────────

def position_size(symbol: str, atr: float) -> float:
    """
    Risk-scaled standard lots:
        size = (equity x risk%) / (ATR_pips x pip_value_per_lot)
    """
    meta          = FX_META.get(symbol, {"pip_size": 0.0001, "pip_value_usd": 10.0})
    pip_size      = meta["pip_size"]
    pip_value_usd = meta["pip_value_usd"]

    if atr <= 0 or pip_size <= 0:
        return 0.0

    atr_pips  = atr / pip_size
    size_lots = (ACCOUNT_SIZE * RISK_PER_TRADE) / (atr_pips * pip_value_usd)
    return round(max(size_lots, 0.01), 2)   # floor at 0.01 (micro lot)


# ─────────────────────── Phase detection ─────────────────────────────────────

def detect_phase(df: pd.DataFrame) -> str:
    """
    IPDA phase on the 20-day window.
    Priority: MANIPULATION > ACCUMULATION > DISTRIBUTION > FLAT
    """
    lb = LOOKBACK_BARS[0]
    if len(df) < lb:
        return "FLAT"

    d = df.iloc[-lb:].copy()

    vol_mean_5  = max(d["volume"].iloc[-5:].mean(), 1.0)
    vol_mean_20 = max(d["volume"].mean(), 1.0)
    vol_last    = d["volume"].iloc[-1]

    returns   = d["close"].pct_change().dropna()
    ret_vol_5 = returns.rolling(5).std().iloc[-1] or 1e-9

    close_last = d["close"].iloc[-1]
    close_prev = d["close"].iloc[-2]
    price_move = abs(close_last - close_prev) / (abs(close_prev) or 1e-9)

    low_last   = d["low"].iloc[-1]
    low_prev   = d["low"].iloc[-2]
    high_last  = d["high"].iloc[-1]
    high_prev  = d["high"].iloc[-2]

    # MANIPULATION – sharp move + volume spike
    if (price_move > MANIP_MOVE_MULT * ret_vol_5 and
            vol_last > MANIP_VOL_MULT * vol_mean_5):
        return "MANIPULATION"

    # ACCUMULATION – higher low + contracting volume
    if low_last > low_prev and vol_last < vol_mean_20:
        return "ACCUMULATION"

    # DISTRIBUTION – lower high + expanding volume
    if high_last < high_prev and vol_last > vol_mean_20:
        return "DISTRIBUTION"

    return "FLAT"


def confirm_with_lookbacks(df: pd.DataFrame, primary_phase: str) -> bool:
    """
    40-day and 60-day windows must agree with the primary phase (or be FLAT).
    Returns True = confirmed, False = blocked.
    """
    if primary_phase == "FLAT":
        return False
    for lb in LOOKBACK_BARS[1:]:
        if len(df) < lb:
            continue   # insufficient history – neutral, don't block
        phase = detect_phase(df.iloc[-lb:])
        if phase not in (primary_phase, "FLAT"):
            return False
    return True


# ─────────────────────── Signal builder ──────────────────────────────────────

def build_signal(symbol: str, phase: str, df: pd.DataFrame,
                 kz: bool, session: str) -> dict:
    base = {
        "symbol":    symbol,
        "action":    "FLAT",
        "size":      0.0,
        "phase":     phase,
        "kill_zone": kz,
        "session":   session,
        "timestamp": time.time(),
        "price":     float(df["close"].iloc[-1]) if len(df) else 0.0,
        "atr":       0.0,
    }

    if not kz or phase in ("FLAT", "MANIPULATION"):
        return base

    atr    = compute_atr(df["high"].values, df["low"].values, df["close"].values)
    size   = position_size(symbol, atr)
    action = "BUY" if phase == "ACCUMULATION" else "SELL"

    base.update({
        "action":    action,
        "size":      size,
        "atr":       round(atr, 6),
        "lookback":  LOOKBACK_DAYS[0],
    })
    return base


# ─────────────────────── Per-pair state ──────────────────────────────────────

@dataclass
class PairState:
    symbol: str
    buf:    deque = field(default_factory=lambda: deque(maxlen=max(LOOKBACK_BARS)))
    phase:  str   = "FLAT"

    def append(self, bar: dict):
        self.buf.append(bar)

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(list(self.buf))
        if df.empty:
            return df
        cols = ["open", "high", "low", "close", "volume"]
        df[cols] = df[cols].astype(float)
        return df


# ─────────────────────── IPDA Core ───────────────────────────────────────────

class IPDACore:
    def __init__(self):
        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0,
                             decode_responses=True, socket_connect_timeout=5)
        self.pairs: dict[str, PairState] = {
            sym: PairState(symbol=sym) for sym in FX_META
        }
        self._ensure_consumer_group()
        self._load_state()

    def _ensure_consumer_group(self):
        try:
            self.r.xgroup_create(CLEAN_STREAM, CONSUMER_GROUP, id="0",
                                 mkstream=True)
            log.info("Consumer group '%s' created.", CONSUMER_GROUP)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def _load_state(self):
        try:
            raw = self.r.get(STATE_KEY)
            if raw:
                s = json.loads(raw)
                log.info("State reloaded: phase=%s dd=%.2f%%",
                         s.get("ipda_phase", "FLAT"),
                         s.get("current_drawdown", 0.0) * 100)
        except Exception as e:
            log.warning("State reload failed: %s", e)

    def _update_state(self, phases: dict, kz: bool):
        try:
            raw   = self.r.get(STATE_KEY)
            state = json.loads(raw) if raw else {}
            state["ipda_phase"]       = json.dumps(phases)
            state["kill_zone_active"] = kz
            self.r.set(STATE_KEY, json.dumps(state))
        except Exception as e:
            log.warning("State update failed: %s", e)

    def _is_halted(self) -> bool:
        try:
            raw = self.r.get(STATE_KEY)
            if raw:
                return json.loads(raw).get("kill_switch_triggered", False)
        except Exception:
            pass
        return False

    def _read_bars(self) -> list[dict]:
        try:
            entries = self.r.xreadgroup(
                CONSUMER_GROUP, CONSUMER_NAME,
                {CLEAN_STREAM: ">"},
                count=100, block=5000
            )
        except redis.exceptions.ConnectionError as e:
            log.error("Redis read error: %s", e)
            return []
        if not entries:
            return []
        bars = []
        for stream, messages in entries:
            for msg_id, fields in messages:
                try:
                    bar = json.loads(fields["payload"])
                    bars.append(bar)
                    self.r.xack(CLEAN_STREAM, CONSUMER_GROUP, msg_id)
                except Exception as e:
                    log.warning("Malformed bar: %s", e)
        return bars

    async def run(self):
        log.info("IPDA Core starting – %d pairs | lookbacks=%s days | %d bars/day",
                 len(self.pairs), LOOKBACK_DAYS, BARS_PER_DAY)
        loop = asyncio.get_event_loop()

        while True:
            if self._is_halted():
                log.warning("[HALTED] Kill-switch active. Pausing 60 s...")
                await asyncio.sleep(60)
                continue

            bars = await loop.run_in_executor(None, self._read_bars)
            if not bars:
                await asyncio.sleep(1)
                continue

            # Route each bar to the correct pair buffer
            updated: set[str] = set()
            for bar in bars:
                sym = bar.get("symbol", "")
                if sym not in self.pairs:
                    continue
                self.pairs[sym].append(bar)
                updated.add(sym)

            kz      = is_kill_zone()
            session = active_session()
            phases  = {}

            for sym in updated:
                ps    = self.pairs[sym]
                df    = ps.to_df()
                phase = detect_phase(df)

                if not confirm_with_lookbacks(df, phase) and phase != "FLAT":
                    log.debug("[%s] %s not confirmed by 40/60-day → FLAT", sym, phase)
                    phase = "FLAT"

                ps.phase    = phase
                phases[sym] = phase

                signal = build_signal(sym, phase, df, kz, session)

                try:
                    self.r.xadd(SIGNAL_STREAM,
                                {"payload": json.dumps(signal)},
                                maxlen=50_000)
                except Exception as e:
                    log.error("[%s] Publish failed: %s", sym, e)
                    continue

                if signal["action"] != "FLAT":
                    log.info(
                        "[SIGNAL] %-8s %-4s size=%-6.2f phase=%-14s "
                        "session=%-6s kz=%s price=%.5f atr=%.6f",
                        sym, signal["action"], signal["size"], phase,
                        session, kz, signal["price"], signal["atr"]
                    )

            if phases:
                self._update_state(phases, kz)

            await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(IPDACore().run())
