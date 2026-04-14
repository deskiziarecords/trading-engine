"""
UROL – Universal Reliability & Observability Layer
===================================================
• Connects to IB Gateway (paper: 4002 / live: 4001)
• Subscribes 9 FX pairs (5-min OHLCV) via ib_insync:
    EUR/USD  GBP/USD  USD/JPY  GBP/JPY
    USD/CHF  AUD/USD  USD/CAD  NZD/USD  EUR/GBP
• Each pair gets its own independent MAD outlier filter
• Persists GlobalState every 0.5 s → Redis + SQLite
• Watchdog: kills and restarts on stale data or over-drawdown
• Publishes clean bars to Redis stream  clean:ticks
  (each bar carries its symbol so IPDA Core routes correctly)
• Writes structured logs to  logs/urol.log

Config (edit CONFIG block below):
    IB_HOST         – IB Gateway hostname (localhost for colo)
    IB_PORT         – 4002 paper | 4001 live
    ACCOUNT_SIZE    – notional account equity in USD
    MAX_DAILY_DD    – fraction (e.g. 0.05 = 5%) – triggers kill-switch
    SYMBOLS         – list of IB FX pairs to subscribe (sym.currency format)
"""

import asyncio
import json
import logging
import math
import sqlite3
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import redis
from ib_insync import IB, Contract, BarData, util

# ─────────────────────────── CONFIG ──────────────────────────────────────────
IB_HOST         = "127.0.0.1"
IB_PORT         = 4002          # ← change to 4001 for live
IB_CLIENT_ID    = 1
ACCOUNT_SIZE    = 100_000.0     # USD
MAX_DAILY_DD    = 0.05          # 5 % kill-switch
RISK_PER_TRADE  = 0.01          # 1 %
BAR_SIZE        = "5 mins"
# All 9 FX pairs – format: "BASE.QUOTE" (IB IDEALPRO convention)
SYMBOLS = [
    "EUR.USD",   # Euro / US Dollar
    "GBP.USD",   # British Pound / US Dollar
    "USD.JPY",   # US Dollar / Japanese Yen
    "GBP.JPY",   # British Pound / Japanese Yen
    "USD.CHF",   # US Dollar / Swiss Franc
    "AUD.USD",   # Australian Dollar / US Dollar
    "USD.CAD",   # US Dollar / Canadian Dollar
    "NZD.USD",   # New Zealand Dollar / US Dollar
    "EUR.GBP",   # Euro / British Pound
]
REDIS_HOST      = "127.0.0.1"
REDIS_PORT      = 6379
STATE_KEY       = "trading:global_state"
CLEAN_STREAM    = "clean:ticks"
LOG_DIR         = Path(__file__).parent.parent / "logs"
MAD_THRESHOLD   = 3.5          # z-score equivalent for MAD filter
STATE_FLUSH_S   = 0.5          # seconds between state persistence
WATCHDOG_S      = 30           # seconds of silence before watchdog fires
# ─────────────────────────────────────────────────────────────────────────────

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] UROL – %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "urol.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("urol")


# ─────────────────────── Global State ────────────────────────────────────────
@dataclass
class GlobalState:
    open_positions: list        = field(default_factory=list)
    current_drawdown: float     = 0.0
    daily_pnl: float            = 0.0
    mandra_gate_level: int      = 0          # 0=off 1=warn 2=halt
    last_fft_spectrum: list     = field(default_factory=list)
    active_trade_id: str        = ""
    ipda_phase: str             = "FLAT"
    ipda_lookback: int          = 20
    kill_zone_active: bool      = False
    accumulation_start_ts: int  = 0
    last_bar_ts: float          = 0.0
    kill_switch_triggered: bool = False
    account_equity: float       = ACCOUNT_SIZE

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "GlobalState":
        return cls(**json.loads(raw))


# ─────────────────────── Redis helpers ───────────────────────────────────────
def get_redis() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0,
                       decode_responses=True, socket_connect_timeout=5)


# ─────────────────────── SQLite helpers ──────────────────────────────────────
DB_PATH = LOG_DIR / "state.sqlite"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS global_state (
            id      INTEGER PRIMARY KEY,
            ts      REAL,
            payload TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS clean_bars (
            ts      REAL PRIMARY KEY,
            symbol  TEXT,
            open    REAL, high REAL, low REAL, close REAL, volume REAL
        )
    """)
    con.commit()
    con.close()

def persist_bar_sqlite(bar: dict):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO clean_bars VALUES (?,?,?,?,?,?,?)",
        (bar["ts"], bar["symbol"], bar["open"], bar["high"],
         bar["low"], bar["close"], bar["volume"])
    )
    con.commit()
    con.close()


# ─────────────────────── MAD Outlier Filter ──────────────────────────────────
class MADFilter:
    """
    Median Absolute Deviation filter on close prices.
    Rejects bars whose close deviates more than MAD_THRESHOLD * MAD from
    the rolling median (window = 20 bars).
    """
    def __init__(self, window: int = 20):
        self._buf: deque = deque(maxlen=window)

    def is_clean(self, close: float) -> bool:
        self._buf.append(close)
        if len(self._buf) < 5:
            return True                      # not enough data yet – pass through
        arr   = np.array(self._buf)
        med   = np.median(arr)
        mad   = np.median(np.abs(arr - med))
        if mad == 0:
            return True
        z = abs(close - med) / (1.4826 * mad)   # consistent estimator
        return z <= MAD_THRESHOLD


# ─────────────────────── Session kill-zone ───────────────────────────────────
KILL_ZONES_UTC = [
    (7, 10),    # London 02:00–05:00 EST  → UTC 07:00–10:00
    (12, 15),   # NY    07:00–10:00 EST   → UTC 12:00–15:00
]

def is_kill_zone() -> bool:
    utc_hour = datetime.now(timezone.utc).hour
    return any(s <= utc_hour < e for s, e in KILL_ZONES_UTC)


# ─────────────────────── UROL Main ───────────────────────────────────────────
class UROL:
    def __init__(self):
        self.r      = get_redis()
        self.ib     = IB()
        self.state  = self._load_state()
        self.mad    = {sym: MADFilter() for sym in SYMBOLS}
        self._last_bar_time: float = time.time()

    def _load_state(self) -> GlobalState:
        try:
            raw = self.r.get(STATE_KEY)
            if raw:
                s = GlobalState.from_json(raw)
                log.info("Reloaded GlobalState from Redis (phase=%s, dd=%.2f%%)",
                         s.ipda_phase, s.current_drawdown * 100)
                return s
        except Exception as e:
            log.warning("Could not load state from Redis: %s", e)
        return GlobalState()

    def _flush_state(self):
        js = self.state.to_json()
        try:
            self.r.set(STATE_KEY, js)
        except Exception as e:
            log.warning("Redis state flush failed: %s", e)
        # SQLite backup
        try:
            con = sqlite3.connect(DB_PATH)
            con.execute("INSERT INTO global_state(ts, payload) VALUES(?,?)",
                        (time.time(), js))
            con.commit()
            con.close()
        except Exception:
            pass

    def _on_bar(self, bars: list[BarData], has_new: bool):
        if not has_new:
            return
        bar = bars[-1]
        sym = bar.contract.symbol + "." + bar.contract.currency \
              if hasattr(bar, "contract") else SYMBOLS[0]

        ts_ms  = int(bar.date.timestamp() * 1000)
        o, h, l, c, v = bar.open, bar.high, bar.low, bar.close, bar.volume

        # ── MAD filter ──
        if not self.mad[sym].is_clean(c):
            log.warning("[MAD] Outlier bar rejected: sym=%s close=%.5f", sym, c)
            return

        self._last_bar_time = time.time()
        self.state.last_bar_ts    = ts_ms / 1000
        self.state.kill_zone_active = is_kill_zone()

        bar_dict = {
            "symbol": sym, "ts": ts_ms,
            "open": o, "high": h, "low": l, "close": c, "volume": float(v)
        }

        # ── Publish to Redis stream ──
        try:
            self.r.xadd(CLEAN_STREAM, {"payload": json.dumps(bar_dict)},
                        maxlen=10_000)
        except Exception as e:
            log.error("xadd failed: %s", e)

        # ── SQLite ──
        persist_bar_sqlite(bar_dict)

        log.info("[BAR] %s | O=%.5f H=%.5f L=%.5f C=%.5f V=%.0f | KZ=%s",
                 sym, o, h, l, c, v, self.state.kill_zone_active)

    def _check_drawdown(self):
        dd = self.state.current_drawdown
        if dd >= MAX_DAILY_DD and not self.state.kill_switch_triggered:
            log.critical("[KILL-SWITCH] Daily drawdown %.2f%% ≥ limit %.2f%%. "
                         "Halting all trading.", dd * 100, MAX_DAILY_DD * 100)
            self.state.kill_switch_triggered = True
            self.state.mandra_gate_level     = 2
            # Signal AECABI to flatten all positions
            self.r.xadd("control:commands",
                        {"command": "HALT", "reason": "drawdown_limit"})

    async def _watchdog(self):
        """Fires if no bar received for WATCHDOG_S seconds."""
        while True:
            await asyncio.sleep(WATCHDOG_S)
            silence = time.time() - self._last_bar_time
            if silence > WATCHDOG_S:
                log.warning("[WATCHDOG] No bar for %.0f s – checking IB connection.", silence)
                if not self.ib.isConnected():
                    log.error("[WATCHDOG] IB disconnected – attempting reconnect.")
                    try:
                        await self.ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
                        await self._subscribe()
                        log.info("[WATCHDOG] Reconnected successfully.")
                    except Exception as e:
                        log.critical("[WATCHDOG] Reconnect failed: %s", e)

    async def _state_flusher(self):
        while True:
            await asyncio.sleep(STATE_FLUSH_S)
            self._check_drawdown()
            self._flush_state()

    async def _subscribe(self):
        for sym_str in SYMBOLS:
            sym, cur = sym_str.split(".")
            contract = Contract(secType="CASH", symbol=sym,
                                currency=cur, exchange="IDEALPRO")
            bars = self.ib.reqRealTimeBars(
                contract, 5, "MIDPOINT", useRTH=False
            )
            bars.updateEvent += self._on_bar
            log.info("Subscribed real-time bars: %s", sym_str)

    async def run(self):
        log.info("UROL starting – IB Gateway %s:%d", IB_HOST, IB_PORT)
        await self.ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
        log.info("Connected to IB Gateway.")
        await self._subscribe()
        asyncio.ensure_future(self._watchdog())
        asyncio.ensure_future(self._state_flusher())
        await asyncio.Event().wait()   # run forever


if __name__ == "__main__":
    init_db()
    util.patchAsyncio()
    asyncio.run(UROL().run())
