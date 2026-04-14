"""
AECABI – Adaptive Execution & Cost-Aware Broker Interface (Multi-Pair)
=======================================================================
Handles 9 FX pairs routed from IPDA Core:
    EUR/USD  GBP/USD  USD/JPY  GBP/JPY
    USD/CHF  AUD/USD  USD/CAD  NZD/USD  EUR/GBP

Per-pair handling:
  • Correct IB contract (CASH, IDEALPRO)
  • Correct pip size for TCA calculation (JPY: 0.01, others: 0.0001)
  • Correct pip value per lot for cost estimation
  • Independent deduplication (SHA-256 trade_id per symbol+phase+window)

Pipeline:
  jax:signals (Redis) -> TCA filter -> IB IOC order + shadow fill -> SQLite
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import redis
from ib_insync import IB, Contract, MarketOrder, util

# ─────────────────────────── CONFIG ──────────────────────────────────────────
REDIS_HOST       = "127.0.0.1"
REDIS_PORT       = 6379
SIGNAL_STREAM    = "jax:signals"
CONTROL_STREAM   = "control:commands"
STATE_KEY        = "trading:global_state"
CONSUMER_GROUP   = "aecabi"
CONSUMER_NAME    = "aecabi-worker-1"

IB_HOST          = "127.0.0.1"
IB_PORT          = 4002           # 4001 for live
IB_CLIENT_ID     = 2

# TCA parameters
TCA_EDGE_MULT     = 3.0           # edge must be > 3x all-in cost
COMMISSION_USD    = 2.0           # per round-turn (IB fixed, all FX pairs)
MAX_SLIPPAGE_PIPS = 2.0           # reject if fill slippage > this
ORDER_TIMEOUT_S   = 10

# Reconnection
RECONNECT_BASE_S  = 2
RECONNECT_MAX_S   = 60

# ── Per-pair FX metadata (mirrors IPDA Core) ──────────────────────────────────
FX_META = {
    "EUR.USD": {"pip_size": 0.0001, "pip_value_usd": 10.00,
                "sym": "EUR", "cur": "USD"},
    "GBP.USD": {"pip_size": 0.0001, "pip_value_usd": 10.00,
                "sym": "GBP", "cur": "USD"},
    "USD.JPY": {"pip_size": 0.01,   "pip_value_usd":  9.10,
                "sym": "USD", "cur": "JPY"},
    "GBP.JPY": {"pip_size": 0.01,   "pip_value_usd":  9.10,
                "sym": "GBP", "cur": "JPY"},
    "USD.CHF": {"pip_size": 0.0001, "pip_value_usd":  9.80,
                "sym": "USD", "cur": "CHF"},
    "AUD.USD": {"pip_size": 0.0001, "pip_value_usd": 10.00,
                "sym": "AUD", "cur": "USD"},
    "USD.CAD": {"pip_size": 0.0001, "pip_value_usd":  7.50,
                "sym": "USD", "cur": "CAD"},
    "NZD.USD": {"pip_size": 0.0001, "pip_value_usd": 10.00,
                "sym": "NZD", "cur": "USD"},
    "EUR.GBP": {"pip_size": 0.0001, "pip_value_usd": 12.50,
                "sym": "EUR", "cur": "GBP"},
}

# Typical half-spread per pair (pips) – used in TCA cost calculation
PAIR_SPREAD_PIPS = {
    "EUR.USD": 0.5,
    "GBP.USD": 0.7,
    "USD.JPY": 0.5,
    "GBP.JPY": 1.5,
    "USD.CHF": 1.0,
    "AUD.USD": 0.7,
    "USD.CAD": 1.0,
    "NZD.USD": 1.0,
    "EUR.GBP": 1.0,
}

LOG_DIR = Path(__file__).parent.parent / "logs"
DB_PATH = LOG_DIR / "shadow_trades.sqlite"
LOG_DIR.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] AECABI – %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "aecabi.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("aecabi")


# ─────────────────────── Data classes ────────────────────────────────────────

@dataclass
class Fill:
    trade_id:     str
    symbol:       str
    action:       str
    size:         float
    signal_price: float
    fill_price:   float
    slippage:     float
    slippage_pips: float
    commission:   float
    phase:        str
    session:      str
    kill_zone:    bool
    ts:           float
    live:         bool
    status:       str   # FILLED | SHADOW | REJECTED | ERROR


# ─────────────────────── SQLite ──────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS fills (
            trade_id      TEXT PRIMARY KEY,
            symbol        TEXT,
            action        TEXT,
            size          REAL,
            signal_price  REAL,
            fill_price    REAL,
            slippage      REAL,
            slippage_pips REAL,
            commission    REAL,
            phase         TEXT,
            session       TEXT,
            kill_zone     INTEGER,
            ts            REAL,
            live          INTEGER,
            status        TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS dedup_ids (
            trade_id TEXT PRIMARY KEY,
            ts       REAL
        )
    """)
    con.commit()
    con.close()


def save_fill(fill: Fill):
    con = sqlite3.connect(DB_PATH)
    d = asdict(fill)
    d["kill_zone"] = int(d["kill_zone"])
    d["live"]      = int(d["live"])
    cols         = ", ".join(d.keys())
    placeholders = ", ".join(["?"] * len(d))
    con.execute(
        f"INSERT OR REPLACE INTO fills ({cols}) VALUES ({placeholders})",
        list(d.values())
    )
    con.commit()
    con.close()


def is_duplicate(trade_id: str) -> bool:
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT 1 FROM dedup_ids WHERE trade_id=?", (trade_id,)
    ).fetchone()
    con.close()
    return row is not None


def mark_sent(trade_id: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT OR IGNORE INTO dedup_ids VALUES (?,?)",
                (trade_id, time.time()))
    con.commit()
    con.close()


# ─────────────────────── TCA Filter ──────────────────────────────────────────

def tca_passes(signal: dict) -> bool:
    """
    Per-pair TCA:
      Expected edge = 0.5 * ATR_pips * pip_value * size   (USD)
      All-in cost   = (spread_pips + commission/lot) * pip_value * size
      Pass if edge >= TCA_EDGE_MULT * cost
    """
    sym        = signal.get("symbol", "EUR.USD")
    atr_price  = signal.get("atr", 0.0)
    size_lots  = signal.get("size", 0.0)

    if atr_price <= 0 or size_lots <= 0:
        return False

    meta          = FX_META.get(sym, FX_META["EUR.USD"])
    pip_size      = meta["pip_size"]
    pip_value_usd = meta["pip_value_usd"]
    spread_pips   = PAIR_SPREAD_PIPS.get(sym, 1.0)

    atr_pips      = atr_price / pip_size
    edge_usd      = atr_pips * 0.5 * pip_value_usd * size_lots

    comm_per_lot  = COMMISSION_USD / max(size_lots, 0.01)
    cost_usd      = (spread_pips + comm_per_lot) * pip_value_usd * size_lots

    passes = edge_usd >= TCA_EDGE_MULT * cost_usd
    log.info("[TCA] %-8s edge=$%.2f cost=$%.2f mult=%.1f -> %s",
             sym, edge_usd, cost_usd, TCA_EDGE_MULT,
             "PASS" if passes else "FAIL")
    return passes


# ─────────────────────── Shadow fill ─────────────────────────────────────────

def shadow_fill(signal: dict, trade_id: str) -> Fill:
    sym       = signal.get("symbol", "EUR.USD")
    action    = signal["action"]
    price     = signal.get("price", 0.0)
    meta      = FX_META.get(sym, FX_META["EUR.USD"])
    pip_size  = meta["pip_size"]

    slip_price = 0.5 * pip_size                          # 0.5 pip slippage
    fill_price = price + slip_price if action == "BUY" else price - slip_price

    fill = Fill(
        trade_id      = trade_id,
        symbol        = sym,
        action        = action,
        size          = signal["size"],
        signal_price  = price,
        fill_price    = fill_price,
        slippage      = slip_price,
        slippage_pips = 0.5,
        commission    = COMMISSION_USD,
        phase         = signal.get("phase", ""),
        session       = signal.get("session", ""),
        kill_zone     = signal.get("kill_zone", False),
        ts            = time.time(),
        live          = False,
        status        = "SHADOW",
    )
    save_fill(fill)
    log.info("[SHADOW] %-8s %s %.2f lots @ %.5f (slip=0.5 pips)",
             sym, action, fill.size, fill_price)
    return fill


# ─────────────────────── IB contract factory ─────────────────────────────────

def make_contract(symbol: str) -> Contract:
    meta = FX_META[symbol]
    return Contract(
        secType  = "CASH",
        symbol   = meta["sym"],
        currency = meta["cur"],
        exchange = "IDEALPRO",
    )


# ─────────────────────── IB order submission ─────────────────────────────────

async def submit_order(ib: IB, signal: dict, trade_id: str) -> Optional[Fill]:
    sym    = signal.get("symbol", "EUR.USD")
    action = signal["action"]
    size   = signal["size"]
    price  = signal["price"]

    meta      = FX_META.get(sym, FX_META["EUR.USD"])
    pip_size  = meta["pip_size"]

    contract       = make_contract(sym)
    order          = MarketOrder(action, size)
    order.tif      = "IOC"
    order.orderRef = trade_id[:40]

    log.info("[ORDER] %-8s %s %.2f lots | id=%s", sym, action, size, trade_id)
    trade = ib.placeOrder(contract, order)

    deadline = time.time() + ORDER_TIMEOUT_S
    while time.time() < deadline:
        await asyncio.sleep(0.1)
        if trade.orderStatus.status in ("Filled", "Cancelled", "Inactive"):
            break

    if trade.orderStatus.status != "Filled":
        log.warning("[ORDER] %-8s not filled – status=%s",
                    sym, trade.orderStatus.status)
        return None

    avg_fill      = trade.orderStatus.avgFillPrice
    slippage      = abs(avg_fill - price)
    slip_pips     = slippage / pip_size

    if slip_pips > MAX_SLIPPAGE_PIPS:
        log.warning("[ORDER] %-8s excessive slippage %.2f pips > %.2f limit",
                    sym, slip_pips, MAX_SLIPPAGE_PIPS)

    fill = Fill(
        trade_id      = trade_id,
        symbol        = sym,
        action        = action,
        size          = size,
        signal_price  = price,
        fill_price    = avg_fill,
        slippage      = slippage,
        slippage_pips = slip_pips,
        commission    = COMMISSION_USD,
        phase         = signal.get("phase", ""),
        session       = signal.get("session", ""),
        kill_zone     = signal.get("kill_zone", False),
        ts            = time.time(),
        live          = True,
        status        = "FILLED",
    )
    save_fill(fill)
    log.info("[FILL] %-8s %s %.2f @ %.5f (slip=%.2f pips)",
             sym, action, size, avg_fill, slip_pips)
    return fill


# ─────────────────────── AECABI main ─────────────────────────────────────────

class AECABI:
    def __init__(self):
        self.r   = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0,
                               decode_responses=True, socket_connect_timeout=5)
        self.ib  = IB()
        self._ensure_consumer_group()
        self._reconnect_delay = RECONNECT_BASE_S

    def _ensure_consumer_group(self):
        try:
            self.r.xgroup_create(SIGNAL_STREAM, CONSUMER_GROUP, id="0",
                                 mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def _connect_ib(self):
        delay = self._reconnect_delay
        while True:
            try:
                await self.ib.connectAsync(IB_HOST, IB_PORT,
                                           clientId=IB_CLIENT_ID)
                log.info("Connected to IB Gateway.")
                self._reconnect_delay = RECONNECT_BASE_S
                return
            except Exception as e:
                log.error("IB connect failed: %s – retry in %ds", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_S)

    def _is_halted(self) -> bool:
        try:
            raw = self.r.get(STATE_KEY)
            if raw:
                return json.loads(raw).get("kill_switch_triggered", False)
        except Exception:
            pass
        return False

    async def _flatten_all(self):
        log.critical("[FLATTEN] Closing all open positions.")
        try:
            for pos in self.ib.positions():
                if pos.position == 0:
                    continue
                action = "SELL" if pos.position > 0 else "BUY"
                self.ib.placeOrder(pos.contract,
                                   MarketOrder(action, abs(pos.position)))
                log.critical("[FLATTEN] %s %.0f %s",
                             action, abs(pos.position), pos.contract.symbol)
        except Exception as e:
            log.error("Flatten error: %s", e)

    async def _watch_control(self):
        while True:
            await asyncio.sleep(1)
            try:
                msgs = self.r.xread({CONTROL_STREAM: "$"}, count=10, block=1000)
                for _, messages in (msgs or []):
                    for _, fields in messages:
                        if fields.get("command") == "HALT":
                            log.critical("[CONTROL] HALT: %s",
                                         fields.get("reason", ""))
                            await self._flatten_all()
            except Exception as e:
                log.warning("Control stream error: %s", e)

    async def _read_signals(self) -> list[dict]:
        loop = asyncio.get_event_loop()

        def _read():
            try:
                return self.r.xreadgroup(
                    CONSUMER_GROUP, CONSUMER_NAME,
                    {SIGNAL_STREAM: ">"},
                    count=50, block=5000
                )
            except Exception as e:
                log.error("Redis read: %s", e)
                return []

        entries = await loop.run_in_executor(None, _read)
        signals = []
        for stream, messages in (entries or []):
            for msg_id, fields in messages:
                try:
                    sig = json.loads(fields["payload"])
                    sig["_msg_id"] = msg_id
                    signals.append(sig)
                    self.r.xack(SIGNAL_STREAM, CONSUMER_GROUP, msg_id)
                except Exception as e:
                    log.warning("Malformed signal: %s", e)
        return signals

    async def on_signal(self, signal: dict):
        action = signal.get("action", "FLAT")
        sym    = signal.get("symbol", "")

        if action == "FLAT":
            return

        if sym not in FX_META:
            log.warning("Unknown symbol in signal: %s – skipped.", sym)
            return

        # Stable dedup key: symbol + phase + 5-min window
        dedup_str = f"{sym}|{action}|{signal['phase']}|{int(signal['timestamp'] // 300)}"
        trade_id  = hashlib.sha256(dedup_str.encode()).hexdigest()[:24]

        if is_duplicate(trade_id):
            log.info("[DEDUP] %s trade_id=%s already processed.", sym, trade_id)
            return
        mark_sent(trade_id)

        if not tca_passes(signal):
            log.info("[TCA FAIL] %s signal filtered.", sym)
            shadow_fill(signal, trade_id + "_tca_shadow")
            return

        # Always shadow-fill
        shadow_fill(signal, trade_id + "_shadow")

        # Live order (only if IB connected and not halted)
        if not self.ib.isConnected():
            log.warning("[LIVE] IB not connected – shadow only for %s.", sym)
            return

        await submit_order(self.ib, signal, trade_id)

    async def run(self):
        init_db()
        await self._connect_ib()
        asyncio.ensure_future(self._watch_control())
        log.info("AECABI running – %d pairs | consuming '%s'",
                 len(FX_META), SIGNAL_STREAM)

        while True:
            if self._is_halted():
                log.warning("[HALTED] Kill-switch active – skipping execution.")
                await asyncio.sleep(60)
                continue

            signals = await self._read_signals()
            for sig in signals:
                try:
                    await self.on_signal(sig)
                except Exception as e:
                    log.error("on_signal error (%s): %s",
                              sig.get("symbol", "?"), e)

            await asyncio.sleep(0)


if __name__ == "__main__":
    util.patchAsyncio()
    asyncio.run(AECABI().run())
