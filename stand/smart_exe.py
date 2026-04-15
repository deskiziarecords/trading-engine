#!/usr/bin/env python3
"""
SMART-EXE Trading Bot - ACTIVE VERSION
Relaxed thresholds for demonstration
"""

import numpy as np
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from typing import Optional
import random

SYMBOL_VALUES = {'B': 900, 'I': -900, 'W': 500, 'w': -500, 'U': 330, 'D': -320, 'X': 100}
PATTERN_STOPS = {'B': 0.008, 'I': 0.008, 'W': 0.006, 'w': 0.006, 'U': 0.010, 'D': 0.010, 'X': 0.005}

class PatternEncoder:
    def __init__(self): self.sequence = deque(maxlen=20)
    def encode_candle(self, o, h, l, c):
        body, range_val = abs(c-o), h-l
        if range_val == 0: return 'X'
        body_ratio = body / range_val
        upper_wick, lower_wick = h-max(o,c), min(o,c)-l
        if upper_wick > range_val * 0.6: return 'W'
        if lower_wick > range_val * 0.6: return 'w'
        if body_ratio < 0.1: return 'X'
        return ('B' if body_ratio > 0.6 else 'U') if c > o else ('I' if body_ratio > 0.6 else 'D')
    def add_candle(self, o, h, l, c): 
        self.sequence.append(self.encode_candle(o, h, l, c))
    def get_sequence_list(self): return list(self.sequence)

class PatternEvaluator:
    def evaluate_sequence(self, seq):
        if not seq: return {'total': 0, 'direction': 'neutral'}
        total = sum(SYMBOL_VALUES[s] * (i+1)/len(seq) for i,s in enumerate(seq))
        return {'total': total, 'direction': 'bullish' if total > 0 else 'bearish' if total < 0 else 'neutral'}
    def predict_next_symbol(self, seq):
        if len(seq) < 5: return 'X', 0, 0
        base = self.evaluate_sequence(seq)['total']
        best, best_delta = 'X', 0
        for s in ['B','I','W','w','U','D','X']:
            delta = abs(self.evaluate_sequence(seq + [s])['total'] - base)
            if delta > best_delta: best, best_delta = s, delta
        return best, best_delta, min(1.0, best_delta/2000)

class EntropyFilter:
    def __init__(self, threshold=0.85): self.threshold = threshold
    def calculate_entropy(self, seq):
        if not seq: return 1.0
        counts = {}
        for s in seq: counts[s] = counts.get(s, 0) + 1
        entropy = sum(-(c/len(seq))*np.log2(c/len(seq)) for c in counts.values() if c > 0)
        return entropy / np.log2(7)

class PatternMemory:
    def __init__(self, max_size=10000, k=3):
        self.max_size, self.k, self.memory = max_size, k, []
    def _seq_to_vec(self, seq):
        vec = np.zeros(140, dtype=np.float32)
        for i,s in enumerate(seq[:20]):
            if s in 'BIWwUDX': vec[i*7 + 'BIWwUDX'.index(s)] = 1
        return vec
    def add_pattern(self, seq, outcome):
        self.memory.append((self._seq_to_vec(seq), outcome))
        if len(self.memory) > self.max_size: self.memory.pop(0)
    def query_memory_bias(self, seq):
        if len(self.memory) < self.k: return 0.0, len(self.memory)
        q = self._seq_to_vec(seq)
        dists = sorted([(np.sqrt(np.sum((q-v)**2)), o) for v,o in self.memory])[:self.k]
        return np.mean([o for _,o in dists]), self.k

class GeometricValidator:
    def __init__(self, energy_threshold=0.8, curl_threshold=0.9):
        self.energy_threshold, self.curl_threshold = energy_threshold, curl_threshold
    def calculate_energy(self, seq):
        if len(seq) < 3: return 1.0
        emb = np.array([SYMBOL_VALUES.get(s,0)/1000 for s in seq], dtype=np.float32)
        diffs = np.diff(emb)
        return min(1.0, np.sum(diffs**2) / len(seq) / 10)
    def validate(self, seq, direction):
        e = self.calculate_energy(seq)
        return e < self.energy_threshold, {'energy': e}

class RiskManager:
    def __init__(self, asset_bias='neutral', max_pos=2.0, min_pos=0.5, 
                 entropy=0.85, conf=0.45, mem=-1.0, energy=0.8):
        self.asset_bias, self.max_pos, self.min_pos = asset_bias, max_pos, min_pos
        self.entropy_threshold, self.min_confidence, self.min_memory_bias = entropy, conf, mem
        self.max_energy, self.consecutive_losses = energy, 0
    def calculate_kelly_size(self, win_rate, avg_win, avg_loss, conf, stability):
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1: return self.min_pos/100
        b, p, q = avg_win/avg_loss, win_rate, 1-win_rate
        kelly = ((b*p - q)/b) * 0.25 * conf * stability
        return max(self.min_pos/100, min(self.max_pos/100, kelly))
    def apply_lambda_gates(self, entropy, mem_bias, conf, energy, direction, size, stop):
        reasons = []
        if entropy >= self.entropy_threshold: reasons.append(f"λ1: entropy {entropy:.3f}")
        if mem_bias <= self.min_memory_bias: reasons.append(f"λ2: memory {mem_bias:.3f}")
        if conf < self.min_confidence: reasons.append(f"λ3: confidence {conf:.3f}")
        if energy >= self.max_energy: reasons.append(f"λ4: energy {energy:.3f}")
        if self.asset_bias == 'bearish' and direction == 'bullish': reasons.append("λ5: bias")
        if self.asset_bias == 'bullish' and direction == 'bearish': reasons.append("λ5: bias")
        if size > self.max_pos/100: reasons.append(f"λ6: size")
        if stop > 0.01: reasons.append(f"λ6: stop")
        return len(reasons) == 0, reasons

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    direction: str = ''
    size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl: float = 0.0
    pattern_sequence: str = ''
    confidence: float = 0.0
    status: str = 'OPEN'

class SMARTEXE:
    def __init__(self, config):
        self.config, self.asset = config, config.get('asset','USD_CAD')
        self.encoder = PatternEncoder()
        self.evaluator = PatternEvaluator()
        self.entropy_filter = EntropyFilter(config.get('entropy_threshold',0.85))
        self.memory = PatternMemory()
        self.geometric = GeometricValidator(config.get('max_energy',0.8))
        self.risk_mgr = RiskManager(
            config.get('asset_bias','neutral'), config.get('max_position_pct',2.0),
            config.get('min_position_pct',0.5), config.get('entropy_threshold',0.85),
            config.get('min_confidence',0.45), config.get('min_memory_bias',-1.0),
            config.get('max_energy',0.8))
        self.current_trade, self.trade_history, self.blocked_trades = None, [], []
        self.capital = config.get('initial_capital',10000.0)
        self.total_trades = self.winning_trades = self.total_pnl = 0
    
    def process_candle(self, timestamp, o, h, l, c, v=0):
        self.encoder.add_candle(o, h, l, c)
        seq = self.encoder.get_sequence_list()
        if len(seq) < 20: 
            print(f"[{timestamp.strftime('%H:%M:%S')}] Building... ({len(seq)}/20)")
            return None
        
        seq_str = ''.join(seq)
        eval_result = self.evaluator.evaluate_sequence(seq)
        pred_sym, delta, conf = self.evaluator.predict_next_symbol(seq)
        entropy_val = self.entropy_filter.calculate_entropy(seq)
        mem_bias, _ = self.memory.query_memory_bias(seq)
        direction = 'bullish' if eval_result['total'] > 0 else 'bearish'
        geom_ok, geom_metrics = self.geometric.validate(seq, direction)
        
        if self.current_trade:
            self._monitor_open_trade(timestamp, c)
            return None
        
        win_rate = 0.55 if self.total_trades == 0 else self.winning_trades/max(1,self.total_trades)
        size = self.risk_mgr.calculate_kelly_size(win_rate, 150, 100, conf, 1-entropy_val)
        stop_pct = PATTERN_STOPS.get(seq[-1], 0.01)
        
        allowed, reasons = self.risk_mgr.apply_lambda_gates(
            entropy_val, mem_bias, conf, geom_metrics['energy'], direction, size, stop_pct)
        
        print(f"[{timestamp.strftime('%H:%M:%S')}] {seq_str[-5:]} | {direction:7} | "
              f"Conf:{conf*100:.0f}% | Ent:{entropy_val:.2f} | Ene:{geom_metrics['energy']:.2f} | "
              f"Mem:{mem_bias:+.2f}")
        
        if not allowed:
            print(f"  ❌ BLOCKED: {' | '.join(reasons[:2])}")
            self.blocked_trades.append({'timestamp': timestamp, 'reasons': reasons})
            return None
        
        print(f"  ✅ TRADE: {direction.upper()} {size*100:.2f}% @ {c:.5f}")
        return self._execute_trade(timestamp, c, 'LONG' if direction=='bullish' else 'SHORT',
                                   size, stop_pct, seq_str, conf)
    
    def _execute_trade(self, ts, price, direction, size, stop_pct, pattern, conf):
        if direction == 'LONG':
            sl, tp = price * (1-stop_pct), price * (1+stop_pct*2)
        else:
            sl, tp = price * (1+stop_pct), price * (1-stop_pct*2)
        trade = Trade(entry_time=ts, entry_price=price, direction=direction, size=size,
                     stop_loss=sl, take_profit=tp, pattern_sequence=pattern, confidence=conf)
        self.current_trade = trade
        self.total_trades += 1
        return {'action': 'TRADE', 'trade': trade}
    
    def _monitor_open_trade(self, ts, price):
        if not self.current_trade: return
        t = self.current_trade
        exit_price = None
        reason = None
        
        if t.direction == 'LONG':
            if price <= t.stop_loss: exit_price, reason = t.stop_loss, 'STOP'
            elif price >= t.take_profit: exit_price, reason = t.take_profit, 'TARGET'
        else:
            if price >= t.stop_loss: exit_price, reason = t.stop_loss, 'STOP'
            elif price <= t.take_profit: exit_price, reason = t.take_profit, 'TARGET'
        
        if exit_price:
            self._close_trade(ts, exit_price, reason)
    
    def force_close_trade(self, ts, price, reason='MANUAL'):
        if self.current_trade:
            self._close_trade(ts, price, reason)
    
    def _close_trade(self, ts, price, reason):
        if not self.current_trade: return
        t = self.current_trade
        t.exit_time, t.exit_price, t.status = ts, price, 'CLOSED'
        t.pnl = ((price-t.entry_price)/t.entry_price*100) if t.direction=='LONG' else ((t.entry_price-price)/t.entry_price*100)
        self.capital *= (1 + t.pnl/100 * t.size)
        self.total_pnl += t.pnl
        if t.pnl > 0: 
            self.winning_trades += 1
            self.risk_mgr.consecutive_losses = 0
        else:
            self.risk_mgr.consecutive_losses += 1
        self.memory.add_pattern(list(t.pattern_sequence)[:20], t.pnl)
        self.trade_history.append(t)
        self.current_trade = None
        print(f"  🔒 CLOSE: {reason} | PnL:{t.pnl:+.2f}% | Cap:${self.capital:.2f}")
    
    def get_statistics(self):
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades/max(1,self.total_trades)*100),
            'total_pnl': self.total_pnl,
            'capital': self.capital,
            'blocked': len(self.blocked_trades),
            'memory': len(self.memory.memory)
        }

def generate_trending_data(n=150, trend='uptrend'):
    data, base = [], 1.3500
    for i in range(n):
        ts = datetime(2024, 1, 1, 9, 0) + timedelta(minutes=i)
        if trend == 'uptrend':
            drift = 0.0002 + (i/n)*0.0003
            o = base + random.uniform(-0.0002, 0.0004)
            c = o + drift + random.uniform(0, 0.0003)
            h = max(o, c) + random.uniform(0, 0.0002)
            l = min(o, c) - random.uniform(0, 0.0001)
        else:
            drift = -0.0002 - (i/n)*0.0003
            o = base + random.uniform(-0.0004, 0.0002)
            c = o + drift - random.uniform(0, 0.0003)
            h = max(o, c) + random.uniform(0, 0.0001)
            l = min(o, c) - random.uniform(0, 0.0002)
        data.append({'timestamp': ts, 'open': o, 'high': h, 'low': l, 'close': c})
        base = c
    return data

if __name__ == "__main__":
    print("=" * 70)
    print("🤖 SMART-EXE TRADING BOT - ACTIVE DEMONSTRATION")
    print("=" * 70)
    
    config = {
        'asset': 'USD_CAD', 'mode': 'paper', 'asset_bias': 'neutral',
        'initial_capital': 10000.0, 'max_position_pct': 2.0, 'min_position_pct': 0.5,
        'entropy_threshold': 0.85, 'min_confidence': 0.45, 
        'min_memory_bias': -1.0, 'max_energy': 0.8
    }
    
    bot = SMARTEXE(config)
    print(f"Initial Capital: ${config['initial_capital']:.2f}")
    print("-" * 70)
    
    print("\n📊 Generating UPTREND data (150 candles)...")
    data = generate_trending_data(150, 'uptrend')
    
    print("\n🚀 Running simulation...")
    print("=" * 70)
    
    for i, candle in enumerate(data):
        result = bot.process_candle(
            timestamp=candle['timestamp'],
            o=candle['open'], h=candle['high'], 
            l=candle['low'], c=candle['close']
        )
        if bot.current_trade and i % 15 == 0 and i > 20:
            bot.force_close_trade(candle['timestamp'], candle['close'], 'TIME_EXIT')
    
    if bot.current_trade:
        bot.force_close_trade(data[-1]['timestamp'], data[-1]['close'], 'END')
    
    stats = bot.get_statistics()
    print("\n" + "=" * 70)
    print("📈 FINAL RESULTS")
    print("=" * 70)
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    print(f"Total PnL: {stats['total_pnl']:+.2f}%")
    print(f"Final Capital: ${stats['capital']:.2f}")
    print(f"Return: {((stats['capital']-10000)/10000*100):+.2f}%")
    print(f"Blocked: {stats['blocked']} | Memory: {stats['memory']}")
    print("=" * 70)
    
    if bot.trade_history:
        print("\n📋 TRADES:")
        for i, t in enumerate(bot.trade_history, 1):
            icon = "✅" if t.pnl > 0 else "❌"
            print(f"{icon} {i}: {t.direction:5} | Entry:{t.entry_price:.5f} | "
                  f"Exit:{t.exit_price:.5f} | PnL:{t.pnl:+.2f}%")
