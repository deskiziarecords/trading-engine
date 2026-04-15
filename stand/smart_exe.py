#!/usr/bin/env python3
"""
SMART-EXE Single-Asset Trading Bot
Complete Python Implementation
"""

import numpy as np
import json
import logging
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
    handlers=[logging.FileHandler('smart_exe.log'), logging.StreamHandler()])
logger = logging.getLogger('SMART-EXE')

# Configuration
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
        symbol = self.encode_candle(o, h, l, c)
        self.sequence.append(symbol)
        return symbol
    def get_sequence_list(self): return list(self.sequence)

class PatternEvaluator:
    def __init__(self):
        self.position_tables = {s: np.clip(-60 + (np.arange(64)%8)*15 + (np.arange(64)//8)*5, -60, 60).reshape(8,8) 
                               for s in ['B','I','W','w','U','D','X']}
    def evaluate_sequence(self, seq):
        if not seq: return {'total': 0, 'direction': 'neutral'}
        material = sum(SYMBOL_VALUES[s] * (i+1)/len(seq) for i,s in enumerate(seq))
        position = sum(self.position_tables[s][min(i,63)//8, min(i,63)%8] * (i+1)/len(seq) for i,s in enumerate(seq))
        total = material + position
        return {'material': material, 'position': position, 'total': total, 
                'direction': 'bullish' if total > 0 else 'bearish' if total < 0 else 'neutral'}
    def predict_next_symbol(self, seq):
        if len(seq) < 5: return 'X', 0, 0
        base = self.evaluate_sequence(seq)['total']
        best, best_delta = 'X', 0
        for s in ['B','I','W','w','U','D','X']:
            delta = abs(self.evaluate_sequence(seq + [s])['total'] - base)
            if delta > best_delta: best, best_delta = s, delta
        return best, best_delta, min(1.0, best_delta/2000)

class EntropyFilter:
    def __init__(self, threshold=0.6): self.threshold = threshold
    def calculate_entropy(self, seq):
        if not seq: return 1.0
        counts = {}
        for s in seq: counts[s] = counts.get(s, 0) + 1
        entropy = sum(-(c/len(seq))*np.log2(c/len(seq)) for c in counts.values())
        return entropy / np.log2(7)
    def check_trade_allowed(self, seq):
        e = self.calculate_entropy(seq)
        return e < self.threshold, e, "OK" if e < self.threshold else "BLOCK"

class PatternMemory:
    def __init__(self, max_size=10000, k=5): self.max_size, self.k, self.memory = max_size, k, []
    def _seq_to_vec(self, seq):
        vec = np.zeros(140, dtype=np.float32)
        for i,s in enumerate(seq[:20]):
            if s in 'BIWwUDX': vec[i*7 + 'BIWwUDX'.index(s)] = 1
        return vec
    def add_pattern(self, seq, outcome):
        self.memory.append((self._seq_to_vec(seq), outcome))
        if len(self.memory) > self.max_size: self.memory.pop(0)
    def query_memory_bias(self, seq):
        if len(self.memory) < self.k: return 0, len(self.memory), "INSUFFICIENT"
        q = self._seq_to_vec(seq)
        dists = sorted([(np.sqrt(np.sum((q-v)**2)), o) for v,o in self.memory])[:self.k]
        return np.mean([o for _,o in dists]), self.k, "OK"

class GeometricValidator:
    def __init__(self, energy_threshold=0.5, curl_threshold=0.8): 
        self.energy_threshold, self.curl_threshold = energy_threshold, curl_threshold
    def calculate_energy(self, seq):
        if len(seq) < 3: return 1.0
        emb = np.array([SYMBOL_VALUES.get(s,0)/1000 for s in seq], dtype=np.float32)
        diffs = np.diff(emb)
        curve = np.diff(diffs) if len(diffs) >= 2 else [0]
        return min(1.0, (np.sum(diffs**2) + np.sum(np.array(curve)**2)) / len(seq) / 10)
    def calculate_divergence(self, seq):
        if len(seq) < 4: return 0
        mid = len(seq)//2
        m1 = sum(SYMBOL_VALUES.get(s,0) for s in seq[:mid])/mid
        m2 = sum(SYMBOL_VALUES.get(s,0) for s in seq[mid:])/(len(seq)-mid)
        return (m2-m1)/1000
    def calculate_curl(self, seq):
        if len(seq) < 3: return 0
        changes = sum(2 if (a in 'BUW' and b in 'IDw') or (a in 'IDw' and b in 'BUW') else 
                     1 if (a in 'Ww') != (b in 'Ww') else 0 
                     for a,b in zip(seq, seq[1:]))
        return min(1.0, changes / (len(seq)*2))
    def validate(self, seq, direction):
        e, d, c = self.calculate_energy(seq), self.calculate_divergence(seq), self.calculate_curl(seq)
        metrics = {'energy': e, 'divergence': d, 'curl': c}
        if e >= self.energy_threshold: return False, metrics, f"ENERGY_BLOCK: {e:.3f}"
        if c > self.curl_threshold: return False, metrics, f"CURL_BLOCK: {c:.3f}"
        if direction == 'bullish' and d < -0.3: return False, metrics, f"DIV_BLOCK: {d:.3f}"
        if direction == 'bearish' and d > 0.3: return False, metrics, f"DIV_BLOCK: {d:.3f}"
        return True, metrics, "GEOM_OK"

class RiskManager:
    def __init__(self, asset_bias='neutral', max_pos=2.0, min_pos=0.5, 
                 entropy=0.6, conf=0.6, mem=0.1, energy=0.5):
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
        if self.asset_bias == 'bearish' and direction == 'bullish': reasons.append("λ5: bias mismatch")
        if self.asset_bias == 'bullish' and direction == 'bearish': reasons.append("λ5: bias mismatch")
        if size > self.max_pos/100: reasons.append(f"λ6: size {size*100:.2f}%")
        if stop > 0.01: reasons.append(f"λ6: stop {stop*100:.2f}%")
        return len(reasons) == 0, reasons
    def adjust_for_consecutive_losses(self, size):
        return size * (0.5 if self.consecutive_losses >= 3 else 0.75 if self.consecutive_losses >= 2 else 1)

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
    entropy: float = 0.0
    status: str = 'OPEN'

class SMARTEXE:
    def __init__(self, config):
        self.config, self.asset, self.mode = config, config.get('asset','USD_CAD'), config.get('mode','paper')
        self.encoder = PatternEncoder()
        self.evaluator = PatternEvaluator()
        self.entropy_filter = EntropyFilter(config.get('entropy_threshold',0.6))
        self.memory = PatternMemory()
        self.geometric = GeometricValidator(config.get('max_energy',0.5))
        self.risk_mgr = RiskManager(
            config.get('asset_bias','neutral'), config.get('max_position_pct',2.0),
            config.get('min_position_pct',0.5), config.get('entropy_threshold',0.6),
            config.get('min_confidence',0.6), config.get('min_memory_bias',0.1),
            config.get('max_energy',0.5))
        self.current_trade, self.trade_history, self.blocked_trades = None, [], []
        self.capital = config.get('initial_capital',10000.0)
        self.total_trades = self.winning_trades = self.total_pnl = 0
    
    def process_candle(self, timestamp, o, h, l, c, v=0):
        symbol = self.encoder.add_candle(o, h, l, c)
        seq = self.encoder.get_sequence_list()
        if len(seq) < 20: return None
        
        eval_result = self.evaluator.evaluate_sequence(seq)
        pred_sym, delta, conf = self.evaluator.predict_next_symbol(seq)
        entropy_ok, entropy_val, _ = self.entropy_filter.check_trade_allowed(seq)
        mem_bias, _, _ = self.memory.query_memory_bias(seq)
        direction = 'bullish' if eval_result['total'] > 0 else 'bearish'
        geom_ok, geom_metrics, _ = self.geometric.validate(seq, direction)
        
        if self.current_trade:
            self._monitor_open_trade(timestamp, c)
            return None
        
        win_rate = 0.55 if self.total_trades == 0 else self.winning_trades/max(1,self.total_trades)
        size = self.risk_mgr.calculate_kelly_size(win_rate, 150, 100, conf, 1-entropy_val)
        size = self.risk_mgr.adjust_for_consecutive_losses(size)
        stop_pct = PATTERN_STOPS.get(symbol, 0.01)
        
        allowed, reasons = self.risk_mgr.apply_lambda_gates(
            entropy_val, mem_bias, conf, geom_metrics['energy'], direction, size, stop_pct)
        
        if not allowed:
            self.blocked_trades.append({'timestamp': timestamp, 'reasons': reasons, 
                'sequence': ''.join(seq), 'metrics': {'entropy': entropy_val, 'memory': mem_bias, 
                'confidence': conf, 'energy': geom_metrics['energy']}})
            return None
        
        return self._execute_trade(timestamp, c, 'LONG' if direction=='bullish' else 'SHORT',
                                   size, stop_pct, ''.join(seq), conf, entropy_val)
    
    def _execute_trade(self, ts, price, direction, size, stop_pct, pattern, conf, entropy):
        if direction == 'LONG':
            sl, tp = price * (1-stop_pct), price * (1+stop_pct*2)
        else:
            sl, tp = price * (1+stop_pct), price * (1-stop_pct*2)
        trade = Trade(entry_time=ts, entry_price=price, direction=direction, size=size,
                     stop_loss=sl, take_profit=tp, pattern_sequence=pattern, confidence=conf, entropy=entropy)
        self.current_trade = trade
        self.total_trades += 1
        logger.info(f"TRADE: {direction} {size*100:.2f}% @ {price:.5f}")
        return {'action': 'TRADE', 'trade': trade}
    
    def _monitor_open_trade(self, ts, price):
        if not self.current_trade: return
        t = self.current_trade
        if t.direction == 'LONG':
            if price <= t.stop_loss: self._close_trade(ts, price, 'STOP')
            elif price >= t.take_profit: self._close_trade(ts, price, 'TARGET')
        else:
            if price >= t.stop_loss: self._close_trade(ts, price, 'STOP')
            elif price <= t.take_profit: self._close_trade(ts, price, 'TARGET')
    
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
        if len(t.pattern_sequence) >= 20:
            self.memory.add_pattern(list(t.pattern_sequence)[:20], t.pnl)
        self.trade_history.append(t)
        self.current_trade = None
        logger.info(f"CLOSE: {reason} | PnL: {t.pnl:+.2f}% | Capital: ${self.capital:.2f}")
    
    def get_statistics(self):
        return {
            'total_trades': self.total_trades,
            'win_rate': (self.winning_trades/max(1,self.total_trades)*100),
            'total_pnl': self.total_pnl,
            'capital': self.capital,
            'blocked': len(self.blocked_trades),
            'memory': len(self.memory.memory)
        }

# Example usage
if __name__ == "__main__":
    config = {
        'asset': 'USD_CAD', 'mode': 'paper', 'asset_bias': 'neutral',
        'initial_capital': 10000, 'max_position_pct': 2.0, 'min_position_pct': 0.5,
        'entropy_threshold': 0.6, 'min_confidence': 0.6, 'min_memory_bias': 0.1, 'max_energy': 0.5
    }
    bot = SMARTEXE(config)
    print("SMART-EXE initialized!")
    
    # Test with sample data
    for i in range(25):
        result = bot.process_candle(
            datetime.now() + timedelta(minutes=i),
            1.3500 + i*0.0001, 1.3550 + i*0.0001, 
            1.3480 + i*0.0001, 1.3520 + i*0.0001
        )
        if result: print(f"Trade executed: {result['trade'].direction}")
    
    stats = bot.get_statistics()
    print(f"\nResults: {stats['total_trades']} trades, {stats['win_rate']:.1f}% win rate, ${stats['capital']:.2f}")
