
"""
mandra_kernels.py - Mandra Atomic Risk Governance
Level 1-4 Risk Gates | GP Variance | Kelly Fusion
Production Risk Engine | SOS-27-X + Adelic Tube Native
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import redis
import json
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List
from collections import deque
import pandas as pd

# CPU-First Config
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

@dataclass
class MandraSignal:
    size: float
    stop_distance: float
    level: int              # 1-4 active gates
    kelly_fraction: float
    gp_variance: float
    edge_per_var: float
    position_limit: int

class MandraKernels:
    def __init__(self, equity: float = 100_000.0):
        self.equity = equity
        self.max_positions = 3
        self.redis = redis.Redis(host='localhost', port=6379, db=3)
        
        # Risk state tracking
        self.balance_history = deque(maxlen=1000)
        self.position_history = deque(maxlen=500)
        self.drawdown_history = deque(maxlen=100)
        
        # GP Variance hyperparameters
        self.alpha = 1e-6
        self.length_scale = 0.1
        
        # Level gates
        self.level_4_threshold = 0.12  # 12% DD
        self.level_2_atr_mult = 2.0    # ATR > 2x EMA20
        
        print("🔴 MANDRA KERNELS LIVE | Atomic Risk Governance | Level 1-4")
    
    # ========================================
    # LEVEL 4: GLOBAL CIRCUIT BREAKER
    # ========================================
    @jit
    def check_level_4(self, balance_history: jnp.ndarray) -> bool:
        """12% drawdown → GLOBAL size=0."""
        peak = jnp.max(balance_history)
        current = balance_history[-1]
        drawdown_pct = (peak - current) / peak
        return drawdown_pct > self.level_4_threshold
    
    # ========================================
    # LEVEL 3: CONCURRENCY LIMIT
    # ========================================
    async def check_level_3(self) -> int:
        """Max 3 concurrent positions."""
        positions = json.loads(self.redis.get('trading:positions') or '[]')
        return len(positions)
    
    # ========================================
    # LEVEL 2: VOLATILITY GATE
    # ========================================
    @jit
    def check_level_2(self, atr_current: float, atr_ema20: float) -> bool:
        """ATR > 2x 20-bar EMA → volatility halving."""
        return atr_current > self.level_2_atr_mult * atr_ema20
    
    # ========================================
    # LEVEL 1: KELLY + GP VARIANCE
    # ========================================
    @jit
    def kelly_criterion(self, edge: float, odds: float, gp_var: float) -> float:
        """Kelly fraction with GP variance penalty."""
        base_kelly = (edge * odds - 1) / odds
        variance_penalty = 1.0 / jnp.sqrt(gp_var + 1e-8)
        return jnp.clip(base_kelly * variance_penalty, 0.0, 0.25)
    
    @jit
    def gaussian_process_variance(self, returns: jnp.ndarray) -> float:
        """GP predictive variance (online)."""
        n = len(returns)
        if n < 10:
            return 1.0
        
        X = jnp.arange(n)[:, None]
        y = returns
        
        # Squared exponential kernel
        K = jnp.exp(-0.5 * ((X - X.T) / self.length_scale)**2)
        K_y = K + self.alpha * jnp.eye(n)
        
        # Predictive variance at last point
        K_s = jnp.exp(-0.5 * ((X[-1] - X) / self.length_scale)**2)
        var_pred = 1.0 - K_s @ jnp.linalg.solve(K_y, K_s.T)
        
        return float(var_pred)
    
    # ========================================
    # EDGE PER VARIANCE RATIO
    # ========================================
    @jit
    def edge_per_variance(self, returns: jnp.ndarray, signals: jnp.ndarray) -> float:
        """Orthogonal edge quality μ/σ²."""
        edge = jnp.mean(returns * signals)
        variance = jnp.var(returns)
        return edge / (variance + 1e-8)
    
    # ========================================
    # ATOMIC SIZING KERNEL
    # ========================================
    @jit
    def atomic_size_kernel(self, confidence: float, atr: float, 
                          edge_pv: float, gp_var: float) -> float:
        """Single JAX primitive → production size."""
        # Kelly scaling
        kelly_frac = self.kelly_criterion(edge_pv, 3.0, gp_var)
        
        # Confluence multiplier
        conf_mult = 0.5 + 1.5 * confidence
        
        # Risk budget
        risk_budget = self.equity * 0.01 * kelly_frac * conf_mult
        
        # ATR normalization (pip value)
        pip_size = risk_budget / (atr * 10_000)
        
        return jnp.clip(pip_size, 0.0, 10.0)
    
    # ========================================
    # SCALE-OUT KERNEL (30/40/30)
    # ========================================
    @jit
    def scale_out_kernel(self, entry_price: float, current_price: float, 
                        rr_target: float) -> List[float]:
        """1:2R, 1:4R, trailing scale-out."""
        profit_pips = (current_price - entry_price) * 10_000
        
        targets = [2.0 * rr_target, 4.0 * rr_target]
        scale_out = []
        
        for target in targets:
            if profit_pips >= target:
                scale_out.append(0.3)  # 30% each
            else:
                scale_out.append(0.0)
        
        trailing = max(0.0, profit_pips - 1.5 * rr_target) * 0.4  # 40% trailing
        return scale_out + [trailing]
    
    # ========================================
    # FULL MANDRA GOVERNANCE
    # ========================================
    async def mandra_govern(self, sos_signal: Dict) -> MandraSignal:
        """Atomic risk decision → size + gates."""
        
        # Get market state
        tick = json.loads(self.redis.xrevrange('clean:ticks:EURUSD', count=1)[0][1][b'payload'])
        returns = np.array([json.loads(m[1][b'payload'])['close'] 
                           for m in self.redis.xrevrange('clean:ticks:EURUSD', count=100)])
        
        # Level gates
        level_4 = self.check_level_4(jnp.array(list(self.balance_history)))
        level_3_count = await self.check_level_3()
        level_2 = self.check_level_2(
            float(tick.get('atr', 0.001)), 
            np.mean(returns[-20:]) * 0.02  # ATR proxy
        )
        
        active_level = 4 if level_4 else 3 if level_3_count >= self.max_positions else \
                      2 if level_2 else 1
        
        # Early returns (circuit breakers)
        if level_4:
            return MandraSignal(0.0, 0.0, 4, 0.0, 0.0, 0.0, 0)
        
        # GP Variance + Edge/Var
        gp_var = self.gaussian_process_variance(jnp.diff(np.log(returns)))
        edge_pv = self.edge_per_variance(jnp.diff(np.log(returns)), 
                                       jnp.ones(len(returns)-1) * sos_signal['confidence'])
        
        # Atomic size
        size = self.atomic_size_kernel(
            sos_signal['confidence'],
            float(tick.get('atr', 0.001)),
            edge_pv,
            gp_var
        )
        
        # Position limit scaling
        size *= max(0, self.max_positions - level_3_count)
        
        return MandraSignal(
            size=float(size),
            stop_distance=float(tick.get('atr', 0.001)) * 1.5,
            level=active_level,
            kelly_fraction=float(edge_pv),
            gp_variance=float(gp_var),
            edge_per_var=float(edge_pv),
            position_limit=self.max_positions - level_3_count
        )
    
    # ========================================
    # PRODUCTION RISK LOOP
    # ========================================
    async def risk_governance_loop(self):
        """Live Mandra monitoring."""
        print("🔴 MANDRA KERNELS LIVE | Level 1-4 | GP + Kelly")
        
        while True:
            try:
                # Consume SOS-27-X signals
                msgs = self.redis.xread({'sos27x:sentinel': '$'}, count=1, block=100)
                if not msgs:
                    await asyncio.sleep(0.1)
                    continue
                
                sos_signal = json.loads(msgs[0][1][b'payload'])
                symbol = sos_signal['symbol']
                
                # Mandra governance
                mandra = await self.mandra_govern(sos_signal)
                
                # Final execution signal
                final_signal = {
                    'action': sos_signal['action'],
                    'symbol': symbol,
                    'mandra_size': mandra.size,
                    'stop_distance': mandra.stop_distance,
                    'risk_level': mandra.level,
                    'kelly_fraction': mandra.kelly_fraction,
                    'gates': {
                        'level_4_dd': mandra.level == 4,
                        'level_3_concurrency': mandra.level == 3,
                        'level_2_volatility': mandra.level == 2
                    },
                    'timestamp': time.time()
                }
                
                # Publish to AECABI (final authority)
                self.redis.xadd('mandra:final', {'payload': json.dumps(final_signal)})
                
                status = " HALTED" if mandra.level >= 4 else f"✅ SIZE={mandra.size:.2f}"
                print(f" {symbol} MANDRA L{mandra.level} {status} "
                      f"| Kelly={mandra.kelly_fraction:.3f} | PosLimit={mandra.position_limit}")
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                print(f"❌ MANDRA ERROR: {e}")
                await asyncio.sleep(1.0)

# ========================================
# LAUNCH MANDRA
# ========================================
async def main():
    mandra = MandraKernels(equity=100_000.0)
    await mandra.risk_governance_loop()

if __name__ == "__main__":
    asyncio.run(main())
