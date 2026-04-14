
"""
koopman_operator.py - Koopman + DMD para Trading Regímenes IPDA
EDMD + Hankel Embeddings | CPU-Optimized | Adelic Tube Integration
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
import redis
import json
import time
from dataclasses import dataclass
from typing import Tuple, Dict
from collections import deque

# CPU Config
jax.config.update('jax_platform_name', 'cpu')

@dataclass
class KoopmanState:
    K_hat: jnp.ndarray     # Koopman matrix approx
    eigenvalues: jnp.ndarray
    modes: jnp.ndarray
    residuals: float
    embedding_dim: int

class KoopmanTrader:
    def __init__(self, embedding_dim: int = 64, delay_dim: int = 8):
        self.embed_dim = embedding_dim
        self.delay_dim = delay_dim
        self.redis = redis.Redis(host='localhost', port=6379, db=1)
        
        # Hankel buffers
        self.hankel_buffer = deque(maxlen=embedding_dim * delay_dim)
        
        # Learned Koopman operator
        self.koopman_state: Optional[KoopmanState] = None
    
    def hankel_embedding(self, time_series: jnp.ndarray, delay: int = 1) -> jnp.ndarray:
        """Hankel-Takens embedding [delay_dim x embedding_dim]."""
        n = len(time_series)
        if n < self.embed_dim * self.delay_dim:
            return jnp.zeros((self.delay_dim, self.embed_dim))
        
        # Delay coordinates
        delays = jnp.array([time_series[i::delay][:self.embed_dim] 
                           for i in range(self.delay_dim)])
        return delays.T  # [embed_dim, delay_dim]
    
    @jit
    def edmd_koopman(self, X: jnp.ndarray, Y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Extended DMD: Koopman operator approximation.
        X → Y = K̂ · X  =>  K̂ = Y · pinv(X)
        """
        # SVD decomposition for stability
        U, s, Vh = jnp.linalg.svd(X, full_matrices=False)
        s_inv = jnp.diag(1.0 / (s + 1e-12))
        K_hat = Y @ U @ s_inv @ Vh
        
        # Residual ||Y - K̂X||
        Y_pred = K_hat @ X
        residual = jnp.linalg.norm(Y - Y_pred) / jnp.linalg.norm(Y)
        
        # Eigenvalues (Koopman spectrum)
        eigenvalues = jnp.linalg.eigvals(K_hat)
        
        return K_hat, eigenvalues, float(residual)
    
    def fit_koopman_online(self, new_data: jnp.ndarray):
        """Online EDMD: Sliding window + residual monitoring."""
        # Build Hankel matrices
        self.hankel_buffer.extend(new_data)
        data = jnp.array(list(self.hankel_buffer))
        
        if len(data) < self.embed_dim * 2:
            return
        
        # Snapshots: X[:-1], Y[1:]
        X = self.hankel_embedding(data[:-1])
        Y = self.hankel_embedding(data[1:])
        
        # Learn Koopman operator
        K_hat, evals, residual = self.edmd_koopman(X.T, Y.T)
        
        self.koopman_state = KoopmanState(
            K_hat=K_hat,
            eigenvalues=evals,
            modes=None,  # Full KMD offline
            residuals=residual,
            embedding_dim=self.embed_dim
        )
        
        # Residual threshold → local prediction
        if residual > 0.1:
            print(f"⚠️ High residual {residual:.3f} → Local mode")
    
    @jit
    def predict_horizon(self, state: jnp.ndarray, horizon: int = 10) -> jnp.ndarray:
        """K^h · φ(x) multi-step forecast."""
        if self.koopman_state is None:
            return state
        
        K_pow = jnp.linalg.matrix_power(self.koopman_state.K_hat, horizon)
        return K_pow @ state
    
    @jit
    def koopman_spectrum_score(self, state: jnp.ndarray) -> float:
        """Dominant Koopman mode alignment."""
        if self.koopman_state is None:
            return 0.5
        
        # Project onto dominant modes
        dominant_ev = jnp.argmax(jnp.abs(self.koopman_state.eigenvalues))
        mode_align = jnp.abs(state @ self.koopman_state.eigenvalues[dominant_ev])
        return float(jnp.tanh(mode_align))
    
    # ========================================
    # IPDA + KOOPMAN FUSION
    # ========================================
    async def koopman_ipda_signal(self, symbol: str) -> Dict:
        """Koopman-enhanced IPDA signal."""
        # UROL clean data
        tick_msgs = self.redis.xread({f'clean:ticks:{symbol}': '$'}, count=256)
        if not tick_msgs:
            return {'action': 'FLAT', 'confidence': 0.0}
        
        prices = jnp.array([json.loads(m[1][b'payload'])['close'] 
                           for m in tick_msgs[-128:]])
        
        # Online Koopman learning
        self.fit_koopman_online(prices)
        
        if self.koopman_state is None:
            return {'action': 'FLAT', 'confidence': 0.0}
        
        # Koopman prediction
        current_embed = self.hankel_embedding(prices)
        pred_10 = self.predict_horizon(current_embed.T[:, -1:], 10)
        
        # Regime score (Koopman alignment)
        regime_score = self.koopman_spectrum_score(current_embed.T[:, -1:])
        
        # IPDA phase (from Adelic Tube)
        phase = self.detect_phase_from_history(prices)
        
        # Fusion signal
        confidence = 0.6 * regime_score + 0.4 * (1.0 if phase in ['ACCUMULATION', 'DISTRIBUTION'] else 0.5)
        action = 'BUY' if regime_score > 0.7 and phase == 'ACCUMULATION' else \
                'SELL' if regime_score < 0.3 and phase == 'DISTRIBUTION' else 'FLAT'
        
        return {
            'action': action,
            'symbol': symbol,
            'confidence': float(confidence),
            'koopman_regime': float(regime_score),
            'phase': phase,
            'residual': float(self.koopman_state.residuals),
            'timestamp': time.time()
        }
    
    def detect_phase_from_history(self, prices: jnp.ndarray) -> str:
        """Simplified IPDA phase (from adelic_tube)."""
        if len(prices) < 20:
            return 'FLAT'
        
        df20 = prices[-2880:]  # 20 days 1min
        higher_low = prices[-1] > prices[-2]
        vol_proxy = jnp.std(prices[-5:])  # volatility as vol proxy
        
        if higher_low and vol_proxy < jnp.std(prices[-20:]):
            return 'ACCUMULATION'
        elif jnp.abs(prices[-1] - prices[-2]) > 2 * jnp.std(prices[-5:]):
            return 'MANIPULATION'
        elif prices[-1] < prices[-2] and vol_proxy > jnp.std(prices[-20:]):
            return 'DISTRIBUTION'
        
        return 'FLAT'

# ========================================
# PRODUCTION INTEGRATION LOOP
# ========================================
async def koopman_production_loop():
    """Koopman Operator → Adelic Tube signals."""
    koop = KoopmanTrader(embedding_dim=64, delay_dim=8)
    
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    
    while True:
        for symbol in symbols:
            signal = await koop.koopman_ipda_signal(symbol)
            
            # Publish to Adelic Tube / AECABI
            redis_client = redis.Redis(host='localhost', port=6379, db=0)
            redis_client.xadd('koopman:signals', {
                'payload': json.dumps(signal),
                'symbol': symbol
            })
            
            print(f"🌀 {symbol} KOOPMAN: {signal['action']} {signal['confidence']:.2f} "
                  f"| regime={signal['koopman_regime']:.2f} | {signal['phase']}")
        
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(koopman_production_loop())
