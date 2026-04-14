
"""
governance_hierarchy.py - Full Trading Governance Stack
L0 Manual → L1 Mandra → L2 SOS27X → L3 Koopman → L4 Adelic → L5 UROL
Atomic Authority Chain | Production Ready
"""

import asyncio
import redis
import json
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import signal
import sys

class GovernanceLevel(Enum):
    L0_MANUAL = 0      # Human kill switch
    L1_MANDRA = 1      # Atomic risk gates
    L2_SOS27X = 2      # Spectral sentinel
    L3_KOOPMAN = 3     # Regime operator
    L4_ADELIC = 4      # IPDA core
    L5_UROL = 5        # Data layer

@dataclass
class GovernanceSignal:
    action: str
    size: float
    authority_level: GovernanceLevel
    confidence: float
    gates_active: List[str]
    final_size: float
    veto_reason: Optional[str] = None

class TradingGovernanceHierarchy:
    def __init__(self, equity: float = 100_000.0):
        self.equity = equity
        self.redis = redis.Redis(host='localhost', port=6379, db=4)
        
        # Authority state
        self.current_level = GovernanceLevel.L5_UROL
        self.global_halt = False
        self.manual_override = False
        
        # Gate tracking
        self.active_gates = set()
        self.position_count = 0
        
        # Emergency state
        self.init_emergency_db()
        
        print("🏛️ GOVERNANCE HIERARCHY LIVE | L0-L5 Authority Chain")
    
    def init_emergency_db(self):
        """L0 Emergency persistence."""
        self.redis.set('governance:global_state', json.dumps({
            'halt_status': False,
            'authority_level': 'L5_UROL',
            'active_gates': [],
            'timestamp': time.time()
        }))
    
    # ========================================
    # L0: MANUAL OVERRIDE (HUMAN AUTHORITY)
    # ========================================
    def l0_manual_halt(self, halt: bool = True):
        """Emergency human kill switch."""
        self.manual_override = halt
        self.redis.set('governance:L0_halt', halt)
        status = "🛑 GLOBAL HALT" if halt else "🟢 MANUAL RESUMED"
        print(f"🚨 L0 MANUAL OVERRIDE: {status}")
    
    # ========================================
    # L1: MANDRA RISK GATES (ATOMIC AUTHORITY)
    # ========================================
    async def l1_mandra_check(self, signal: Dict) -> GovernanceSignal:
        """Level 1-4 atomic gates."""
        # Level 4: 12% DD circuit breaker
        state = json.loads(self.redis.get('trading:global_state') or '{}')
        drawdown = 1 - (state.get('equity', self.equity) / self.equity)
        
        if drawdown > 0.12:
            self.active_gates.add('L4_DRAWDOWN')
            return GovernanceSignal('HALT', 0.0, GovernanceLevel.L1_MANDRA, 0.0, 
                                  ['L4_DRAWDOWN'], 0.0, '12% Circuit Breaker')
        
        # Level 3: Max 3 positions
        positions = json.loads(self.redis.get('trading:positions') or '[]')
        if len(positions) >= 3:
            self.active_gates.add('L3_CONCURRENCY')
            return GovernanceSignal('HOLD', 0.0, GovernanceLevel.L1_MANDRA, 0.0, 
                                  ['L3_CONCURRENCY'], 0.0, 'Max Positions')
        
        # Level 2: Volatility gate
        tick = json.loads(self.redis.xrevrange(f'clean:ticks:{signal["symbol"]}', count=1)[0][1][b'payload'])
        atr_current = tick.get('atr', 0.001)
        atr_ema20 = np.mean([t.get('atr', 0.001) for t in 
                           self.redis.xrevrange(f'clean:ticks:{signal["symbol"]}', count=20)])
        
        if atr_current > 2.0 * atr_ema20:
            self.active_gates.add('L2_VOLATILITY')
            return GovernanceSignal('REDUCE', signal['size']*0.5, GovernanceLevel.L1_MANDRA, 
                                  signal['confidence'], ['L2_VOLATILITY'], signal['size']*0.5)
        
        # L1 Kelly pass
        return GovernanceSignal(signal['action'], signal['size'], GovernanceLevel.L1_MANDRA,
                              signal['confidence'], list(self.active_gates), signal['size'])
    
    # ========================================
    # L2: SOS-27-X SPECTRAL VALIDATION
    # ========================================
    async def l2_sos27x_validate(self, signal: Dict) -> bool:
        """Spectral confidence threshold."""
        if signal.get('sos27x', {}).get('confidence', 0) < 0.6:
            self.active_gates.add('L2_SPECTRAL_LOW')
            return False
        return True
    
    # ========================================
    # L3: KOOPMAN REGIME ALIGNMENT
    # ========================================
    async def l3_koopman_check(self, symbol: str) -> float:
        """Regime stability score."""
        koopman_msgs = self.redis.xrevrange('koopman:signals', count=1)
        if not koopman_msgs:
            return 0.5
        
        regime_score = json.loads(koopman_msgs[0][1][b'payload']).get('koopman_regime', 0.5)
        return regime_score if regime_score > 0.3 else 0.0
    
    # ========================================
    # FULL AUTHORITY CHAIN
    # ========================================
    async def authority_chain(self, raw_signal: Dict) -> GovernanceSignal:
        """L0→L5 complete governance."""
        
        # L0: Manual override (absolute authority)
        if self.manual_override:
            return GovernanceSignal('HALT', 0.0, GovernanceLevel.L0_MANUAL, 0.0, 
                                  ['L0_MANUAL'], 0.0, 'Human Override')
        
        # L1: Mandra gates (atomic veto power)
        mandra_result = await self.l1_mandra_check(raw_signal)
        if mandra_result.veto_reason:
            return mandra_result
        
        # L2: SOS-27-X spectral validation
        sos_valid = await self.l2_sos27x_validate(raw_signal)
        if not sos_valid:
            return GovernanceSignal('HOLD', 0.0, GovernanceLevel.L2_SOS27X, 0.0,
                                  ['L2_SPECTRAL_LOW'], 0.0, 'Spectral Threshold')
        
        # L3: Koopman regime alignment
        regime_score = await self.l3_koopman_check(raw_signal['symbol'])
        if regime_score < 0.3:
            return GovernanceSignal('HOLD', 0.0, GovernanceLevel.L3_KOOPMAN, regime_score,
                                  ['L3_REGIME_MISMATCH'], 0.0, 'Regime Misalignment')
        
        # L4-L5: Signal passes all gates
        return GovernanceSignal(
            action=raw_signal['action'],
            size=raw_signal['size'],
            authority_level=GovernanceLevel.L1_MANDRA,
            confidence=raw_signal['confidence'],
            gates_active=list(self.active_gates),
            final_size=raw_signal['size'],
            veto_reason=None
        )
    
    # ========================================
    # PRODUCTION GOVERNANCE LOOP
    # ========================================
    async def governance_loop(self):
        """Live authority chain execution."""
        print("🏛️ GOVERNANCE HIERARCHY ACTIVE | L0-L5 Chain")
        
        while True:
            try:
                # Consume all upstream signals
                streams = ['sos27x:sentinel', 'koopman:signals', 'mandra:final', 'adelic:signals']
                msgs = self.redis.xread(streams, block=100, count=1)
                
                for stream, msg in msgs:
                    raw_signal = json.loads(msg[1][b'payload'])
                    symbol = raw_signal['symbol']
                    
                    # Full authority chain
                    final_signal = await self.authority_chain(raw_signal)
                    
                    # Publish FINAL EXECUTION AUTHORITY
                    execution_signal = {
                        'symbol': symbol,
                        'action': final_signal.action,
                        'final_size': final_signal.final_size,
                        'authority_level': final_signal.authority_level.value,
                        'gates_active': final_signal.gates_active,
                        'veto_reason': final_signal.veto_reason,
                        'timestamp': time.time()
                    }
                    
                    self.redis.xadd('governance:EXECUTE', {'payload': json.dumps(execution_signal)})
                    
                    # Live status
                    status = f"✅ EXECUTE {final_signal.final_size:.2f}" if final_signal.final_size > 0 else f"🚫 {final_signal.veto_reason}"
                    print(f"🏛️ [{symbol}] L{final_signal.authority_level.value} {status}")
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                print(f"❌ GOVERNANCE ERROR: {e}")
                await asyncio.sleep(0.1)
    
    # ========================================
    # EMERGENCY HANDLERS
    # ========================================
    def emergency_halt(self):
        """Global system halt."""
        self.l0_manual_halt(True)
        self.redis.set('governance:EMERGENCY_HALT', '1')
        print("🛑 EMERGENCY GLOBAL HALT ACTIVATED")
    
    def signal_handler(self, sig, frame):
        """Graceful shutdown."""
        print("\n🛑 Graceful shutdown...")
        self.emergency_halt()
        sys.exit(0)

# ========================================
# LAUNCH GOVERNANCE HIERARCHY
# ========================================
async def main():
    hierarchy = TradingGovernanceHierarchy(equity=100_000.0)
    
    # Emergency signal handler
    signal.signal(signal.SIGINT, hierarchy.signal_handler)
    signal.signal(signal.SIGTERM, hierarchy.signal_handler)
    
    await hierarchy.governance_loop()

if __name__ == "__main__":
    asyncio.run(main())
