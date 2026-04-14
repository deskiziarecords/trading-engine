#!/usr/bin/env python3
"""
Reverse Period Detector — CLI entry point
"""

import argparse
import asyncio
import yaml
from pathlib import Path

from .core.state import PhaseMachine
from .meta.obnfe import OnlineBayesianFusion
from .meta.tda import TDAReverseDetector
from .meta.ensemble import MetaEnsemble
from .io.stream import MarketFeed
from .risk.circuit import CircuitBreaker


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to asset config')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], default='paper')
    parser.add_argument('--checkpoint', help='Resume from OBNFE/TDA checkpoint')
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Initialize components
    phase = PhaseMachine(cfg['phases'])
    obnfe = OnlineBayesianFusion(**cfg['obnfe'])
    tda = TDAReverseDetector(**cfg['tda'])
    ensemble = MetaEnsemble(obnfe, tda, weights=cfg['ensemble_weights'])
    circuit = CircuitBreaker(threshold=cfg['risk']['max_drawdown'])
    
    # Restore state if checkpoint provided
    if args.checkpoint:
        ensemble.load(args.checkpoint)
    
    # Main event loop
    feed = MarketFeed(cfg['feed'])
    async for tick in feed.subscribe():
        # Update all detectors
        sigma = phase.current()
        lambdas = compute_lambdas(tick, cfg['detectors'])
        
        r_obnfe = obnfe.update(lambdas)
        r_tda = tda.update(tick, sigma)
        r_ensemble = ensemble.fuse(r_obnfe, r_tda, sigma)
        
        # Adelic constraint
        s_hat = compute_signal(tick)
        if not adelic_check(s_hat, cfg['adelic']):
            r_ensemble = 0  # Force no-trade
        
        # State transition
        if r_ensemble and ensemble.severity > cfg['thresholds']['reverse']:
            phase.transition(0)  # Reset to accumulation
            circuit.record_reverse()
        
        # Risk check
        if circuit.check_drawdown():
            await emergency_shutdown(ensemble, cfg['persistence'])
            break
        
        # Execution decision
        if phase.current() == 2 and not r_ensemble:
            size = position_size(tick, ensemble.confidence)
            route = schur_optimize(tick['orderbook'], size)
            await execute(route, mode=args.mode)


if __name__ == '__main__':
    asyncio.run(main())
