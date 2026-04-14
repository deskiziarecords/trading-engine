@pytest.mark.slow
def test_reverse_period_recovery():
    """Full pipeline: synthetic chop → reverse detected → displacement → normal"""
    from reverse_period.meta.ensemble import MetaEnsemble
    
    # Generate synthetic regime: 20 bars of distribution-without-expansion
    synthetic = generate_stalled_distribution(bars=20, volatility=0.3)
    
    detector = MetaEnsemble.from_config({'obnfe': {}, 'tda': {}})
    reverses = []
    
    for tick in synthetic:
        r = detector.update(tick)
        reverses.append(r)
    
    # Should trigger reverse period
    assert any(reverses[10:18])  # Mid-period detection
    
    # Add displacement
    displacement_tick = generate_displacement(atr_multiple=2.5)
    r_post = detector.update(displacement_tick)
    
    # Should exit reverse mode
    assert detector.phase == 2  # Back to distribution/expansion
