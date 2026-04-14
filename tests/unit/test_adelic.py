import pytest
from reverse_period.core.adelic import padic_valuation, adelic_check

def test_padic_valuation():
    assert padic_valuation(12, 2) == 2   # 12 = 2^2 * 3
    assert padic_valuation(12, 3) == 1   # 12 = 2^2 * 3^1
    assert padic_valuation(1/4, 2) == -2 # 4 = 2^2 in denominator

def test_adelic_manifold():
    # Rational number with finite support
    assert adelic_check(15/8, primes=[2,3,5,7,11,13], max_nonzero=3) == True
    
    # Too many nonzero valuations
    assert adelic_check(2*3*5*7*11/13, primes=[2,3,5,7,11,13], max_nonzero=3) == False
