import pytest
from src.orchestrator.sos27x_sentinel import SOS27XProductionSystem

def test_initialization():
    system = SOS27XProductionSystem()
    assert system.max_positions == 3
    assert system.balance == 50000.0
