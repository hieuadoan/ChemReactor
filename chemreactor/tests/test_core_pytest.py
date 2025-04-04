import pytest
import numpy as np
from chemreactor.core.base_reactor import Reaction, BaseReactor

# --- Test Reaction Class ---

def test_reaction_init():
    """Test Reaction class initialization."""
    reaction = Reaction(reactants={'A': 1}, products={'B': 1}, rate_constant=0.1)
    assert reaction.reactants == {'A': 1}
    assert reaction.products == {'B': 1}
    assert reaction.k == 0.1
    assert reaction.rate_equation is None

def test_calculate_rate_mass_action():
    """Test default mass action rate calculation."""
    # First order: A -> B
    reaction1 = Reaction(reactants={'A': 1}, products={'B': 1}, rate_constant=0.1)
    concentrations1 = {'A': 2.0, 'B': 0.0}
    rate1 = reaction1.calculate_rate(concentrations1)
    assert rate1 == pytest.approx(0.1 * 2.0**1)

    # Second order: 2A -> C
    reaction2 = Reaction(reactants={'A': 2}, products={'C': 1}, rate_constant=0.05)
    concentrations2 = {'A': 3.0, 'C': 0.0}
    rate2 = reaction2.calculate_rate(concentrations2)
    assert rate2 == pytest.approx(0.05 * 3.0**2)

    # Mixed order: A + B -> D
    reaction3 = Reaction(reactants={'A': 1, 'B': 1}, products={'D': 1}, rate_constant=0.2)
    concentrations3 = {'A': 2.0, 'B': 1.5, 'D': 0.0}
    rate3 = reaction3.calculate_rate(concentrations3)
    assert rate3 == pytest.approx(0.2 * 2.0**1 * 1.5**1)

def test_calculate_rate_custom_equation():
    """Test custom rate equation."""
    def custom_rate(k, conc):
        return k * conc['A'] / (1 + conc['B'])

    reaction = Reaction(reactants={'A': 1}, products={'C': 1}, rate_constant=0.5, rate_equation=custom_rate)
    concentrations = {'A': 4.0, 'B': 1.0, 'C': 0.0}
    rate = reaction.calculate_rate(concentrations)
    assert rate == pytest.approx(0.5 * 4.0 / (1 + 1.0))

# --- Test BaseReactor Methods (using a Dummy Reactor) ---

class DummyReactor(BaseReactor):
     def run(self, end_time: float, time_points: np.ndarray | None = None) -> dict:
         pass # Not needed for these tests

@pytest.fixture
def initial_conc_br():
    return {'A': 10.0, 'B': 1.0, 'C': 0.0}

@pytest.fixture
def reactions_br():
    r1 = Reaction(reactants={'A': 1}, products={'B': 1}, rate_constant=0.1) # A -> B
    r2 = Reaction(reactants={'B': 1}, products={'C': 1}, rate_constant=0.05) # B -> C
    return [r1, r2]

@pytest.fixture
def dummy_reactor(initial_conc_br, reactions_br):
    return DummyReactor(initial_concentrations=initial_conc_br, reactions=reactions_br)

def test_calculate_rates(dummy_reactor, reactions_br):
    """Test the net rate calculation for multiple reactions."""
    current_conc = {'A': 5.0, 'B': 5.0, 'C': 1.0}
    net_rates = dummy_reactor.calculate_rates(current_conc)

    r1, r2 = reactions_br
    rate_r1 = r1.calculate_rate(current_conc) # 0.1 * 5.0 = 0.5
    rate_r2 = r2.calculate_rate(current_conc) # 0.05 * 5.0 = 0.25

    assert net_rates['A'] == pytest.approx(-rate_r1) # Consumed in r1
    assert net_rates['B'] == pytest.approx(rate_r1 - rate_r2) # Produced in r1, consumed in r2
    assert net_rates['C'] == pytest.approx(rate_r2) # Produced in r2

def test_reset(dummy_reactor, initial_conc_br):
    """Test resetting the reactor state."""
    # Simulate some change
    dummy_reactor.concentrations = {'A': 0.5, 'B': 0.5, 'C': 1.0} # Adjusted to include C
    dummy_reactor.time = 10.0
    # Note: pytest doesn't automatically track history like unittest's setUp might imply
    # For stateful tests, manage history explicitly or redesign if possible.
    # Here, we just test the reset logic based on initial state.
    dummy_reactor.history = {
        'time': [0.0, 10.0],
        'concentrations': [initial_conc_br.copy(), dummy_reactor.concentrations.copy()],
        'temperature': [dummy_reactor.temperature],
        'pressure': [dummy_reactor.pressure]
     }


    dummy_reactor.reset()

    assert dummy_reactor.concentrations == initial_conc_br
    assert dummy_reactor.time == 0.0
    assert len(dummy_reactor.history['time']) == 1
    assert dummy_reactor.history['time'][0] == 0.0
    assert len(dummy_reactor.history['concentrations']) == 1
    assert dummy_reactor.history['concentrations'][0] == initial_conc_br

def test_residence_time():
    """Test residence time calculation."""
    initial_conc = {'A': 1.0}
    reactions = []
    # Test with flow rate
    reactor_flow = DummyReactor(initial_concentrations=initial_conc, reactions=reactions, volume=2.0, flow_rate=0.5)
    assert reactor_flow.residence_time == pytest.approx(2.0 / 0.5)

    # Test without flow rate (e.g., Batch)
    reactor_no_flow = DummyReactor(initial_concentrations=initial_conc, reactions=reactions, volume=2.0, flow_rate=None)
    assert reactor_no_flow.residence_time is None

    # Test with zero flow rate
    reactor_zero_flow = DummyReactor(initial_concentrations=initial_conc, reactions=reactions, volume=2.0, flow_rate=0)
    assert reactor_zero_flow.residence_time is None
