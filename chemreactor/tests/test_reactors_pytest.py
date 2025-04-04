import pytest
import numpy as np
from chemreactor.core.base_reactor import Reaction
from chemreactor.reactors.batch_reactor import BatchReactor
from chemreactor.reactors.cstr import CSTR
from chemreactor.reactors.pfr import PFR

# --- Fixtures for Reactor Tests ---

@pytest.fixture
def simple_reaction_details():
    """Provides details for a simple A -> B reaction."""
    return {
        'initial_conc': {'A': 1.0, 'B': 0.0},
        'reaction': Reaction(reactants={'A': 1}, products={'B': 1}, rate_constant=0.1),
        'k': 0.1
    }

# --- Test BatchReactor ---

@pytest.fixture
def batch_reactor_instance(simple_reaction_details):
    """Fixture to create a BatchReactor instance."""
    return BatchReactor(
        initial_concentrations=simple_reaction_details['initial_conc'],
        reactions=[simple_reaction_details['reaction']],
        volume=1.0
    )

def test_batch_init(batch_reactor_instance, simple_reaction_details):
    """Test BatchReactor initialization."""
    reactor = batch_reactor_instance
    assert reactor.concentrations == simple_reaction_details['initial_conc']
    assert len(reactor.reactions) == 1
    assert reactor.species_list == ['A', 'B'] # Order matters for ODEs
    assert reactor.time == 0.0

def test_batch_ode_system(batch_reactor_instance, simple_reaction_details):
    """Test the ODE system definition."""
    reactor = batch_reactor_instance
    reaction_details = simple_reaction_details
    # Test at initial conditions
    y0 = np.array([reactor.concentrations[species] for species in reactor.species_list])
    dydt = reactor._ode_system(0, y0)

    rate = reaction_details['reaction'].calculate_rate(reaction_details['initial_conc']) # 0.1 * 1.0 = 0.1
    expected_dAdt = -rate
    expected_dBdt = rate

    assert dydt[reactor.species_list.index('A')] == pytest.approx(expected_dAdt)
    assert dydt[reactor.species_list.index('B')] == pytest.approx(expected_dBdt)


def test_batch_run(batch_reactor_instance, simple_reaction_details):
    """Test running the batch reactor simulation."""
    reactor = batch_reactor_instance
    reaction_details = simple_reaction_details
    end_time = 10.0
    results = reactor.run(end_time)

    assert 'time' in results
    assert 'concentrations' in results
    assert results['time'][-1] == pytest.approx(end_time)
    assert len(results['time']) == len(results['concentrations'])

    # Check final concentrations (analytical solution for A -> B is C_A = C_A0 * exp(-kt))
    expected_A_final = reaction_details['initial_conc']['A'] * np.exp(-reaction_details['k'] * end_time)
    # Check conservation: A + B should equal initial A
    final_conc = results['concentrations'][-1]

    assert final_conc['A'] == pytest.approx(expected_A_final, abs=1e-5)
    assert final_conc['A'] + final_conc['B'] == pytest.approx(reaction_details['initial_conc']['A'], abs=1e-5)
    assert reactor.time == pytest.approx(end_time)
    assert reactor.concentrations == final_conc

# --- Test CSTR ---

@pytest.fixture
def cstr_reactor_instance(simple_reaction_details):
    """Fixture to create a CSTR instance."""
    details = simple_reaction_details
    return CSTR(
        inlet_concentrations=details['initial_conc'],
        reactions=[details['reaction']],
        volume=10.0, # m^3
        flow_rate=1.0 # m^3/s
        # Residence time tau = V/F = 10.0 s
    )

def test_cstr_init(cstr_reactor_instance, simple_reaction_details):
    """Test CSTR initialization."""
    reactor = cstr_reactor_instance
    assert reactor.inlet_concentrations == simple_reaction_details['initial_conc']
    assert reactor.concentrations == simple_reaction_details['initial_conc'] # Initial state matches inlet
    assert len(reactor.reactions) == 1
    assert reactor.species_list == ['A', 'B']
    assert reactor.residence_time == pytest.approx(reactor.volume / reactor.flow_rate)

def test_cstr_steady_state_equations(cstr_reactor_instance, simple_reaction_details):
    """Test the steady-state residual equations."""
    reactor = cstr_reactor_instance
    reaction = simple_reaction_details['reaction']
    inlet_conc = simple_reaction_details['initial_conc']

    # Test with some arbitrary concentration
    test_conc_arr = np.array([0.5, 0.5]) # A=0.5, B=0.5
    residuals = reactor._steady_state_equations(test_conc_arr)

    test_conc_dict = {'A': 0.5, 'B': 0.5}
    rate = reaction.calculate_rate(test_conc_dict) # 0.1 * 0.5 = 0.05
    net_rates = reactor.calculate_rates(test_conc_dict) # {'A': -0.05, 'B': 0.05}

    # Equation for A: F*(A_in - A) + r_A*V = 0
    expected_residual_A = reactor.flow_rate * (inlet_conc['A'] - test_conc_dict['A']) + net_rates['A'] * reactor.volume
    # Equation for B: F*(B_in - B) + r_B*V = 0
    expected_residual_B = reactor.flow_rate * (inlet_conc['B'] - test_conc_dict['B']) + net_rates['B'] * reactor.volume

    assert residuals[reactor.species_list.index('A')] == pytest.approx(expected_residual_A)
    assert residuals[reactor.species_list.index('B')] == pytest.approx(expected_residual_B)


def test_cstr_run_steady_state(cstr_reactor_instance, simple_reaction_details):
    """Test running the CSTR to find steady state."""
    reactor = cstr_reactor_instance
    reaction_details = simple_reaction_details
    results = reactor.run(end_time=0) # end_time is ignored for CSTR

    assert 'steady_state' in results
    ss_conc = results['steady_state']['concentrations']

    # Analytical solution for A -> B in CSTR: C_A = C_A0 / (1 + k*tau)
    tau = reactor.residence_time
    expected_A_ss = reaction_details['initial_conc']['A'] / (1 + reaction_details['k'] * tau)
    expected_B_ss = reaction_details['initial_conc']['A'] - expected_A_ss # From stoichiometry

    assert ss_conc['A'] == pytest.approx(expected_A_ss, abs=1e-5)
    assert ss_conc['B'] == pytest.approx(expected_B_ss, abs=1e-5)
    assert reactor.concentrations == ss_conc

# --- Test PFR ---

@pytest.fixture
def pfr_reactor_instance(simple_reaction_details):
    """Fixture to create a PFR instance."""
    details = simple_reaction_details
    return PFR(
        inlet_concentrations=details['initial_conc'],
        reactions=[details['reaction']],
        volume=10.0, # m^3
        flow_rate=1.0, # m^3/s
        diameter=1.0 # m
    )

def test_pfr_init(pfr_reactor_instance, simple_reaction_details):
    """Test PFR initialization."""
    reactor = pfr_reactor_instance
    assert reactor.inlet_concentrations == simple_reaction_details['initial_conc']
    assert reactor.concentrations == simple_reaction_details['initial_conc'] # Initial state matches inlet
    assert len(reactor.reactions) == 1
    assert reactor.species_list == ['A', 'B']
    expected_cross_section = np.pi * reactor.diameter**2 / 4.0
    expected_length = reactor.volume / expected_cross_section
    assert reactor.cross_section == pytest.approx(expected_cross_section)
    assert reactor.length == pytest.approx(expected_length)
    assert reactor.residence_time == pytest.approx(reactor.volume / reactor.flow_rate)


def test_pfr_steady_state_ode_equations(pfr_reactor_instance, simple_reaction_details):
    """Test the steady-state ODE system definition for PFR."""
    reactor = pfr_reactor_instance
    inlet_conc = simple_reaction_details['initial_conc']
    # Test at inlet conditions
    y0 = np.array([inlet_conc[species] for species in reactor.species_list])
    dydz = reactor._steady_state_ode_equations(0, y0)

    rates_inlet = reactor.calculate_rates(inlet_conc) # {'A': -0.1, 'B': 0.1}

    # Equation: dC/dz = r * A_c / F
    expected_dAdz = rates_inlet['A'] * reactor.cross_section / reactor.flow_rate
    expected_dBdz = rates_inlet['B'] * reactor.cross_section / reactor.flow_rate

    assert dydz[reactor.species_list.index('A')] == pytest.approx(expected_dAdz)
    assert dydz[reactor.species_list.index('B')] == pytest.approx(expected_dBdz)


def test_pfr_run(pfr_reactor_instance, simple_reaction_details):
    """Test running the PFR simulation."""
    reactor = pfr_reactor_instance
    reaction_details = simple_reaction_details
    results = reactor.run(end_time=0) # end_time is ignored for PFR

    assert 'position' in results
    assert 'concentrations' in results
    assert 'time' not in results # Time should be replaced by position
    assert results['position'][-1] == pytest.approx(reactor.length)
    assert len(results['position']) == len(results['concentrations'])

    # Check final concentrations (analytical solution for A -> B in PFR is C_A = C_A0 * exp(-k*tau))
    tau = reactor.residence_time
    expected_A_final = reaction_details['initial_conc']['A'] * np.exp(-reaction_details['k'] * tau)
    # Check conservation: A + B should equal initial A
    final_conc = results['concentrations'][-1]

    assert final_conc['A'] == pytest.approx(expected_A_final, abs=1e-5)
    assert final_conc['A'] + final_conc['B'] == pytest.approx(reaction_details['initial_conc']['A'], abs=1e-5)
    assert reactor.concentrations == final_conc
