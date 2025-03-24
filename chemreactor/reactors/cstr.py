from typing import Dict, List, Optional
from scipy.optimize import fsolve
import numpy as np
from chemreactor.core.base_reactor import BaseReactor, Reaction


class CSTR(BaseReactor):
    """Class representing a continuous stirred-tank reactor"""

    def __init__(
        self,
        inlet_concentrations: Dict[str, float],
        reactions: List[Reaction],
        flow_rate: float,
        volume: float = 1.0,
        temperature: float = 298.15,
        pressure: float = 101325.0,
    ):
        """
        Initialize a CSTR

        Parameters:
        -----------
            inlet_concentrations : (Dict[str, float])
                Dictionary mapping species names to their initial concentrations
            reactions : List[Reaction]
                List of Reaction objects representing the reactions in the system
            volume : float
                Reactor volume in L
            temperature : float
                Reactor temperature in K
            pressure : float
                Reactor pressure in Pa
            flow_rate : float
                Volumetric flow rate in L

        """
        super().__init__(inlet_concentrations, reactions, flow_rate, volume, temperature, pressure)

        self.inlet_concentrations = inlet_concentrations

        # Create ordered list of species for ODE solver
        self.species_list = list(inlet_concentrations.keys())

    def _steady_state_equations(self, concentrations_array: np.ndarray) -> np.ndarray:
        """
        Steady-state equations for the CSTR: flow_rate * (Cin - C) + rV = 0

        Parameters:
        ----------
        concentrations_array : np.ndarray
            Array of species concentrations

        Returns:
        -------
        np.ndarray : Residuals for Steady-state equations
        """
        # Convert concentrations array to Dictionary
        concentrations = {
            species: concentrations_array[i]
            for i, species in enumerate(self.species_list)
        }

        # Calcualte reaction rate
        rates = self.calculate_rates(concentrations)

        # Calculate residuals
        residuals = []
        for species in self.species_list:
            flow_term = self.flow_rate * (
                self.inlet_concentrations[species] - concentrations[species]
            )
            residual = flow_term + rates[species] * self.volume
            residuals.append(residual)

        return np.array(residuals)

    def run(self, end_time: float, time_points: Optional[np.ndarray] = None) -> Dict:
        """
        Run the CSTR simulation

        Parameters are ignored for steady-state simulation

        Returns:
        -------
            Dict : Steady-state results
        """
        # Use inlet concentrations as initial guess
        initial_guess = np.array(
            [self.inlet_concentrations[species] for species in self.species_list]
        )

        # Solve for steady state
        solution = fsolve(self._steady_state_equations, initial_guess)
        # Convert solution array to concentrations dictionary
        steady_state_conc = {
            species: solution[i] for i, species in enumerate(self.species_list)
        }
        # Update steady-state values
        self.concentrations = steady_state_conc
        # Update history
        self.history = {
            "steady_state": {
                "concentrations": steady_state_conc,
                "temperature": self.temperature,
                "pressure": self.pressure,
            }
        }
        return self.history
