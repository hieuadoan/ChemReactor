from typing import Dict, List, Optional
from scipy.integrate import solve_ivp
import numpy as np
from chemreactor.core.base_reactor import BaseReactor, Reaction


class PFR(BaseReactor):
    """Class representing a plug flow reactor"""

    def __init__(
        self,
        inlet_concentrations: Dict[str, float],
        reactions: List[Reaction],
        diameter: float,
        flow_rate: float,
        volume: float = 1.0,
        temperature: float = 298.15,
        pressure: float = 101325.0,
    ):
        """
        Initialize a PFR

        Parameters:
        -----------
            inlet_concentrations : (Dict[str, float])
                Dictionary mapping species names to their initial concentrations
            reactions : List[Reaction]
                List of Reaction objects representing the reactions in the system
            diameter : float
                Diamater of the reactor in dm
            flow_rate : float
                Volumetric flow rate in L/s
            temperature : float
                Reactor temperature in K
            pressure : float
                Reactor pressure in Pa

        """
        super().__init__(
            inlet_concentrations, reactions, flow_rate, volume, temperature, pressure
        )
        self.diameter = diameter
        self.cross_section = (
            np.pi * diameter**2 / 4.0
        )  # Cross sectional area of the pfr in dm^2
        self.length = volume / self.cross_section
        self.flow_rate = flow_rate
        self.inlet_concentrations = inlet_concentrations

        # Create ordered list of species for ODE solver
        self.species_list = list(inlet_concentrations.keys())

    def _steady_state_ode_equations(self, z, y: np.ndarray) -> np.ndarray:
        """
        Steady-state ode equation for the PFR w.r.t to horizional position: dCdz = rates * cross_section / flow_rate

        Parameters:
        ----------
            z : float
                Current position
            y : np.ndarray
                Array of species concentrations

        Returns:
        -------
        np.ndarray : Array of derivatives
        """
        # Convert y array to concentrations dictionary
        concentrations = {species: y[i] for i, species in enumerate(self.species_list)}

        # Calculate rates
        rates = self.calculate_rates(concentrations)

        # Convert rates dictionary to derivatives array
        dydz = np.array(
            [
                rates[species] * self.cross_section / self.flow_rate
                for species in self.species_list
            ]
        )

        return dydz

    def run(self, end_time: float, time_points: Optional[np.ndarray] = None) -> Dict:
        """
        Run the plug flow reactor simulation

        For PFR we ingnored the time parameters and solve along the reactor length

        Returns:
        -------
            Dict : Simulation results
        """
        # Set up the position points
        position_points = np.linspace(0, self.length, 100)

        # Convert initial concentrations to array
        y0 = np.array([self.concentrations[species] for species in self.species_list])

        # Solve ODE system
        solution = solve_ivp(
            self._steady_state_ode_equations,
            (0, self.length),
            y0,
            method="RK45",
            t_eval=position_points,
            rtol=1e-6,
            atol=1e-8,
        )

        # Process results
        if solution.success:
            positions = solution.t
            concentrations = solution.y

            # Update history, replace tims with positions
            self.history["position"] = list(positions)
            self.history.pop("time", None)
            
            # Re-initialize the concentrations list before populating
            self.history['concentrations'] = []

            # Update concentration history
            for i, _ in enumerate(positions):
                conc_dict = {
                    species: concentrations[j, i]
                    for j, species in enumerate(self.species_list)
                }
                self.history["concentrations"].append(conc_dict)

            # Update current state
            self.concentrations = {
                species: concentrations[i, -1]
                for i, species in enumerate(self.species_list)
            }

        return self.history
