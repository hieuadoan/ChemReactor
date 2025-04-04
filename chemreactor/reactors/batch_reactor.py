from typing import Dict, List, Optional
from scipy.integrate import solve_ivp
import numpy as np
from chemreactor.core.base_reactor import BaseReactor, Reaction

class BatchReactor(BaseReactor):
    """Class representing a batch reactor"""
    
    def __init__(self,
                 initial_concentrations: Dict[str, float],
                 reactions: List[Reaction],
                 volume: float = 1.0,
                 temperature: float = 298.15,
                 pressure: float = 101325.0):
        
        """
        Initialize a batch reactor
        
        Parameters:
        -----------
            initial_concentrations : (Dict[str, float])
                Dictionary mapping species names to their initial concentrations
            reactions : List[Reaction]
                List of Reaction objects representing the reactions in the system
            volume : float
                Reactor volume in m^3
            temperature : float
                Reactor temperature in K
            pressure : float
                Reactor pressure in Pa
        """
        super().__init__(initial_concentrations, reactions, volume, temperature, pressure)
        
        # Create ordered list of species for ODE solver
        self.species_list = list(initial_concentrations.keys())
        
    def _ode_system(self,
                    t,
                    y: np.ndarray) -> np.ndarray:
        """
        ODE system for the batch reactor
        
        Parameters:
        ----------
            t : float
                Current time
            y : np.ndarray
                Array of species concentrations
                
        Returns:
        -------
        np.ndarray : Array of derivatives
        """
        # Convert y array to concentrations dictionary
        concentrations = {species: y[i] for i, species in enumerate(self.species_list)}
        
        # Calculate rates
        rates  = self.calculate_rates(concentrations)
        
        # Convert rates dictionary to derivatives array
        dydt = np.array([rates[species] for species in self.species_list])
        
        return dydt
    
    def run(self,
            end_time: float,
            time_points: Optional[np.ndarray] = None) -> Dict:
        """
        Run the batch reactor simulation
        
        Parameters:
        ----------
            end_time : float
                End time for the simulation in seconds
            time_points : Optional[np.ndarray]
                Array of time points at which to record results
        
        Returns:
        -------
            Dict : Simulation results
        """
        # Set up time points if not provided, in seconds
        if time_points is None:
            time_points = np.linspace(0, end_time, 100)
        
        # Ensure time_points includes the end time
        #if time_points[-1] != end_time:
        #    time_points = np.append(time_points, end_time)
        
        # Convert initial concentrations to array
        y0 = np.array([self.concentrations[species] for species in self.species_list])
        
        # Solve ODE system
        solution = solve_ivp(self._ode_system,
                             (0, end_time),
                             y0,
                             method='RK45',
                             t_eval=time_points,
                             rtol=1e-6,
                             atol=1e-8)
        
        # Process results
        if solution.success:
            times = solution.t
            concentrations = solution.y
            
            # Upgrade history
            self.history['time'] = list(times)
           	
            # Re-initialize the concentration list before populating
            self.history['concentrations'] = []
 
            # Update concentration history
            for i, _ in enumerate(times):
                conc_dict = {species: concentrations[j, i]
                             for j, species in enumerate(self.species_list)}
                self.history['concentrations'].append(conc_dict)
                
            # Update current state
            self.time = end_time
            self.concentrations = {species: concentrations[i, -1]
                                   for i, species in enumerate(self.species_list)}
            
        return self.history
