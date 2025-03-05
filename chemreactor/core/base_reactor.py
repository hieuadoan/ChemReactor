from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable

class Reaction:
	"""Class represeting a chemical reaaction"""

	def __init__(self,
				reactants: Dict[str, float],
				products: Dict[str, float],
				rate_constant: float,
				rate_equation: Optional[Callable] = None):			
		"""
		Initialize a chemical reaction

		Parameters:
		-----------
		reactants : Dict[str, float]
			Dictionary mapping reactant names to their stoichiometric coefficients
		products : Dict[str, float]
			Dictionary mapping product names to their stoichiometric coefficients
		rate_constant : float
			Rate constant for the reaction
		rate_equation : Optional[Callable]
			Custom rate equation function. If None, a standard mass rate law is applied
		"""
		self.reactants = reactants
		self.products = products
		self.k = rate_constant
		self.rate_equation = rate_equation

	def calculate_rate(self, concentration: Dict[str, float]) -> float:
		"""
		Calculate reaction rate based on concentrations

		Parameters:
		-----------
		concentration : Dict[str, float]
			Dictionary mapping species names to their concentrations
		
		Returns:
		-----------
		float : Reaction rate
		"""
		if self.rate_equation is not None:
			return self.rate_equation(self.k, concentrations)

		# Default mass action kinetics
		rate = self.k
		for species, coeff in self.reactants.items():
			if species in concentrations:
				rate *= concentrations[species] ** abs(coeff)

		return rate

class BaseReactor(ABC):
	"""Abstract base class for all reactor types"""

	def __init__(self, 
				 initial_concentrations: Dict[str, float],
				 reactions: List[Reaction],
				 volume: float = 1.0,
				 temperature: float = 298.15,
				 pressure: float = 101325.0):
		"""
		Initialize the reactor

		Parameters:
		-----------
		initial_concentration : Dict[str, float]
			Dictionary mapping species names to initial concentrations
		reactions : List[Reaction]
			List of Reaction objects representing the reactions in the system
		volume : float
			Reactor volume in m^3
		temperature : float
			Reactor temperature in K
		pressure : float
			Reactor pressure in Pa
		"""
		self.concentrations = initial_concentrations.copy()
		self.initial_concentrations = initial_concentrations.copy()
		self.reactions = reactions
		self.volume = volume
		self.temperature = temperature
		self.pressure = pressure
		self.time = 0.0
		self.history = {
			'time': [0.0],
			'concentrations': [initial_concentrations.copy()],
			'temperature': [temperature],
			'pressure': [pressure]
		}

	@abstractmethod
	def run(self,
			end_time: float,
			time_points: Optional[np.ndarray] = None) -> Dict:
		"""
		Run the reactor simulation

		Parameters:
		-----------
		end_time : float
			End time for the simulation
		time_points : Optional[np.ndarray]
			Array of time points at which to record results

		Returns:
		-----------
		Dict : Simulation results
		"""
		pass

	def calculate_rates(self,
						concentrations: Dict[str, float]) -> Dict[str, float]:
		"""
		Calculate net production/consumption rates for all species

		Parameters:
		-----------
		concentrations : Dict[str, float]
			Dictionary mapping species names to concentrations
		
		Returns:
		-----------
		Dict[str, float] : Net rates for all species
		"""
		# Initialize rates dictionary with zeros
		net_rates = {species: 0.0 for species in concentrations}
		
		# Calculate contribution from each reaction
		for reactions in self.reactions:
			rxn_rate = reaction.calculate_rate(concentrations) 	

			# Update rates for reactants (consumption)
			for species, coeff in reaction.reactants.items():
				net_rates[species] -= coeff * rxn_rate

			# Update rates for products (production)
			for species, coeff in reaction.products.items():
				net_rates[species] += coeff * rxn_rate

		return net_rates

	def reset(self):
		"""Reset the eractor to initial conditions"""
		self.concentrations = self.initial_concentrations.copy()
		self.time = 0.0
		self.history = {
			'time': [0.0],
			'concentrations': [self.initial_concentrations.copy()],
			'temperature': [self.temperature],
			'pressure': [self.pressure]
		}


	def get_results(self) -> Dict:
		"""Return simulation results"""
		return self.history
