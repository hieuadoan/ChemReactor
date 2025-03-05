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
	reactants: Dict[str, float]
		Dictionary mapping reactant names to their stoichiometric coefficients
	products: Dict[str, float]
		Dictionary mapping product names to their stoichiometric coefficients
	rate_constant: float
		Rate constant for the reaction
	rate_equation: Optional[Callable]
		Custom rate equation function. If None, a standard mass rate law is applied
	"""
	self.reactants = reactants
	self.products = products
	self.k = rate_constant
	self.rate_equation = rate_equation

	def calculate_rate(self, concentration: Dict[str, float] -> float:
	"""
	Calculate reaction rate based on concentrations

	Parameters:
	-----------
	concentration: Dict[str, float]
		Dictionary mapping species names to their concentrations
	
	Returns:
	-----------
	float: Reaction rate
	"""
	if self.rate_equation is not None:
		return self.rate_equation(self.k, concentrations)

	# Default mass action kinetics
	rate = self.k
	for species, coeff in self.reactants.items():
		if species in concentrations:
			rate *= concentrations[species] ** abs(coeff)

	return rate
