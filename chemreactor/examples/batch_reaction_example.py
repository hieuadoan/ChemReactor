import numpy as np
import matplotlib.pyplot as plt
from chemreactor.core.base_reactor import Reaction
from chemreactor.reactors.batch_reactor import BatchReactor
from chemreactor.visualization.reactor_animation import ReactorAnimation

# Define a simple first-order reaction: A -> B
reaction1 = Reaction(reactants={'A':1.0},
                        products={'B': 1.0},
                        rate_constant=0.1  # k = 0.1 s^-1
                    )
# Define initial concentrations
initial_concentrations = {'A': 1.0, # mol/L
                            'B': 0.0  # mol/L
                            }

# Create a batch reactor
reactor = BatchReactor(initial_concentrations=initial_concentrations,
                        reactions=[reaction1],
                        volume=1.0, # 1 L
                        temperature=298.15 # 25 C
                        )

# Run the simulation
results = reactor.run(end_time=60.0, # 60 seconds
                      time_points=np.linspace(0, 60., 200) # 200 time points in seconds
                      )

# Create and display an animation
animator = ReactorAnimation(reactor)
animation = animator.create_batch_animation(results=results,
                                            species_colors={'A': 'red', 'B': 'blue'},
                                            n_particles=200)

plt.show()

# Optionally save the animation
#animator.save_animation('batch_reaction.mp4')