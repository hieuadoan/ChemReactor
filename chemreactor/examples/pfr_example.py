import matplotlib.pyplot as plt
from chemreactor.core.base_reactor import Reaction
from chemreactor.reactors.pfr import PFR
from chemreactor.visualization.reactor_animation import ReactorAnimation

# Define a simple first-order reaction: A -> B
reaction1 = Reaction(
    reactants={"A": 1.0},
    products={"B": 1.0},
    rate_constant=0.1,  # k = 0.1 s^-1
)
# Define initial concentrations
initial_concentrations = {
    "A": 1.0,  # mol/L
    "B": 0.0,  # mol/L
}

# Create a batch reactor
reactor = PFR(
    inlet_concentrations=initial_concentrations,
    reactions=[reaction1],
    diameter=0.5,  # dm
    flow_rate=0.1,
    volume=1.0,  # 1 L
    temperature=298.15,  # 25 C
)

# Run the simulation
results = reactor.run(end_time=1)
# print(results["concentrations"][-1])

# Create and display an animation
animator = ReactorAnimation(reactor)
animation = animator.create_pfr_animation(
    results=results,
    species_colors={"A": "red", "B": "blue"},
)
plt.show()


# Optionally save the animation
# animator.save_animation('batch_reaction.mp4')
