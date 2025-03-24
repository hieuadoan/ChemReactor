import matplotlib.pyplot as plt
from chemreactor.core.base_reactor import Reaction
from chemreactor.reactors.cstr import CSTR
from chemreactor.visualization.reactor_animation import ReactorAnimation

# Define a simple first-order reaction: A -> B
reaction1 = Reaction(
    reactants={"A": 1.0},
    products={"B": 1.0},
    rate_constant=2.0,  # k = 0.1 s^-1
)
# Define initial concentrations
initial_concentrations = {
    "A": 1.0,  # mol/L
    "B": 0.0,  # mol/L
}

# Create a CSTR
reactor = CSTR(
    inlet_concentrations=initial_concentrations,
    reactions=[reaction1],
    volume=1.0,  # 1 L
    temperature=298.15,  # 25 C
    flow_rate=1.0,  # L/s
)

# Run the simulation
results = reactor.run(end_time=1)

# Create custom colors for species
species_colors = {"A": "red", "B": "blue"}

# Create animation
animator = ReactorAnimation(reactor)
animation = animator.create_cstr_animation(
    results=results,
    species_colors=species_colors,
    n_particles=300,
    animation_duration=5.0,
    n_frames=100,
)

# Display the animation (in Jupyter notebook, you would use IPython.display.HTML)
plt.show()

# Save the animation if needed
# animator.save_animation("cstr_animation.mp4", fps=30)
