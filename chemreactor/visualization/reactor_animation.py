from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
from chemreactor.core.base_reactor import BaseReactor


class ReactorAnimation:
    """Class for creating reactor visualization and animations"""

    def __init__(self, reactor: BaseReactor):
        """
        Initialize a reactor animation

        Parameters:
        ----------
            reactor : BaseReactor
                Reactor object to visualize
        """
        self.reactor = reactor
        self.fig = None
        self.animation = None

    def create_batch_animation(
        self,
        results: Dict,
        species_colors: Optional[Dict[str, str]] = None,
        n_particles: int = 200,
    ) -> FuncAnimation:
        """
        Create a batch reactor animation

        Parameters:
        ----------
            results : Dict
                Simulation results from reactor.run()
            species_colors : Optional[Dict[str, str]]
                Dictionary mapping species names to colors
            n_particles : int
                Number of particles to show in animation

        Returns:
        -------
            FuncAnimation : Matplotlib animation object
        """
        # Extract data
        times = results["time"]
        conc_history = results["concentrations"]

        # Get species names
        species_names = list(conc_history[0].keys())

        # Set default colors if not provided
        if species_colors is None:
            default_colors = ["red", "green", "blue", "purple", "orange", "cyan"]
            species_colors = {
                species: default_colors[i % len(default_colors)]
                for i, species in enumerate(species_names)
            }

        # Create figure and axes
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        plt.subplots_adjust(wspace=0.3)

        # Set up concentration plot
        lines = {}
        for species in species_names:
            (line,) = ax1.plot([], [], color=species_colors[species], label=species)
            lines[species] = line

        ax1.set_xlim(0, times[-1])

        # Find max concentration for y-limit
        max_conc = 0
        for conc_dict in conc_history:
            for _, conc in conc_dict.items():
                max_conc = max(max_conc, conc)

        ax1.set_ylim(0, max_conc * 1.15)
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Concentration (M)")
        ax1.set_title("Species concentrations")
        ax1.grid(True)
        ax1.legend()

        # Set up reactor visualization
        container = Rectangle((0, 0), 1, 1, fc="lightgray", ec="black")
        ax2.add_patch(container)

        # Create particles
        particles = {species: [] for species in species_names}
        positions = np.random.rand(n_particles, 2)  # Shape (n_particles, 2)

        for species in species_names:
            for i in range(n_particles):
                particle = Circle(
                    (positions[i, 0], positions[i, 1]),
                    0.015,
                    fc=species_colors[species],
                    alpha=0,
                )
                particles[species].append(particle)
                ax2.add_patch(particle)

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_aspect("equal")
        ax2.set_title("Batch Reactor Visualization")
        ax2.axis("off")

        # Create legend elements
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=species_colors[species],
                markersize=10,
                label=species,
            )
            for species in species_names
        ]

        # Create a floating legend with transparent background and place it in top-right corener
        # Use framealpha to control the background transparency of the legend box
        ax2.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            title="Species",
        )

        # Text for displaying time
        time_text = ax2.text(
            0.05,
            0.95,
            "",
            transform=ax2.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3),
            zorder=1000,
        )  # Ensure time text is also above particles

        # Animation initializaiton function
        def init():
            for species, line in lines.items():
                line.set_data([], [])

            time_text.set_text("")

            # Initialize all particles as invisible
            for species in species_names:
                for particle in particles[species]:
                    particle.set_alpha(0)

            return (
                [line for line in lines.values()]
                + [time_text]
                + sum([particles[s] for s in species_names], [])
            )

        # Animation update function
        def update(frame):
            # Update concentration plot
            for species, line in lines.items():
                conc_values = [conc_history[i][species] for i in range(frame + 1)]
                line.set_data(times[: frame + 1], conc_values)

            # Update current time display
            time_text.set_text(f"Time: {times[frame]: .2f}")

            # Get current concentrations
            current_conc = conc_history[frame]
            total_conc = sum(current_conc.values())

            # Update particles
            particle_counts = {}
            for species in species_names:
                # Calculate how many particles to show based on concentration fraction
                if total_conc > 0:
                    fraction = current_conc[species] / total_conc
                    particle_counts[species] = int(n_particles * fraction)
                else:
                    particle_counts[species] = 0

            # Show/hide particles based on calculated counts
            used_particles = 0
            for species in species_names:
                for i, particle in enumerate(particles[species]):
                    if i < particle_counts[species]:
                        particle.set_alpha(1)  # Show particle
                        # Position in the next available spot
                        particle.center = positions[used_particles]
                        used_particles += 1
                    else:
                        particle.set_alpha(0)  # Hide particle

            return (
                [line for line in lines.values()]
                + [time_text]
                + sum([particles[s] for s in species_names], [])
            )

        # Create and return animation
        self.animation = FuncAnimation(
            self.fig, update, frames=len(times), init_func=init, interval=50, blit=True
        )

        plt.suptitle("Batch Reactor Simulation", fontsize=14)
        plt.tight_layout()

        return self.animation

    def create_cstr_animation(
        self,
        results: Dict,
        species_colors: Optional[Dict[str, str]] = None,
        n_particles: int = 200,
        animation_duration: float = 10.0,
        n_frames: int = 100,
    ) -> FuncAnimation:
        """
        Create a CSTR animation for steady-state results

        Parameters:
        ----------
            results : Dict
                Steady-state results from CSTR.run()
            species_colors : Optional[Dict[str, str]]
                Dictionary mapping species names to colors
            n_particles : int
                Number of particles to show in animation
            animation_duration : float
                Duration of the animation in seconds
            n_frames : int
                Number of frames to generate for the animation

        Returns:
        -------
            FuncAnimation : Matplotlib animation object
        """
        # Extract steady-state data
        if "steady_state" not in results:
            raise ValueError(
                "Results do not contain steady-state data. Make sure you're using a CSTR reactor."
            )

        steady_state_conc = results["steady_state"]["concentrations"]

        # Get species names
        species_names = list(steady_state_conc.keys())

        # Set default colors if not provided
        if species_colors is None:
            default_colors = ["red", "green", "blue", "purple", "orange", "cyan"]
            species_colors = {
                species: default_colors[i % len(default_colors)]
                for i, species in enumerate(species_names)
            }

        # Create figure and axes
        self.fig, ax = plt.subplots(figsize=(10, 8))

        # Create CSTR visualization with inlet and outlet
        # Main tank
        tank = Rectangle((0.2, 0.2), 0.6, 0.6, fc="lightgray", ec="black")
        ax.add_patch(tank)

        # Inlet pipe
        inlet = Rectangle((0.05, 0.5), 0.15, 0.1, fc="lightgray", ec="black")
        ax.add_patch(inlet)

        # Outlet pipe
        outlet = Rectangle((0.8, 0.5), 0.15, 0.1, fc="lightgray", ec="black")
        ax.add_patch(outlet)

        # Arrows for flow direction
        ax.arrow(
            0.05,
            0.55,
            0.1,
            0,
            head_width=0.05,
            head_length=0.05,
            fc="black",
            ec="black",
        )
        ax.arrow(
            0.8, 0.55, 0.1, 0, head_width=0.05, head_length=0.05, fc="black", ec="black"
        )

        # Text labels
        ax.text(0.1, 0.65, "Inlet", fontsize=12, ha="center")
        ax.text(0.9, 0.65, "Outlet", fontsize=12, ha="center")
        ax.text(0.5, 0.85, "CSTR", fontsize=14, ha="center", weight="bold")

        # Create particles
        # We need two sets: one for the inlet and one for the reactor
        inlet_particles = {species: [] for species in species_names}
        reactor_particles = {species: [] for species in species_names}
        outlet_particles = {species: [] for species in species_names}

        # Calculate relative concentration
        total_conc = sum(steady_state_conc.values())

        # For inlet, use the reactor.inlet_concentrations
        inlet_total_conc = sum(self.reactor.initial_concentrations.values())

        # Create random positions for particles
        # Inlet area (smaller rectangle on left)
        inlet_positions = np.random.uniform(
            low=[0.05, 0.5], high=[0.2, 0.6], size=(n_particles // 5, 2)
        )

        # Reactor area (main tank)
        reactor_positions = np.random.uniform(
            low=[0.2, 0.2], high=[0.8, 0.8], size=(n_particles, 2)
        )

        # Outlet area (smaller rectangle on right)
        outlet_positions = np.random.uniform(
            low=[0.8, 0.5], high=[0.95, 0.6], size=(n_particles // 5, 2)
        )

        # Create particles for each species
        for species in species_names:
            # Inlet particles
            inlet_fraction = (
                self.reactor.initial_concentrations.get(species, 0) / inlet_total_conc
                if inlet_total_conc > 0
                else 0
            )
            n_inlet_particles = int((n_particles // 5) * inlet_fraction)

            for _ in range(n_inlet_particles):
                particle = Circle((0, 0), 0.01, fc=species_colors[species], alpha=0.8)
                inlet_particles[species].append(particle)
                ax.add_patch(particle)

            # Reactor particles
            reactor_fraction = (
                steady_state_conc.get(species, 0) / total_conc if total_conc > 0 else 0
            )
            n_reactor_particles = int(n_particles * reactor_fraction)

            for _ in range(n_reactor_particles):
                particle = Circle((0, 0), 0.01, fc=species_colors[species], alpha=0.8)
                reactor_particles[species].append(particle)
                ax.add_patch(particle)

            # Outlet particles (same distribution as reactor)
            n_outlet_particles = int((n_particles // 5) * reactor_fraction)

            for _ in range(n_outlet_particles):
                particle = Circle((0, 0), 0.01, fc=species_colors[species], alpha=0.8)
                outlet_particles[species].append(particle)
                ax.add_patch(particle)

        # Set up the plot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("Continuous Stirred-Tank Reactor (CSTR) - Steady State")
        ax.axis("off")

        # Create legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=species_colors[species],
                markersize=10,
                label=species,
            )
            for species in species_names
        ]

        ax.legend(handles=legend_elements, loc="upper right", title="Species")

        # Create text for concentration display
        conc_text = ax.text(
            0.5,
            0.1,
            "",
            ha="center",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3),
        )

        # Create text to show residence time
        res_time_text = ax.text(
            0.5,
            0.05,
            f"Residence time: {self.reactor.residence_time:.2f} s",
            ha="center",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3),
        )

        # Generate artificial time points for animation
        time_points = np.linspace(0, animation_duration, n_frames)

        # Animation initialization function
        def init():
            conc_text.set_text("")

            # Initialize particle positions
            particle_index = 0
            all_patches = []

            # Inlet particles
            for species in species_names:
                for i, particle in enumerate(inlet_particles[species]):
                    if particle_index < len(inlet_positions):
                        particle.center = inlet_positions[particle_index]
                        particle_index += 1
                    all_patches.append(particle)

            # Reset particle index for reactor particles
            particle_index = 0

            # Reactor particles
            for species in species_names:
                for i, particle in enumerate(reactor_particles[species]):
                    if particle_index < len(reactor_positions):
                        particle.center = reactor_positions[particle_index]
                        particle_index += 1
                    all_patches.append(particle)

            # Reset particle index for outlet particles
            particle_index = 0

            # Outlet particles
            for species in species_names:
                for i, particle in enumerate(outlet_particles[species]):
                    if particle_index < len(outlet_positions):
                        particle.center = outlet_positions[particle_index]
                        particle_index += 1
                    all_patches.append(particle)

            return [conc_text, res_time_text] + all_patches

        # Animation update function
        def update(frame):
            # Update concentration text
            conc_str = "Steady-state concentrations:\n"
            for species in species_names:
                conc_str += f"{species}: {steady_state_conc[species]:.4f} M\n"
            conc_text.set_text(conc_str)

            # Update particle positions - make them move randomly for visual effect
            all_patches = []

            # Randomly move inlet particles
            particle_index = 0
            for species in species_names:
                for particle in inlet_particles[species]:
                    # Small random movement
                    if particle_index < len(inlet_positions):
                        jitter = np.random.normal(0, 0.01, 2)
                        new_pos = inlet_positions[particle_index] + jitter

                        # Keep particles within inlet area
                        new_pos[0] = min(max(new_pos[0], 0.05), 0.2)
                        new_pos[1] = min(max(new_pos[1], 0.5), 0.6)

                        particle.center = new_pos
                        particle_index += 1
                    all_patches.append(particle)

            # Randomly move reactor particles
            particle_index = 0
            for species in species_names:
                for particle in reactor_particles[species]:
                    # Small random movement
                    if particle_index < len(reactor_positions):
                        jitter = np.random.normal(
                            0, 0.02, 2
                        )  # More movement in the reactor
                        new_pos = reactor_positions[particle_index] + jitter

                        # Keep particles within reactor area
                        new_pos[0] = min(max(new_pos[0], 0.2), 0.8)
                        new_pos[1] = min(max(new_pos[1], 0.2), 0.8)

                        particle.center = new_pos
                        particle_index += 1
                    all_patches.append(particle)

            # Randomly move outlet particles
            particle_index = 0
            for species in species_names:
                for particle in outlet_particles[species]:
                    # Small random movement
                    if particle_index < len(outlet_positions):
                        jitter = np.random.normal(0, 0.01, 2)
                        new_pos = outlet_positions[particle_index] + jitter

                        # Keep particles within outlet area
                        new_pos[0] = min(max(new_pos[0], 0.8), 0.95)
                        new_pos[1] = min(max(new_pos[1], 0.5), 0.6)

                        particle.center = new_pos
                        particle_index += 1
                    all_patches.append(particle)

            return [conc_text, res_time_text] + all_patches

        # Create and return animation
        self.animation = FuncAnimation(
            self.fig,
            update,
            frames=len(time_points),
            init_func=init,
            interval=50,
            blit=True,
        )

        plt.tight_layout()

        return self.animation

    def create_pfr_animation(
        self,
        results: Dict,
        species_colors: Optional[Dict[str, str]] = None,
        n_particles_per_species: int = 200,
    ) -> FuncAnimation:
        """
        Create a batch reactor animation

        Parameters:
        ----------
            results : Dict
                Simulation results from reactor.run()
            species_colors : Optional[Dict[str, str]]
                Dictionary mapping species names to colors
            n_particles : int
                Number of particles to show in animation

        Returns:
        -------
            FuncAnimation : Matplotlib animation object
        """
        # Extract data
        positions = results["position"]
        conc_history = results["concentrations"]

        # Get species names
        species_names = list(conc_history[0].keys())

        # Set default colors if not provided
        if species_colors is None:
            default_colors = ["red", "green", "blue", "purple", "orange", "cyan"]
            species_colors = {
                species: default_colors[i % len(default_colors)]
                for i, species in enumerate(species_names)
            }

        # Create figure and axes
        self.fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
        )
        plt.subplots_adjust(hspace=0.3)

        # Set up concentration plot
        lines = {}
        for species in species_names:
            (line,) = ax1.plot([], [], color=species_colors[species], label=species)
            lines[species] = line

        ax1.set_xlim(0, positions[-1])

        # Find max concentration for y-limit
        max_conc = 0
        for conc_dict in conc_history:
            for _, conc in conc_dict.items():
                max_conc = max(max_conc, conc)

        ax1.set_ylim(0, max_conc * 1.15)
        ax1.set_xlabel("Position (dm)")
        ax1.set_ylabel("Concentration (M)")
        ax1.set_title("Species concentrations")
        ax1.grid(True)
        ax1.legend()

        # Position text for the left subplot
        position_text = ax1.text(
            0.05,
            0.95,
            "",
            transform=ax1.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3),
        )

        # Setup PFR visualization in right subplot
        # Draw the reactor tube with the same length as the concentration plot
        ax2.set_xlim(0, positions[-1])
        ax2.set_ylim(0, 1)

        # Draw the reactor tube with the same x-axis scale
        tube_height = 0.6
        tube_y_pos = 0.2
        tube = Rectangle(
            (0.0, tube_y_pos),
            positions[-1],
            tube_height,
            fc="lightgray",
            ec="black",
            alpha=0.3,
        )
        ax2.add_patch(tube)

        # Add inlet and outlet labels
        ax2.text(0.0, 0.5, "Inlet", fontsize=12, ha="right", va="center")
        ax2.text(positions[-1], 0.5, "Outlet", fontsize=12, ha="left", va="center")

        # Add markers for position reference (match concentration plot x-axis)
        num_markers = min(10, len(positions))
        marker_positions = np.linspace(0, positions[-1], num_markers)
        for pos in marker_positions:
            ax2.axvline(x=pos, color="gray", linestyle="--", alpha=0.3)

        # Create particles for visualization
        particles = {}

        for species in species_names:
            particles[species] = []
            for i in range(n_particles_per_species):
                particle = Circle((0, 0), 0.03, fc=species_colors[species], alpha=0)
                particles[species].append(particle)
                ax2.add_patch(particle)

        # Set ax2 properties
        ax2.set_xlabel("Position (dm)")
        ax2.set_title("Reactor Flow Visualization")
        ax2.set_yticks([])

        # Add legend to second subplot
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=species,
            )
            for species, color in species_colors.items()
        ]
        ax2.legend(handles=legend_elements, loc="upper right")

        # Current position line (vertical line that moves with animation)
        current_pos_line = ax1.axvline(x=0, color="black", linestyle="-", alpha=0.5)
        current_pos_line2 = ax2.axvline(x=0, color="black", linestyle="-", alpha=0.5)

        # Animation initialization function
        def init():
            for line in lines.values():
                line.set_data([], [])
            position_text.set_text("")
            current_pos_line.set_xdata([0])
            current_pos_line2.set_xdata([0])

            # Initialize all particles as invisible
            for species in species_names:
                for particle in particles[species]:
                    particle.set_alpha(0)

            return (
                [line for line in lines.values()]
                + [position_text]
                + sum([particles[s] for s in species_names], [])
            )

        # Animation update function

        def update(frame):
            current_pos = positions[frame]

            # Update concentration plot
            for species, line in lines.items():
                # Show full concentration profiles
                conc_values = [conc_history[i][species] for i in range(len(positions))]
                line.set_data(positions, conc_values)

            # Update current position display and vertical line
            position_text.set_text(f"Position: {current_pos:.2f} dm")
            current_pos_line.set_xdata([current_pos])
            current_pos_line2.set_xdata([current_pos])

            # Clear previous particles by setting alpha to 0
            for species in species_names:
                for particle in particles[species]:
                    particle.set_alpha(0)

            # Create more sample points to ensure smooth distribution
            num_samples = 100  # Increase for smoother distribution
            sample_points = np.linspace(0, current_pos, num_samples)

            # For each position up to the current position, display particles based on local concentration
            for species in species_names:
                particles_used = 0

                # Calculate total particles to ensure we use all available particles
                total_allowed_particles = min(
                    n_particles_per_species,
                    int(frame / len(positions) * n_particles_per_species * 1.5),
                )

                # For each position sample point
                for pos_idx, pos in enumerate(sample_points):
                    # Find the closest position in our data
                    closest_idx = min(
                        frame, len(positions) - 1
                    )  # Ensure we don't go beyond our data
                    frame_fraction = pos / current_pos if current_pos > 0 else 0
                    interp_idx = int(frame_fraction * closest_idx)
                    actual_idx = min(interp_idx, len(positions) - 1)

                    # Get concentration at this position
                    local_conc = conc_history[actual_idx][species]
                    # Calculate relative concentration
                    rel_conc = local_conc / max_conc if max_conc > 0 else 0

                    # Number of particles per position is proportional to the position fraction
                    particles_per_pos = max(
                        1, int(total_allowed_particles / num_samples)
                    )

                    # Adjust based on concentration
                    local_n_particles = int(particles_per_pos * rel_conc * 2)

                    # Place particles at this position
                    for i in range(local_n_particles):
                        if particles_used < n_particles_per_species:
                            particle = particles[species][particles_used]

                            # Position with small random offset
                            x_jitter = (positions[-1] / 100) * np.random.uniform(-1, 1)
                            x_pos = pos + x_jitter
                            x_pos = max(
                                0, min(current_pos, x_pos)
                            )  # Keep within bounds

                            # Random y position within tube
                            y_pos = tube_y_pos + np.random.random() * tube_height

                            # Set particle properties
                            particle.center = (x_pos, y_pos)
                            particle.radius = (
                                0.015 + 0.015 * rel_conc
                            )  # Size proportional to concentration
                            particle.set_alpha(
                                min(0.9, 0.4 + 0.5 * rel_conc)
                            )  # Opacity also varies with concentration

                            particles_used += 1

            return (
                [line for line in lines.values()]
                + [position_text, current_pos_line, current_pos_line2]
                + sum([particles[s] for s in species_names], [])
            )

        # Create and return animation
        self.animation = FuncAnimation(
            self.fig,
            update,
            frames=len(positions),
            init_func=init,
            interval=40,
            blit=True,
        )

        plt.suptitle("Plug Flow Simulation", fontsize=14)
        plt.tight_layout()

        return self.animation

    def save_animation(self, filename: str, fps: int = 30):
        """
        Save the animation to a file

        Parameters:
        ----------
        filename : str
            Filename to save to (should end with .mp4, .gif, etc.)
        fps : int
            Frames per second
        """
        if self.animation is not None:
            self.animation.save(filename, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("No animation has been created yet.")
