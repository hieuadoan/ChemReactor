import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import base64

from chemreactor.core.base_reactor import Reaction
from chemreactor.reactors.cstr import CSTR
from chemreactor.reactors.batch_reactor import BatchReactor
from chemreactor.reactors.pfr import PFR
from chemreactor.visualization.reactor_animation import ReactorAnimation


# Function to convert animation to GIF for Streamlit display
def animation_to_gif(animation, fps=10):
    """
    Convert a matplotlib animation to a GIF for displaying in Streamlit

    Parameters:
    ----------
    animation : FuncAnimation
        Matplotlib animation object
    fps : int
        Frames per second for the GIF

    Returns:
    -------
    bytes : GIF data as bytes
    """
    # Create a temporary file path
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp:
        temp_path = temp.name

    # Save animation to the temporary file
    animation.save(temp_path, writer="pillow", fps=fps)

    # Read the file content
    with open(temp_path, "rb") as file:
        gif_data = file.read()

    # Clean up
    os.unlink(temp_path)

    return gif_data


def display_gif(gif_data):
    """Display an animated GIF using HTML"""
    b64 = base64.b64encode(gif_data).decode()
    html = f'<img src="data:image/gif;base64,{b64}" alt="reactor animation" style="max-width: 100%;">'
    st.markdown(html, unsafe_allow_html=True)


# Cache the reactor simulations to improve performance
@st.cache_data
def run_batch_reactor_calc(k_value, initial_conc_A, initial_conc_B, end_time, volume):
    """
    Run batch reactor simulation and return only the results, not the reactor
    """
    # Define reaction
    reaction = Reaction(
        reactants={"A": 1.0}, products={"B": 1.0}, rate_constant=k_value
    )

    # Define initial concentrations
    initial_concentrations = {"A": initial_conc_A, "B": initial_conc_B}

    # Create and run reactor
    reactor = BatchReactor(
        initial_concentrations=initial_concentrations,
        reactions=[reaction],
        volume=volume,
        temperature=298.15,
    )

    time_points = np.linspace(0, end_time, 100)
    results = reactor.run(end_time, time_points)

    return reactor, results


def run_batch_reactor(k_value, initial_conc_A, initial_conc_B, end_time, volume):
    """
    Wrapper function to get the reactor and results from cache calculations
    """
    # Simply call the cached function and return its results
    return run_batch_reactor_calc(
        k_value, initial_conc_A, initial_conc_B, end_time, volume
    )


@st.cache_data
def run_cstr_calc(k_value, inlet_conc_A, inlet_conc_B, flow_rate, volume):
    """
    Run CSTR simulation and return only the results, not the reactor
    """
    # Define reaction
    reaction = Reaction(
        reactants={"A": 1.0}, products={"B": 1.0}, rate_constant=k_value
    )

    # Define inlet concentrations
    inlet_concentrations = {"A": inlet_conc_A, "B": inlet_conc_B}

    # Create and run reactor
    reactor = CSTR(
        inlet_concentrations=inlet_concentrations,
        reactions=[reaction],
        flow_rate=flow_rate,
        volume=volume,
        temperature=298.15,
    )

    results = reactor.run(end_time=1)  # end_time is not used for CSTR

    return reactor, results


def run_cstr(k_value, inlet_conc_A, inlet_conc_B, flow_rate, volume):
    """
    Wrapper function to get the reactor and results from cached calculations
    """
    # Simply call the cached function and return its results
    return run_cstr_calc(k_value, inlet_conc_A, inlet_conc_B, flow_rate, volume)


# Cache the reactor simulations to improve performance
@st.cache_data
def run_pfr_calc(k_value, initial_conc_A, initial_conc_B, diameter, flow_rate, volume):
    """
    Run pfr simulation and return only the results, not the reactor
    """
    # Define reaction
    reaction = Reaction(
        reactants={"A": 1.0}, products={"B": 1.0}, rate_constant=k_value
    )

    # Define initial concentrations
    initial_concentrations = {"A": initial_conc_A, "B": initial_conc_B}

    # Create and run reactor
    reactor = PFR(
        inlet_concentrations=initial_concentrations,
        reactions=[reaction],
        diameter=diameter,
        flow_rate=flow_rate,
        volume=volume,  # 1 m^3
        temperature=298.15,  # 25 C
    )

    # Run the simulation
    results = reactor.run(end_time=1)

    return reactor, results


def run_pfr(k_value, initial_conc_A, initial_conc_B, diameter, flow_rate, volume):
    """
    Wrapper function to get the reactor and results from cache calculations
    """
    # Simply call the cached function and return its results
    return run_pfr_calc(
        k_value, initial_conc_A, initial_conc_B, diameter, flow_rate, volume
    )


# Don't cache animation creation directly
def create_batch_animation(reactor, results, species_colors):
    """
    Create batch reactor animation (not cached)
    """
    animator = ReactorAnimation(reactor)
    animation = animator.create_batch_animation(
        results=results, species_colors=species_colors, n_particles=200
    )
    return animation


def create_cstr_animation(reactor, results, species_colors):
    """
    Create CSTR animation (not cached)
    """
    animator = ReactorAnimation(reactor)
    animation = animator.create_cstr_animation(
        results=results,
        species_colors=species_colors,
        n_particles=200,
        animation_duration=5.0,
        n_frames=100,
    )
    return animation


def create_pfr_animation(reactor, results, species_colors):
    """
    Create PFR animation (not cached)
    """
    animator = ReactorAnimation(reactor)
    animation = animator.create_pfr_animation(
        results=results,
        species_colors=species_colors,
        n_particles_per_species=100,
    )
    return animation


# Cache the GIF data instead
@st.cache_data
def cached_batch_animation_gif(
    k_value, initial_conc_A, initial_conc_B, end_time, volume, fps=10
):
    """Cache the GIF data for batch animation"""
    reactor, results = run_batch_reactor(
        k_value, initial_conc_A, initial_conc_B, end_time, volume
    )
    species_colors = {"A": "red", "B": "blue"}
    animation = create_batch_animation(reactor, results, species_colors)
    return animation_to_gif(animation, fps)


@st.cache_data
def cached_cstr_animation_gif(
    k_value, inlet_conc_A, inlet_conc_B, flow_rate, volume, fps=10
):
    """Cache the GIF data for CSTR animation"""
    reactor, results = run_cstr(k_value, inlet_conc_A, inlet_conc_B, flow_rate, volume)
    species_colors = {"A": "red", "B": "blue"}
    animation = create_cstr_animation(reactor, results, species_colors)
    return animation_to_gif(animation, fps)


@st.cache_data
def cached_pfr_animation_gif(
    k_value, inlet_conc_A, inlet_conc_B, diameter, flow_rate, volume, fps=10
):
    """Cache the GIF data for CSTR animation"""
    reactor, results = run_pfr(
        k_value, inlet_conc_A, inlet_conc_B, diameter, flow_rate, volume
    )
    species_colors = {"A": "red", "B": "blue"}
    animation = create_pfr_animation(reactor, results, species_colors)
    return animation_to_gif(animation, fps)


# Calculate residence times for CSTR conversion plot
@st.cache_data
def calculate_cstr_conversions(
    k_value, inlet_conc_A, inlet_conc_B, volume, residence_times
):
    """Calculate CSTR conversions for different residence times"""
    conversions = []

    for rt in residence_times:
        # Calculate flow rate from residence time
        flow_rate = volume / rt
        _, results = run_cstr_calc(
            k_value, inlet_conc_A, inlet_conc_B, flow_rate, volume
        )

        if inlet_conc_A > 0:
            conv = (
                1 - results["steady_state"]["concentrations"]["A"] / inlet_conc_A
            ) * 100
            conversions.append(conv)
        else:
            conversions.append(0)

    return conversions


# Main Streamlit app
def main():
    st.title("Chemical Reactor Visualizations")

    st.markdown("For the reaction:")
    st.markdown(r"$A \xrightarrow{k} B$")
    st.markdown("""
    this app visualizes different chemical reactor types with interactive parameters.
    Select a reactor type and adjust the parameters in the sidebar to see how they affect the reaction.
    """)

    # Reactor type selection
    reactor_type = st.radio(
        "Select Reactor Type:",
        (
            "Batch Reactor",
            "CSTR (Continuous Stirred-Tank Reactor)",
            "PFR (Plug Flow Reactor)",
        ),
    )

    # Common parameters in sidebar
    st.sidebar.header("Reaction Parameters")
    k_value = st.sidebar.slider(
        "Rate Constant (k, s⁻¹)",
        0.01,
        1.0,
        0.1,
        0.01,
        help="Reaction rate constant for A → B",
    )

    st.sidebar.header("Reactor Parameters")
    volume = st.sidebar.slider(
        "Reactor Volume (L)", 0.1, 5.0, 1.0, 0.1, help="Volume of the reactor"
    )

    # Set default colors
    # species_colors = {"A": "red", "B": "blue"}

    if reactor_type == "Batch Reactor":
        st.header("Batch Reactor Simulation")

        # Batch specific parameters
        st.sidebar.header("Batch Reactor Parameters")
        initial_conc_A = st.sidebar.slider(
            "Initial Concentration A (mol/L)",
            0.1,
            10.0,
            5.0,
            0.1,
            help="Initial concentration of reactant A",
        )
        initial_conc_B = st.sidebar.slider(
            "Initial Concentration B (mol/L)",
            0.0,
            5.0,
            0.0,
            0.1,
            help="Initial concentration of product B",
        )
        end_time = st.sidebar.slider(
            "Simulation Time (s)", 5.0, 120.0, 30.0, 5.0, help="Total time to simulate"
        )

        # Run simulation using your existing code
        _, results = run_batch_reactor(
            k_value, initial_conc_A, initial_conc_B, end_time, volume
        )

        # Create concentration plot
        fig, ax = plt.subplots(figsize=(10, 6))
        times = results["time"]
        concentrations = results["concentrations"]

        # Ensure times and concentrations have the same length
        if len(times) != len(concentrations):
            # Use the minimum length to avoid index errors
            min_length = min(len(times), len(concentrations))
            times = times[:min_length]
            concentrations = concentrations[:min_length]

        ax.plot(times, [c["A"] for c in concentrations], "r-", label="A")
        ax.plot(times, [c["B"] for c in concentrations], "b-", label="B")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Concentration (mol/L)")
        ax.set_title(f"Batch Reactor - k = {k_value:.2f} s⁻¹")
        ax.grid(True)
        ax.legend()

        # Add half-life line if initial concentration of A > 0
        if initial_conc_A > 0:
            half_conc = initial_conc_A / 2
            ax.axhline(y=half_conc, color="gray", linestyle="--")

            # Calculate theoretical half-life for first-order reaction
            theoretical_half_life = np.log(2) / k_value
            if theoretical_half_life < end_time:
                ax.axvline(x=theoretical_half_life, color="gray", linestyle="--")
                ax.text(
                    theoretical_half_life,
                    initial_conc_A * 0.9,
                    f"t½ = {theoretical_half_life:.2f}s",
                    ha="right",
                    bbox=dict(facecolor="white", alpha=0.7),
                )

        st.pyplot(fig)

        # Generate and display animation
        with st.spinner("Generating animation..."):
            gif_data = cached_batch_animation_gif(
                k_value, initial_conc_A, initial_conc_B, end_time, volume
            )

        display_gif(gif_data)

        # Add download button for animation
        st.download_button(
            label="Download Animation",
            data=gif_data,
            file_name="batch_reactor.gif",
            mime="image/gif",
        )

        # Add explanation
        st.markdown(f"""
        ### Batch Reactor Explanation
        
        In a batch reactor:
        - All reactants are loaded at the beginning
        - No inflow or outflow during the reaction
        - Concentration changes with time until equilibrium
        
        **Current Parameters:**
        - Rate constant (k): {k_value:.2f} s⁻¹
        - Initial concentration of A: {initial_conc_A:.1f} mol/L
        - Initial concentration of B: {initial_conc_B:.1f} mol/L
        - Reactor volume: {volume:.1f} L
        - Simulation time: {end_time:.1f} s
        
        **Observations:**
        """)

        # Calculate conversion at the end
        if initial_conc_A > 0 and len(concentrations) > 0:
            final_conversion = (1 - concentrations[-1]["A"] / initial_conc_A) * 100
            st.markdown(f"- Final conversion: {final_conversion:.1f}%")

            # Reaction half-life
            half_life = np.log(2) / k_value
            st.markdown(f"- Theoretical half-life: {half_life:.2f} s")

            # Reaction is fast or slow?
            if k_value > 0.5:
                st.markdown("- Fast reaction rate")
            elif k_value < 0.1:
                st.markdown("- Slow reaction rate")
            else:
                st.markdown("- Moderate reaction rate")

            # Show which species is dominant at the end
            if concentrations[-1]["A"] > concentrations[-1]["B"]:
                st.markdown("- Reactant A remains the dominant species at the end")
            else:
                st.markdown("- Product B becomes the dominant species by the end")

    elif reactor_type == "CSTR (Continuous Stirred-Tank Reactor)":  # CSTR
        st.header("CSTR Simulation")

        # CSTR specific parameters
        st.sidebar.header("CSTR Parameters")
        inlet_conc_A = st.sidebar.slider(
            "Inlet Concentration A (mol/L)",
            0.1,
            10.0,
            5.0,
            0.1,
            help="Concentration of reactant A in the feed",
        )
        inlet_conc_B = st.sidebar.slider(
            "Inlet Concentration B (mol/L)",
            0.0,
            5.0,
            0.0,
            0.1,
            help="Concentration of product B in the feed",
        )
        flow_rate = st.sidebar.slider(
            "Flow Rate (L/s)",
            0.05,
            5.0,
            0.5,
            0.05,
            help="Volumetric flow rate through the reactor",
        )

        # Run simulation using your existing code
        _, results = run_cstr(k_value, inlet_conc_A, inlet_conc_B, flow_rate, volume)

        # Calculate residence time
        residence_time = volume / flow_rate

        # Create concentration bar chart
        steady_state = results["steady_state"]["concentrations"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            ["A", "B"],
            [steady_state.get("A", 0), steady_state.get("B", 0)],
            color=["red", "blue"],
        )
        ax.set_ylabel("Concentration (mol/L)")
        ax.set_title(
            f"CSTR Steady State - k = {k_value:.2f} s⁻¹, Flow Rate = {flow_rate:.2f} L/s"
        )

        # Add feed concentrations as dashed outline bars
        ax.bar(
            ["A", "B"],
            [inlet_conc_A, inlet_conc_B],
            fill=False,
            linestyle="dashed",
            edgecolor=["red", "blue"],
        )
        ax.set_ylim(0, max(inlet_conc_A, steady_state.get("B", 0)) * 1.1)

        # Add conversion text
        if inlet_conc_A > 0:
            conversion = (1 - steady_state.get("A", 0) / inlet_conc_A) * 100
            ax.text(
                0.5,
                max(inlet_conc_A, steady_state.get("B", 0)) * 1.05,
                f"Conversion: {conversion:.1f}%",
                ha="center",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        st.pyplot(fig)

        # Create a second plot showing the effect of residence time
        st.subheader("Effect of Residence Time on Conversion")

        # Calculate conversions for various residence times
        residence_times = np.linspace(0.1, 100.0, 20)
        conversions = calculate_cstr_conversions(
            k_value, inlet_conc_A, inlet_conc_B, volume, residence_times
        )

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(residence_times, conversions, "bo-")

        # Mark current residence time
        ax2.axvline(x=residence_time, color="red", linestyle="--")
        ax2.text(
            residence_time,
            50,
            f"Current τ = {residence_time:.2f} s",
            color="red",
            ha="right",
            rotation=90,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        ax2.set_xlabel("Residence Time τ (s)")
        ax2.set_ylabel("Conversion (%)")
        ax2.set_ylim(0, 100)
        ax2.set_title("Conversion vs. Residence Time")
        ax2.grid(True)

        # Analytical solution for first-order reactions in CSTR
        # X = kτ/(1+kτ) where X is fractional conversion
        analytical_conv = (
            100 * k_value * residence_times / (1 + k_value * residence_times)
        )
        ax2.plot(
            residence_times,
            analytical_conv,
            "g-",
            alpha=0.7,
            label="Analytical Solution",
        )
        ax2.legend()

        st.pyplot(fig2)

        # Generate and display animation
        with st.spinner("Generating animation..."):
            gif_data = cached_cstr_animation_gif(
                k_value, inlet_conc_A, inlet_conc_B, flow_rate, volume
            )

        display_gif(gif_data)

        # Add download button for animation
        st.download_button(
            label="Download Animation",
            data=gif_data,
            file_name="cstr_reactor.gif",
            mime="image/gif",
        )

        # Add explanation
        st.markdown(f"""
        ### CSTR Explanation
        
        In a Continuous Stirred-Tank Reactor (CSTR):
        - Reactants continuously flow in
        - Products continuously flow out
        - Perfect mixing is assumed
        - Reaches a steady state where concentrations remain constant
        
        **Current Parameters:**
        - Rate constant (k): {k_value:.2f} s⁻¹
        - Inlet concentration of A: {inlet_conc_A:.1f} mol/L
        - Inlet concentration of B: {inlet_conc_B:.1f} mol/L
        - Flow rate: {flow_rate:.2f} L/s
        - Reactor volume: {volume:.1f} L
        - Residence time (τ): {residence_time:.2f} s
        
        **Observations:**
        """)

        # Add observations based on parameters
        if inlet_conc_A > 0:
            # Conversion
            conversion = (1 - steady_state.get("A", 0) / inlet_conc_A) * 100
            st.markdown(f"- Steady-state conversion: {conversion:.1f}%")

            # Residence time interpretation
            if residence_time < 0.2:
                st.markdown(
                    "- Very short residence time: Limited conversion due to insufficient time for reaction"
                )
            elif residence_time > 5.0:
                st.markdown(
                    "- Long residence time: High conversion, approaching batch reactor performance"
                )
            else:
                st.markdown(
                    "- Moderate residence time: Balanced flow and reaction rates"
                )

    else:  # PFR
        st.header("PFR Simulation")

        # PFR specific parameters
        st.sidebar.header("PFR Parameters")
        inlet_conc_A = st.sidebar.slider(
            "Inlet Concentration A (mol/L)",
            0.1,
            5.0,
            1.0,
            0.1,
            help="Concentration of reactant A in the feed",
        )
        inlet_conc_B = st.sidebar.slider(
            "Inlet Concentration B (mol/L)",
            0.0,
            5.0,
            0.0,
            0.1,
            help="Concentration of product B in the feed",
        )
        flow_rate = st.sidebar.slider(
            "Flow Rate (L/s)",
            0.05,
            5.0,
            0.1,
            0.05,
            help="Volumetric flow rate through the reactor",
        )
        diameter = st.sidebar.slider(
            "Reactor diameter (dm)",
            0.05,
            5.0,
            0.5,
            0.05,
            help="Diameter of the reactor",
        )
        # Run simulation using your existing code
        _, results = run_pfr(
            k_value, inlet_conc_A, inlet_conc_B, diameter, flow_rate, volume
        )

        # Generate and display animation
        with st.spinner("Generating animation..."):
            gif_data = cached_pfr_animation_gif(
                k_value, inlet_conc_A, inlet_conc_B, diameter, flow_rate, volume
            )

        display_gif(gif_data)

        # Add download button for animation
        st.download_button(
            label="Download Animation",
            data=gif_data,
            file_name="pfr.gif",
            mime="image/gif",
        )

        # Add explanation
        st.markdown(f"""
        ### Plug Flow Reactor Explanation
        
        A plug-flow reactor is a tubular reactor:
        - Most used often for gas-phase reactions
        - In which reactants are contnually consumed as they flow down the legnth of the reactor
        - In which the flow field may be modeled by that of a plug-flow profile (uniform readial velcocity)
        - In which there is no radial vairation in the reaction rate
        
        **Current Parameters:**
        - Rate constant (k): {k_value:.2f} s⁻¹
        - Initial concentration of A: {inlet_conc_A:.1f} mol/L
        - Initial concentration of B: {inlet_conc_B:.1f} mol/L
        - Reactor volume: {volume:.1f} L
        - Reactor diameter: {diameter:.1f} dm
        - Flow rate: {flow_rate: .1f} L
        """)


if __name__ == "__main__":
    main()
