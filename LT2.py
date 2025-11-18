import numpy as np
from cylinder_functions import BaseState
import time
import matplotlib.pyplot as plt


def plot_and_save(states, r_range, time, n=20, filename="cylinder_LT2.png"):
    """
    Plots various simulation results for n evenly spaced states and saves the figure.
    """
    print(f"\nGenerating and saving plot to {filename}...")
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("LT2 (hoop driven hoop stress)", fontsize=16)
    
    num_total_states = len(states)
    if n >= num_total_states:
        plot_indices = np.arange(num_total_states)
    else:
        # Ensure the first and last states are included
        plot_indices = np.unique(np.linspace(0, num_total_states - 1, n, dtype=int))

    states_to_plot = [(i, states[i]) for i in plot_indices]
    colors = plt.cm.viridis(np.linspace(0, 1, len(states_to_plot)))

    # 1. Plot Growth Factor (gt) vs. R
    ax = axs[0, 0]
    for i, (state_idx, state) in enumerate(states_to_plot):
        ax.plot(r_range, state.gt, color=colors[i], label=f"t= {time[state_idx]:.3f}")
    ax.set_title("Cumulative Growth (gt) vs. Material Coordinate (R)")
    ax.set_xlabel("R")
    ax.set_ylabel(r"$g_\theta(R)$")
    ax.grid(True)

    # 2. Plot Displacement (r) vs. R
    ax = axs[0, 1]
    for i, (state_idx, state) in enumerate(states_to_plot):
        ri = state.find_inner_radius()
        r_values = [state.compute_r(ri, s) for s in r_range]
        ax.plot(r_range, r_values, color=colors[i], label=f"Iter {state_idx}")
    ax.set_title("Displacement (r) vs. Material Coordinate (R)")
    ax.set_xlabel("R")
    ax.set_ylabel("r(R)")
    ax.grid(True)

    # 3. Plot Radial Stress vs. R for selected states
    ax = axs[1, 0]
    stress_plot_points = np.linspace(1.0, 2.0, 50)
    
    for i, (state_idx, state) in enumerate(states_to_plot):
        ri = state.find_inner_radius()
        stress_values = [state.radial_stress(ri, s) * (s / state.compute_r(ri, s)) for s in stress_plot_points]
        ax.plot(stress_plot_points, stress_values, color=colors[i], label=f"Iter {state_idx}")
    
    ax.set_title("Radial Stress (Cauchy) vs. Material Coordinate (R)")
    ax.set_xlabel("R")
    ax.set_ylabel("Radial Stress")
    ax.grid(True)

    # 4. Plot Growth Trigger vs. R
    ax = axs[1, 1]
    for i, (state_idx, state) in enumerate(states_to_plot):
        ri = state.find_inner_radius()
        triggers = [state.compute_dgt(ri, s) for s in r_range]
        ax.plot(r_range, triggers, color=colors[i], label=f"Iter {state_idx}")
    ax.axhline(1, color='r', linestyle='--', label="Equilibrium (0)")
    ax.set_title("Growth Trigger vs. Material Coordinate (R)")
    ax.set_xlabel("R")
    ax.set_ylabel(r"dgt = $\frac{R}{r}g_\theta - a_r^*$")
    ax.grid(True)

    # 5. Plot Circumferential Stress vs. R for selected states
    ax = axs[2, 0]
    for i, (state_idx, state) in enumerate(states_to_plot):
        ri = state.find_inner_radius()
        stress_values = [state.angular_stress(ri, s) * (state.compute_r(ri, s) / s) / (state.gr_interp(s) * state.gt_interp(s)) for s in stress_plot_points]
        ax.plot(stress_plot_points, stress_values, color=colors[i], label=f"Iter {state_idx}")
    ax.set_title("Circumferential Stress (Cauchy) vs. Material Coordinate (R)")
    ax.set_xlabel("R")
    ax.set_ylabel("Circumferential Stress")
    ax.grid(True)

    # 6. Plot p
    ax = axs[2, 1]
    for i, (state_idx, state) in enumerate(states_to_plot):
        ri = state.find_inner_radius()
        p_values = [state.compute_p(ri, s) for s in r_range]
        ax.plot(r_range, p_values, color=colors[i], label=f"Iter {state_idx}")
    ax.set_title("Pressure (p) vs. Material Coordinate (R)")
    ax.set_xlabel("R")
    ax.set_ylabel("p(R)")
    ax.grid(True)

    # Add a single legend for the iterations
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title="Time")
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(filename)
    plt.close(fig)
    print("✓ Plot saved.")

class LT2State(BaseState):

    def compute_dgt(self, ri, s):

        stress_term = self.hoop_cauchy(ri, s)

        dgt = self.tau * (stress_term - self.set_point) / self.set_point + 1

        return dgt

    def compute_dgr(self, ri, s):
        return 1.0

def main():
    R_range = np.arange(1, 2 + 1/64, 1/64)
    # Initialize base state
    initial_gr = np.ones_like(R_range)  # No initial growth
    initial_gt = np.ones_like(R_range)

    base_state = LT2State(
        R=R_range,
        gr=initial_gr,
        gt=initial_gt,
        bc=-0.05,
        mu=1.0,
        gMax=1.5,
        set_point=0.5,
        gamma=1,
        tau=0.025
    )
    
    print("Initial state:", base_state)
    
    # Initial calculations
    ri = base_state.find_inner_radius()
    stress_points = np.arange(1, 2.1, 0.1)
    stress_data = [base_state.radial_stress(ri, x) for x in stress_points]
    
    print(f"\nInitial inner radius: {ri:.15f}")
    print("Initial stress data:")
    print("[" + ",".join(f"{x:.15f}" for x in stress_data) + "]")
    
    # Iterate state updates
    print("\nIterating states...")
    states = [base_state]
    current_state = base_state
    
    n = 20
    for i in range(n):
        print(f"  Iteration {i+1}/{n}", end="", flush=True)
        start = time.time()
        current_state = current_state.update()
        states.append(current_state)
        print(f" - {time.time()-start:.3f}s")
    
    # Final calculations
    print("\nFinal state:", states[-1])
    
    last_state = states[-1]
    ri = last_state.find_inner_radius()
    stress_data = [last_state.radial_stress(ri, x) for x in stress_points]
    
    # Convert PK1 stress to Cauchy stress
    # Multiply by (r/R)^2
    stress_data = [
        stress_data[i] * (last_state.compute_r(ri, stress_points[i]) / stress_points[i])**2
        for i in range(len(stress_points))
    ]    

    print("\nFinal stress data:")
    print("[" + ",".join(f"{x:.15f}" for x in stress_data) + "]")
    
    # Final gs values
    gs = [state.gr_interp(np.arange(1, 2 + 1/64, 1/64)).tolist() 
          for state in states]
    print("\nFirst gs:", gs[0][:3], "...")
    print("Last gs:", gs[-1][:3], "...")
    
    time_points = np.arange(n) * base_state.tau
    plot_and_save(states[:-1], R_range, time=time_points)

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"\n✓ Total execution time: {total_time:.2f} seconds")
