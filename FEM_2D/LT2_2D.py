import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fem_framework as ff

output_dir = Path(
    "simple_growth",
)  # Specify the folder where all the data should be saved
output_dir.mkdir(parents=True, exist_ok=True)  # Make the folder if it doesn't exist

''' PLOTTING '''
def plot_line(r_i, line_history, line_points, time, analytical_solutions=None):
    """Plot current values of all variables along the line defined by line_points.

    Args:
        analytical_solutions: Dict with variable names as keys and analytical functions as values.
                             Each function should take line_points as input and return values.
    """

    # Get all variable names from line history
    variables = list(line_history.keys())
    num_vars = len(variables)

    # Determine indices for 10 evenly spaced time points
    if variables:
        num_time_steps = len(line_history[variables[0]])
        num_plots = min(20, num_time_steps)  # Don't try to plot more than available
        if num_plots > 0:
            indices_to_plot = np.linspace(0, num_time_steps - 1, num_plots, dtype=int)
        else:
            indices_to_plot = []
    else:
        indices_to_plot = []

    # Calculate grid dimensions - always use 1 column
    ncols = 1
    nrows = num_vars  # Each variable gets its own row

    # Evenly spaced grid from 0 to 1 in length on line_poin
    distances = line_points[:, 0]

    # Create subplots grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True)

    # Handle case where there's only one variable
    if num_vars == 1:
        axes = [axes]
    else:
        axes = axes.reshape(-1)  # Flatten to 1D array for single column

    for idx, var_name in enumerate(variables):
        ax = axes[idx]
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style="plain", axis="y")

        # Plot numerical solution
        if len(line_history[var_name]) > 0:
            for t_idx in indices_to_plot:
                time_step_values = line_history[var_name][t_idx]
                color = plt.cm.viridis(t_idx / max(1, len(line_history[var_name]) - 1))
                timestamp = time[t_idx] if t_idx < len(time) else t_idx

                # Only add labels for the first subplot to avoid duplicate legend entries
                label = f"t = {timestamp:.3f}" if idx == 0 else None

                ax.plot(
                    distances,
                    time_step_values,
                    color=color,
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.7,
                    label=label,
                )

        # Plot analytical solution if provided
        if analytical_solutions and var_name in analytical_solutions:
            print("var_name", var_name)
            analytical_values = analytical_solutions[var_name](line_points, r_i)
            ax.plot(
                distances,
                analytical_values,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label="Analytical",
            )

        ax.set_ylabel(var_name)
        ax.set_title(f"{var_name} (t = {time[-1]:.3f})")
        ax.grid(True, alpha=0.3)
    

    # Set x-label for bottom plot
    axes[-1].set_xlabel("Distance along line")
    fig.legend(fontsize="small", loc="center right")

    plt.tight_layout()
    fig.subplots_adjust(right=0.85)

    output_path = output_dir / "variables_line_spatial.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    comm = MPI.COMM_WORLD
    output_dir = Path("annulus_growth_output")

    mesh, _ , facet_tags = gmshio.read_from_msh(
        "annulus.msh", comm, rank=0, gdim=2
    )

    params = {
        "R_i": 1.0,
        "R_o": 2.0,
        "c": -0.05,
        "dt": 0.025,
        "set_point": 0.5,
        "num_steps": 1,
        "mu": 1.0,
        "g_r": 1.0,
        "g_t": 1.0,
    }

    line_points = np.array(
        [[r, 0.0] for r in np.linspace(params["R_i"], params["R_o"], 64)]
    )

    problem_vars = ff.setup_common_variables(mesh, params)
    weak_form_defs = ff.setup_problem(mesh, facet_tags, params, problem_vars)

    data_collector = ff.DataCollector(comm, output_dir, line_points)
    data_collector.register_function("u", problem_vars["u"])
    data_collector.register_function("p", problem_vars["p"])
    data_collector.register_function("Hoop Stress", problem_vars["cauchy_ff"])
    data_collector.register_function("Radial Stress", problem_vars["cauchy_nn"])
    data_collector.register_line_data("Hoop Stress", problem_vars["cauchy_ff"])
    data_collector.register_line_data("Radial Stress", problem_vars["cauchy_nn"])
    data_collector.register_line_data("Cumulative Hoop Growth", problem_vars["g_t"])
    data_collector.register_line_data("p", problem_vars["p"])
    data_collector.setup_writers()

    sims = ff.run_simulation(
        params, problem_vars, weak_form_defs, data_collector
        )
    
    data_collector.close()

    breakpoint()

if __name__ == "__main__":
    main()