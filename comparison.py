import LT2
import FEM_2D.annulus_growth
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

''' 1D Solution '''
#region
# Initialize base state
r_range = np.linspace(1.0, 2.0, 64)  # 64 points from 1 to 2 with step 1/63
initial_gf = np.ones_like(r_range)
initial_gr = np.ones_like(r_range)  # No initial growth

base_state = LT2.FastState(
    _Ri=1.0,
    gr=initial_gr,
    gt=initial_gf,
    bc=-0.05,
    mu=1.0,
    gMax=1.5,
    lambdaCrit=0.5,
    gamma=1,
    tau=0.025
)

print("Initial state:", base_state)

# Initial calculations
ri = base_state.find_inner_radius()
stress_points = np.arange(1, 2.1, 0.1)
stress_data = [base_state.radial_stress(ri, x) for x in stress_points]

states = [base_state]
next_state = base_state.update()
#endregion

''' 2D Solution '''

output_dir = Path(
    "simple_growth",
)  # Specify the folder where all the data should be saved
output_dir.mkdir(parents=True, exist_ok=True)  # Make the folder if it doesn't exist

comm = MPI.COMM_WORLD

mesh, _ , facet_tags = gmshio.read_from_msh(
    "annulus.msh", comm, rank=0, gdim=2
)

params = {
    "R_i": 1.0,
    "R_o": 2.0,
    "c": -0.05,
    "dt": 0.025,
    "set_point": 0.5,
    "num_steps": 2,
    "mu": 1.0,
    "g_2": 1.0,
}

line_points = np.array(
    [[r, 0.0] for r in np.linspace(params["R_i"], params["R_o"], 64)]
)

problem_vars = FEM_2D.annulus_growth.setup_problem(mesh, params)
weak_form_defs = FEM_2D.annulus_growth.weak_formulation(mesh, facet_tags, params, problem_vars)

data_collector = FEM_2D.annulus_growth.DataCollector(comm, output_dir, line_points)
data_collector.register_function("u", problem_vars["u"])
data_collector.register_function("p", problem_vars["p"])
data_collector.register_function("Hoop Stress", problem_vars["stress_ff"])
data_collector.register_function("Radial Stress", problem_vars["stress_nn"])
data_collector.register_line_data("Hoop Stress", problem_vars["stress_ff"])
data_collector.register_line_data("Radial Stress", problem_vars["stress_nn"])
data_collector.register_line_data("Cumulative Hoop Growth", problem_vars["g_2"])
data_collector.register_line_data("p", problem_vars["p"])
data_collector.setup_writers()

time = FEM_2D.annulus_growth.run_simulation(
    params, problem_vars, weak_form_defs, data_collector
    )

np.array([next_state.radial_stress(ri, s) for s in r_range])

# Create a figure with 3 subplots
fig, axs = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("1D vs 2D Model After First Growth Step")

# --- Radial Stress Plot ---
ri_1d = next_state.find_inner_radius()
radial_stress_1d = np.array([next_state.radial_stress(ri_1d, s) * (s / next_state.compute_r(ri_1d, s)) for s in r_range])
radial_stress_2d = data_collector.line_history.get("Radial Stress")[1].flatten()

axs[0].plot(r_range, radial_stress_1d, label="1D (ODE)")
axs[0].plot(r_range, radial_stress_2d, label="2D (FEM)", linestyle='--')
axs[0].set_title("Radial Stress (Cauchy)")
axs[0].set_xlabel("Radius (R)")
axs[0].set_ylabel("Stress")
axs[0].legend()
axs[0].grid(True)

# --- Hoop Stress Plot ---
hoop_stress_1d = np.array([next_state.angular_stress(ri_1d, s) * (next_state.compute_r(ri_1d, s) / s) for s in r_range])
hoop_stress_2d = data_collector.line_history.get("Hoop Stress")[1].flatten()

axs[1].plot(r_range, hoop_stress_1d, label="1D (ODE)")
axs[1].plot(r_range, hoop_stress_2d, label="2D (FEM)", linestyle='--')
axs[1].set_title("Hoop Stress (Cauchy)")
axs[1].set_xlabel("Radius (R)")
axs[1].set_ylabel("Stress")
axs[1].legend()
axs[1].grid(True)

# --- Pressure Plot ---
pressure_1d = np.array([next_state.compute_p(ri_1d, s) for s in r_range])
pressure_2d = data_collector.line_history.get("p")[1].flatten()

axs[2].plot(r_range, pressure_1d, label="1D (ODE)")
axs[2].plot(r_range, pressure_2d, label="2D (FEM)", linestyle='--')
axs[2].set_title("Pressure (p)")
axs[2].set_xlabel("Radius (R)")
axs[2].set_ylabel("Pressure")
axs[2].legend()
axs[2].grid(True)

# --- Growth Plot ---
### NOTE THE DIFFERENCE BETWEEN g_theta AND g_2 ###
dgt = np.array([next_state.compute_dgt(ri_1d, s) for s in r_range])
g_2 = next_state.gt*dgt
growth_2d = data_collector.line_history.get("Cumulative Hoop Growth")[1].flatten()
axs[3].plot(r_range, g_2, label="1D (ODE)")
axs[3].plot(r_range, growth_2d, label="2D (FEM)", linestyle='--')
axs[3].set_title("Cumulative Hoop Growth (g_theta)")
axs[3].set_xlabel("Radius (R)")
axs[3].set_ylabel("Growth")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.savefig("1D_vs_2D_comparison.png")

data_collector.close()