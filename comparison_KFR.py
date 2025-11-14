import KFR
import FEM_2D.KFR_2D
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
# initial_gt linear profile from 1 to 1.5
initial_gt = np.ones_like(r_range)
initial_gr = np.ones_like(r_range)  # No initial growth
# initial_gt = np.ones_like(r_range)

base_state = KFR.FastState(
    _Ri=1.0,
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

states = [base_state]
prev_state = base_state
num_steps = 5
for step in range(1, num_steps + 1):  # 2 time steps
    print(f"Time step {step}")
    next_state = prev_state.update()
    states.append(next_state)
    prev_state = next_state
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
    "num_steps": num_steps,
    "mu": 1.0,
    "g_r": 1.0,
    "g_t": 1.0,
}

line_points = np.array(
    [[r, 0.0] for r in np.linspace(params["R_i"], params["R_o"], 64)]
)

problem_vars = FEM_2D.KFR_2D.setup_problem(mesh, params)
weak_form_defs = FEM_2D.KFR_2D.weak_formulation(mesh, facet_tags, params, problem_vars)

data_collector = FEM_2D.KFR_2D.DataCollector(comm, output_dir, line_points)
data_collector.register_function("u", problem_vars["u"])
data_collector.register_function("p", problem_vars["p"])
data_collector.register_function("Hoop Stress", problem_vars["stress_ff"])
data_collector.register_function("Radial Stress", problem_vars["stress_nn"])
data_collector.register_line_data("Hoop Stress", problem_vars["stress_ff"])
data_collector.register_line_data("Radial Stress", problem_vars["stress_nn"])
data_collector.register_line_data("Elastic Hoop Strain", problem_vars["elastic_strain_ff"])
data_collector.register_line_data("Cumulative Hoop Growth", problem_vars["g_t"])
data_collector.register_line_data("Incremental Hoop Growth", problem_vars["dgt"])
data_collector.register_line_data("p", problem_vars["p"])
data_collector.setup_writers()

time = FEM_2D.KFR_2D.run_simulation(
    params, problem_vars, weak_form_defs, data_collector
    )

# Create a figure with 3 subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 5))
fig.suptitle("1D vs 2D Model KFR")

# --- Radial Stress Plot ---
radial_stress_2d_history = data_collector.line_history.get("Radial Stress")
print("FEM len: " + str(len(radial_stress_2d_history)))
print("ODE len: " + str(len(states)))
for i, state in enumerate(states):
    ri_1d = state.find_inner_radius()
    radial_stress_1d = np.array([state.radial_stress(ri_1d, s) * (s / state.compute_r(ri_1d, s)) for s in r_range])
    axs[0, 0].plot(r_range, radial_stress_1d, color='C0', alpha=(i+1)/(num_steps + 1), label=f"1D (ODE) {i}")

    radial_stress_2d = radial_stress_2d_history[i].flatten()
    axs[0, 0].plot(r_range, radial_stress_2d, color='C1', linestyle='--', alpha=(i+1)/(num_steps + 1), label=f"2D (FEM) {i}")

axs[0, 0].set_title("Radial Stress (Cauchy)")
axs[0, 0].set_xlabel("Radius (R)")
axs[0, 0].set_ylabel("Stress")
# axs[0, 0].legend()
axs[0, 0].grid(True)

# --- Hoop Stress Plot ---
hoop_stress_1d = np.array([base_state.angular_stress(ri_1d, s) * (base_state.compute_r(ri_1d, s) / s) / (base_state.gr_interp(s) * base_state.gt_interp(s)) for s in r_range])
hoop_stress_2d = data_collector.line_history.get("Hoop Stress")[0].flatten()
for i, state in enumerate(states):
    ri_1d = state.find_inner_radius()
    hoop_stress_1d = np.array([state.angular_stress(ri_1d, s) * (state.compute_r(ri_1d, s) / s) / (state.gr_interp(s) * state.gt_interp(s)) for s in r_range])
    axs[0, 1].plot(r_range, hoop_stress_1d, color='C0', alpha=(i+1)/(num_steps + 1), label=f"1D (ODE) {i}")
    hoop_stress_2d = data_collector.line_history.get("Hoop Stress")[i].flatten()
    axs[0, 1].plot(r_range, hoop_stress_2d, color='C1', linestyle='--', alpha=(i+1)/(num_steps + 1), label=f"2D (FEM) {i}")

axs[0, 1].set_title("Hoop Stress (Cauchy)")
axs[0, 1].set_xlabel("Radius (R)")
axs[0, 1].set_ylabel("Stress")
# axs[0, 1].legend()
axs[0, 1].grid(True)

# --- Pressure Plot ---
for i, state in enumerate(states):
    ri_1d = state.find_inner_radius()
    pressure_1d = np.array([state.compute_p(ri_1d, s) for s in r_range])
    axs[0, 2].plot(r_range, pressure_1d, color='C0', alpha=(i+1)/(num_steps + 1), label=f"1D (ODE) {i}")
    pressure_2d = data_collector.line_history.get("p")[i].flatten()
    axs[0, 2].plot(r_range, pressure_2d, color='C1', linestyle='--', alpha=(i+1)/(num_steps + 1), label=f"2D (FEM) {i}")

axs[0, 2].set_title("Pressure (p)")
axs[0, 2].set_xlabel("Radius (R)")
axs[0, 2].set_ylabel("Pressure")
# axs[0, 2].legend()
axs[0, 2].grid(True)

# --- Growth Plot ---
### NOTE THE DIFFERENCE BETWEEN g_theta AND g_2 ###
for i, state in enumerate(states):
    ri_1d = state.find_inner_radius()
    g_2 = state.gt
    axs[1,0].plot(r_range, g_2, color='C0', alpha=(i+1)/(num_steps + 1), label=f"1D (ODE) {i}")
    growth_2d = data_collector.line_history.get("Cumulative Hoop Growth")[i].flatten()
    axs[1,0].plot(r_range, growth_2d, color='C1', linestyle='--', alpha=(i+1)/(num_steps + 1), label=f"2D (FEM) {i}")
axs[1,0].set_title("Cumulative Hoop Growth (g_theta)")
axs[1,0].set_xlabel("Radius (R)")
axs[1,0].set_ylabel("Growth")
# axs[1,0].legend()
axs[1,0].grid(True)

# --- Incremental Growth Plot ---
for i, state in enumerate(states):
    ri_1d = state.find_inner_radius()
    dgt = np.array([state.compute_dgt(ri_1d, s) for s in r_range])
    axs[1,1].plot(r_range, dgt, color='C0', alpha=(i+1)/(num_steps + 1), label=f"1D (ODE) {i}")
    dgt_FEM = data_collector.line_history.get("Incremental Hoop Growth")[i].flatten()
    axs[1,1].plot(r_range, dgt_FEM, color='C1', linestyle='--', alpha=(i+1)/(num_steps + 1), label=f"2D (FEM) {i}")
axs[1,1].set_title("Incremental Hoop Growth (dgt)") 
axs[1,1].set_xlabel("Radius (R)")
axs[1,1].set_ylabel("Growth")
# axs[1,1].legend()
axs[1,1].grid(True)

for i, state in enumerate(states):
    ri_1d = state.find_inner_radius()
    elastic_hoop_strain_1d = np.array([state.elastic_hoop_strain(ri_1d, s) for s in r_range])
    axs[1,2].plot(r_range, elastic_hoop_strain_1d, color='C0', alpha=(i+1)/(num_steps + 1), label=f"1D (ODE) {i}")
    hoop_strain_2d = data_collector.line_history.get("Elastic Hoop Strain")[i].flatten()
    axs[1,2].plot(r_range, hoop_strain_2d, color='C1', linestyle='--', alpha=(i+1)/(num_steps + 1), label=f"2D (FEM) {i}")
axs[1,2].set_title("Hoop Strain") 
axs[1,2].set_xlabel("Radius (R)")
axs[1,2].set_ylabel("Strain")
axs[1,2].grid(True)

# Create a single legend for the entire figure, showing every other pair
handles, labels = axs[0,0].get_legend_handles_labels()
selected_handles = []
selected_labels = []
for i in range(0, len(handles), 4):
    selected_handles.extend(handles[i:i+2])
    selected_labels.extend(labels[i:i+2])
fig.legend(selected_handles, selected_labels, loc='center right')

plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make room for the legend
plt.savefig("1D_vs_2D_KFR.png")

data_collector.close()