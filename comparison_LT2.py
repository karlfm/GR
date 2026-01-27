import LT2
import FEM_2D.fem_framework as ff
import numpy as np
from cylinder_functions import BaseState
from dolfinx.io import gmshio
from mpi4py import MPI
from pathlib import Path
import matplotlib.pyplot as plt
import plotter 
import time

start_time = time.perf_counter()
''' 1D Solution '''
#region
# Initialize base state
R_range = np.linspace(1.0, 2.0, 64)  # 64 points from 1 to 2 with step 1/63
# initial_gt linear profile from 1 to 1.5
initial_gt = np.ones_like(R_range)
initial_gr = np.ones_like(R_range)  # No initial growth
# initial_gt = np.ones_like(r_range)

dt = 0.00625
mu=1.0
strain_set_point = 0.1
stress_set_point = 0.05 # mu * strain_set_point ** 2
print("Stress set point:", stress_set_point)
class LT2State(BaseState):

    def compute_dgt(self, ri, s):

        stress_term = self.hoop_cauchy(ri, s)

        dgt = self.tau * (stress_term - self.set_point) / self.set_point + 1

        return dgt

    def compute_dgr(self, ri, s):
        return 1.0
    
initial_state = LT2State(
    R=R_range,
    gr=initial_gr,
    gt=initial_gt,
    bc=-0.05,
    mu=mu,
    gMax=1.5,
    set_point=stress_set_point,
    gamma=1,
    tau=dt
)

print("Initial state:", initial_state)

# Initial calculations
ri = initial_state.find_inner_radius()
stress_points = np.arange(1, 2.1, 0.1)
stress_data = [initial_state.radial_stress(ri, x) for x in stress_points]

states = [initial_state]
prev_state = initial_state
num_steps = 128
for step in range(1, num_steps + 1):
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
    "dt": dt,
    "set_point": stress_set_point,
    "num_steps": num_steps,
    "mu": mu,
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
data_collector.register_line_data("Hoop Strain", problem_vars["strain_ff"])
data_collector.register_line_data("Radial Strain", problem_vars["strain_nn"])
data_collector.register_line_data("Cumulative Hoop Growth", problem_vars["g_t"])
data_collector.register_line_data("Incremental Hoop Growth", problem_vars["dgt"])
data_collector.register_line_data("p", problem_vars["p"])
data_collector.register_line_data("u_r", problem_vars["u_r"])
data_collector.setup_writers()

sims = ff.run_simulation(
    params, problem_vars, weak_form_defs, data_collector
    )

# --- Pre-calculate all data for plotting ---
print("--- Pre-calculating data for plots ---")
plot_data_1d = {
    "radial_stress": [], "hoop_stress": [], "radial_strain": [],
    "hoop_strain": [], "radial_growth": [], "hoop_growth": [], "displacement": [],
    "Ricci": []
}
plot_data_2d = {
    "radial_stress": [], "hoop_stress": [], "radial_strain": [],
    "hoop_strain": [], "radial_growth": [], "hoop_growth": [], "displacement": []
}
power_data = {"power": [], "entropy": [], "internal_entropy": []}

# Calculate data for each state
number_of_lines = 8
states_to_plot_idx = np.linspace(0, num_steps, num=number_of_lines, dtype=int)
states_to_plot_1d = [states[i] for i in states_to_plot_idx]
plot_data_2d = {
    key: [value_list[i] for i in states_to_plot_idx]
    for key, value_list in data_collector.line_history.items()
}
for state in states_to_plot_1d:
    ri_1d = state.find_inner_radius()
    plot_data_1d["radial_stress"].append(np.array([state.radial_stress(ri_1d, s) * (s / state.compute_r(ri_1d, s)) for s in R_range]))
    plot_data_1d["hoop_stress"].append(np.array([state.angular_stress(ri_1d, s) * (state.compute_r(ri_1d, s) / s) / (state.gr_interp(s) * state.gt_interp(s)) for s in R_range]))
    plot_data_1d["radial_strain"].append(np.array([state.radial_strain(ri_1d, s) for s in R_range]))
    plot_data_1d["hoop_strain"].append(np.array([state.hoop_strain(ri_1d, s) for s in R_range]))
    plot_data_1d["radial_growth"].append(state.gr)
    plot_data_1d["hoop_growth"].append(state.gt)
    plot_data_1d["displacement"].append(np.array([state.compute_r(ri_1d, s) for s in R_range]))
    plot_data_1d["Ricci"].append(np.array([state.Ricci_curvature(ri_1d, s) for s in R_range]))


# Calculate data between states (power)
for i in range(number_of_lines - 1):
    state1 = states_to_plot_1d[i]
    state2 = states_to_plot_1d[i+1]
    # power_split = KFR.power(state1, state2, R_range, dt)
    power_direct = BaseState.power_direct(state1, state2, R_range, dt)
    # power_data["split"].append(power_split)
    power_data["power"].append(power_direct)
    entropy = BaseState.entropy(state1, state2, R_range, dt)
    power_data["internal_entropy"].append(entropy)
    power_data["entropy"].append(power_direct - entropy)
    # power_data["difference"].append(power_direct - power_split)
print("--- Plotting results ---")

# --- Use the Plotter Class ---
plotter_instance = plotter.ComparisonPlotter(R_range, number_of_lines, model_name="LT2")

# Plot spatial data
plotter_instance.plot_spatial_panel((0, 0), "Radial Stress (Cauchy)", "Stress", plot_data_1d["radial_stress"], plot_data_2d.get("Radial Stress"))
plotter_instance.plot_spatial_panel((1, 0), "Hoop Stress (Cauchy)", "Stress", plot_data_1d["hoop_stress"], plot_data_2d.get("Hoop Stress"), set_point=stress_set_point)
plotter_instance.plot_spatial_panel((0, 1), "Radial Strain", "Strain", plot_data_1d["radial_strain"], plot_data_2d.get("Radial Strain"))
plotter_instance.plot_spatial_panel((1, 1), "Hoop Strain", "Strain", plot_data_1d["hoop_strain"], plot_data_2d.get("Hoop Strain"))
plotter_instance.plot_spatial_panel((0, 2), "Radial Growth", "Growth", plot_data_1d["radial_growth"], plot_data_2d.get("Cumulative Radial Growth"))
plotter_instance.plot_spatial_panel((1, 2), "Hoop Growth", "Growth", plot_data_1d["hoop_growth"], plot_data_2d.get("Cumulative Hoop Growth"))
plotter_instance.plot_spatial_panel((2, 0), "Displacement (r)", "Displacement", plot_data_1d["displacement"], plot_data_2d.get("u_r"))

# Plot special cases
plotter_instance.plot_ricci((2, 1), plot_data_1d["Ricci"])
plotter_instance.plot_dissipation((2, 2), power_data, dt)

# Finalize and save
plotter_instance.finalize_and_save("1D_vs_2D_LT2.png")

data_collector.close()

end_time = time.perf_counter()
print(f"--- 2D Simulation finished in {end_time - start_time:.4f} seconds ---")


# # Create a figure with 3 subplots
# fig, axs = plt.subplots(2, 3, figsize=(18, 5))
# fig.suptitle("1D vs 2D Model")

# # --- Radial Stress Plot ---
# radial_stress_2d_history = data_collector.line_history.get("Radial Stress")
# print("FEM len: " + str(len(radial_stress_2d_history)))
# print("ODE len: " + str(len(states)))
# for i, state in enumerate(states):
#     ri_1d = state.find_inner_radius()
#     radial_stress_1d = np.array([state.radial_stress(ri_1d, s) * (s / state.compute_r(ri_1d, s)) for s in r_range])
#     axs[0, 0].plot(r_range, radial_stress_1d, color='C0', alpha=(i+1)/(num_steps + 1), label=f"1D (ODE) {i}")

#     radial_stress_2d = radial_stress_2d_history[i].flatten()
#     axs[0, 0].plot(r_range, radial_stress_2d, color='C1', linestyle='--', alpha=(i+1)/(num_steps + 1), label=f"2D (FEM) {i}")

# axs[0, 0].set_title("Radial Stress (Cauchy)")
# axs[0, 0].set_xlabel("Radius (R)")
# axs[0, 0].set_ylabel("Stress")
# # axs[0, 0].legend()
# axs[0, 0].grid(True)

# # --- Hoop Stress Plot ---
# hoop_stress_1d = np.array([base_state.angular_stress(ri_1d, s) * (base_state.compute_r(ri_1d, s) / s) / (base_state.gr_interp(s) * base_state.gt_interp(s)) for s in r_range])
# hoop_stress_2d = data_collector.line_history.get("Hoop Stress")[0].flatten()
# for i, state in enumerate(states):
#     ri_1d = state.find_inner_radius()
#     hoop_stress_1d = np.array([state.angular_stress(ri_1d, s) * (state.compute_r(ri_1d, s) / s) / (state.gr_interp(s) * state.gt_interp(s)) for s in r_range])
#     axs[0, 1].plot(r_range, hoop_stress_1d, color='C0', alpha=(i+1)/(num_steps + 1), label=f"1D (ODE) {i}")
#     hoop_stress_2d = data_collector.line_history.get("Hoop Stress")[i].flatten()
#     axs[0, 1].plot(r_range, hoop_stress_2d, color='C1', linestyle='--', alpha=(i+1)/(num_steps + 1), label=f"2D (FEM) {i}")

# axs[0, 1].set_title("Hoop Stress (Cauchy)")
# axs[0, 1].set_xlabel("Radius (R)")
# axs[0, 1].set_ylabel("Stress")
# # axs[0, 1].legend()
# axs[0, 1].grid(True)

# # --- Pressure Plot ---
# for i, state in enumerate(states):
#     ri_1d = state.find_inner_radius()
#     pressure_1d = np.array([state.compute_p(ri_1d, s) for s in r_range])
#     axs[0, 2].plot(r_range, pressure_1d, color='C0', alpha=(i+1)/(num_steps + 1), label=f"1D (ODE) {i}")
#     pressure_2d = data_collector.line_history.get("p")[i].flatten()
#     axs[0, 2].plot(r_range, pressure_2d, color='C1', linestyle='--', alpha=(i+1)/(num_steps + 1), label=f"2D (FEM) {i}")

# axs[0, 2].set_title("Pressure (p)")
# axs[0, 2].set_xlabel("Radius (R)")
# axs[0, 2].set_ylabel("Pressure")
# # axs[0, 2].legend()
# axs[0, 2].grid(True)

# # --- Growth Plot ---
# ### NOTE THE DIFFERENCE BETWEEN g_theta AND g_2 ###
# for i, state in enumerate(states):
#     ri_1d = state.find_inner_radius()
#     g_2 = state.gt
#     axs[1,0].plot(r_range, g_2, color='C0', alpha=(i+1)/(num_steps + 1), label=f"1D (ODE) {i}")
#     growth_2d = data_collector.line_history.get("Cumulative Hoop Growth")[i].flatten()
#     axs[1,0].plot(r_range, growth_2d, color='C1', linestyle='--', alpha=(i+1)/(num_steps + 1), label=f"2D (FEM) {i}")
# axs[1,0].set_title("Cumulative Hoop Growth (g_theta)")
# axs[1,0].set_xlabel("Radius (R)")
# axs[1,0].set_ylabel("Growth")
# # axs[1,0].legend()
# axs[1,0].grid(True)

# # --- Incremental Growth Plot ---
# for i, state in enumerate(states):
#     ri_1d = state.find_inner_radius()
#     dgt = np.array([state.compute_dgt(ri_1d, s) for s in r_range])
#     axs[1,1].plot(r_range, dgt, color='C0', alpha=(i+1)/(num_steps + 1), label=f"1D (ODE) {i}")
#     dgt_FEM = data_collector.line_history.get("Incremental Hoop Growth")[i].flatten()
#     axs[1,1].plot(r_range, dgt_FEM, color='C1', linestyle='--', alpha=(i+1)/(num_steps + 1), label=f"2D (FEM) {i}")
# axs[1,1].set_title("Incremental Hoop Growth (dgt)") 
# axs[1,1].set_xlabel("Radius (R)")
# axs[1,1].set_ylabel("Growth")
# # axs[1,1].legend()
# axs[1,1].grid(True)

# # Create a single legend for the entire figure, showing every other pair
# handles, labels = axs[0,0].get_legend_handles_labels()
# selected_handles = []
# selected_labels = []
# for i in range(0, len(handles), 4):
#     selected_handles.extend(handles[i:i+2])
#     selected_labels.extend(labels[i:i+2])
# fig.legend(selected_handles, selected_labels, loc='center right')

# plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make room for the legend
# plt.savefig("1D_vs_2D_comparison_constant_scaled_growth.png")
