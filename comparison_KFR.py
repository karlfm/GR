import numpy as np
from cylinder_functions import BaseState
from dolfinx.io import gmshio
from mpi4py import MPI
from pathlib import Path
import plotter
import FEM_2D.fem_framework as ff
import dolfinx
import ufl

''' 1D Solution '''
#region
# Initialize base state
R_range = np.linspace(1.0, 2.0, 64)  # 64 points from 1 to 2 with step 1/63
initial_gt = np.ones_like(R_range)
initial_gr = np.ones_like(R_range)  # No initial growth

dt = 0.05
mu=1.0
strain_set_point = 1.1
#Green-Lagrange set point
GL_set_point = 0.5*(strain_set_point**2 - 1)

class KFRState(BaseState):

    def compute_dgt(self, ri, s):
        """Compute growth rate based on circumferential stress."""
        
        elastic_strain = self.elastic_hoop_strain(ri, s)

        dg = (self.tau*(np.sqrt(2 * elastic_strain + 1) - 1 - self.set_point) + 1)**(1/3)

        return dg
    
    def compute_dgr(self, ri, s):
        """Compute growth rate based on circumferential stress."""
        
        elastic_strain = self.elastic_hoop_strain(ri, s)

        dg = (self.tau*(np.sqrt(2 * elastic_strain + 1) - 1 - self.set_point) + 1)**(1/3)

        return dg

print("Using Green-Lagrange set point:", GL_set_point)
base_state = KFRState(
    R=R_range,
    gr=initial_gr,
    gt=initial_gt,
    bc=-0.05,
    mu=mu,
    gMax=1.5,
    set_point=GL_set_point,
    gamma=1,
    tau=dt
)

print("Initial state:", base_state)

# Initial calculations
ri = base_state.find_inner_radius()
stress_points = np.arange(1, 2.1, 0.1)
stress_data = [base_state.radial_stress(ri, x) for x in stress_points]

states = [base_state]
prev_state = base_state
num_steps = 512
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
    "dt": dt,
    "set_point": GL_set_point,
    "num_steps": num_steps,
    "mu": mu,
    "g_r": 1.0,
    "g_t": 1.0,
}

line_points = np.array(
    [[r, 0.0] for r in np.linspace(params["R_i"], params["R_o"], 64)]
)

problem_vars = ff.setup_common_variables(mesh, params)

def kfr_growth_law(problem_vars, params, scalar_space):

    return {
        "dgr_expr": dolfinx.fem.Expression(pow((params["dt"] * (ufl.sqrt(2*problem_vars["elastic_strain_ff"] + 1) - 1 - params["set_point"]) + 1), 1/3), scalar_space.element.interpolation_points()),
        "dgt_expr": dolfinx.fem.Expression(pow((params["dt"] * (ufl.sqrt(2*problem_vars["elastic_strain_ff"] + 1) - 1 - params["set_point"]) + 1), 1/3), scalar_space.element.interpolation_points()),

        "gr_expr": dolfinx.fem.Expression(problem_vars["g_r"] * pow((params["dt"] * (ufl.sqrt(2*problem_vars["elastic_strain_ff"] + 1) - 1 - params["set_point"]) + 1), 1/3), scalar_space.element.interpolation_points()),
        "gt_expr": dolfinx.fem.Expression(problem_vars["g_t"] * pow((params["dt"] * (ufl.sqrt(2*problem_vars["elastic_strain_ff"] + 1) - 1 - params["set_point"]) + 1), 1/3), scalar_space.element.interpolation_points()),
    }

weak_form_defs = ff.setup_problem(mesh, facet_tags, params, problem_vars, kfr_growth_law)

data_collector = ff.DataCollector(comm, output_dir, line_points)
data_collector.register_function("u", problem_vars["u"])
data_collector.register_function("p", problem_vars["p"])
data_collector.register_function("Hoop Stress", problem_vars["cauchy_ff"])
data_collector.register_function("Radial Stress", problem_vars["cauchy_nn"])
data_collector.register_line_data("Radial Strain", problem_vars["strain_nn"])
data_collector.register_line_data("Hoop Strain", problem_vars["strain_ff"])
data_collector.register_line_data("Radial Stress", problem_vars["cauchy_nn"])
data_collector.register_line_data("Hoop Stress", problem_vars["cauchy_ff"])
data_collector.register_line_data("Elastic Hoop Strain", problem_vars["elastic_strain_ff"])
data_collector.register_line_data("Cumulative Radial Growth", problem_vars["g_r"])
data_collector.register_line_data("Cumulative Hoop Growth", problem_vars["g_t"])
data_collector.register_line_data("Incremental Hoop Growth", problem_vars["dgt"])
data_collector.register_line_data("Incremental Radial Growth", problem_vars["dgr"])
data_collector.register_line_data("u_r", problem_vars["u_r"])
data_collector.register_line_data("p", problem_vars["p"])
data_collector.setup_writers()

sims = ff.run_simulation(
    params, problem_vars, weak_form_defs, data_collector
    )

# --- Pre-calculate all data for plotting ---
print("--- Pre-calculating data for plots ---")
plot_data_1d = {
    "radial_stress": [], "hoop_stress": [], "radial_strain": [],
    "hoop_strain": [], "radial_growth": [], "hoop_growth": [], "displacement": [],
    "Ricci": [], "Elastic Hoop Strain": []
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
    plot_data_1d["Elastic Hoop Strain"].append(np.array([state.elastic_hoop_strain(ri_1d, s) for s in R_range]))

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
plotter_instance = plotter.ComparisonPlotter(R_range, num_steps, model_name="KFR")

# Plot spatial data
plotter_instance.plot_spatial_panel((0, 0), "Radial Stress (Cauchy)", "Stress", plot_data_1d["radial_stress"], plot_data_2d.get("Radial Stress"))
plotter_instance.plot_spatial_panel((1, 0), "Hoop Stress (Cauchy)", "Stress", plot_data_1d["hoop_stress"], plot_data_2d.get("Hoop Stress"))
plotter_instance.plot_spatial_panel((0, 1), "Radial Strain", "Strain", plot_data_1d["radial_strain"], plot_data_2d.get("Radial Strain"))
plotter_instance.plot_spatial_panel((1, 1), "Hoop Strain", "Strain", plot_data_1d["hoop_strain"], plot_data_2d.get("Hoop Strain"))
plotter_instance.plot_spatial_panel((0, 2), "Radial Growth", "Growth", plot_data_1d["radial_growth"], plot_data_2d.get("Cumulative Radial Growth"))
plotter_instance.plot_spatial_panel((1, 2), "Hoop Growth", "Growth", plot_data_1d["hoop_growth"], plot_data_2d.get("Cumulative Hoop Growth"))
plotter_instance.plot_spatial_panel((2, 0), "Displacement (r)", "Displacement", plot_data_1d["displacement"], plot_data_2d.get("u_r"))

# Plot special cases
plotter_instance.plot_ricci((2, 1), plot_data_1d["Ricci"])
plotter_instance.plot_spatial_panel((2, 2), "Elastic Hoop Strain", "Displacement", plot_data_1d["Elastic Hoop Strain"], plot_data_2d.get("Elastic Hoop Strain"), set_point=GL_set_point)

# Finalize and save
plotter_instance.finalize_and_save("1D_vs_2D_KFR.png")

data_collector.close()