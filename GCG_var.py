import FEM_2D.fem_framework as ff
from cylinder_functions import BaseState
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI
from pathlib import Path
from scipy import interpolate
import saver

''' 1D Solution '''
#region
# Initialize base state
R_range = np.linspace(1.0, 2.0, 64)  # 64 points from 1 to 2 with step 1/63
# initial_gt linear profile from 1 to 1.5
initial_gt = np.ones_like(R_range)
initial_gr = np.ones_like(R_range)  # No initial growth
# initial_gt = np.ones_like(R_range)

class GCGState(BaseState):
    
    def compute_dgt(self, ri, s):
        """Compute growth rate based on circumferential stress."""
        
        mandel_trace = self.mandel_trace(ri, s)

        dgt = self.tau * (mandel_trace - self.set_point(s)) * (self.gMax - self.gt_interp(s)) / (self.gMax - 1)

        return dgt
    
    def compute_dgr(self, ri, s):
        """Compute growth rate based on circumferential stress."""
        return 0.0
    
    # Need to overwrite update because the GCG model is additative
    def update(self):
        """Create updated state"""
        ri = self.find_inner_radius()

        # Vectorized dgt computation
        dgt = np.array([self.compute_dgt(ri, s) for s in self.R])
        new_gt = self.gt + dgt
        dgr = np.array([self.compute_dgr(ri, s) for s in self.R])
        new_gr = self.gr + dgr

        return self.__class__(
            self.R, new_gr, new_gt, self.bc, self.mu,
            self.gMax, self.set_point, self.gamma, self.tau
        )
    
dt = 0.001
mu= 1.0
stretch_set_point = 1.1
stress_set_point = stretch_set_point**2 - 0.05/2
gMax = 0.5
print("Stress set point:", stress_set_point)

init_state = GCGState(
    R=R_range,
    gr=initial_gr,
    gt=initial_gt,
    bc=-0.1,
    mu=mu,
    gMax=gMax,
    set_point=0,
    gamma=1,
    tau=dt
)

# Initial calculations
ri = init_state.find_inner_radius()
stress_data = [init_state.angular_stress(ri, x) for x in R_range]
#interpolate stress data to get set point
stress_set_point = interpolate.interp1d(
        R_range, stress_data, kind='cubic',
        bounds_error=False, fill_value='extrapolate'
        )

base_state = GCGState(
    R=R_range,
    gr=initial_gr,
    gt=initial_gt,
    bc=-0.05,
    mu=mu,
    gMax=gMax,
    set_point=stress_set_point,
    gamma=1,
    tau=dt
)

# Initial calculations
ri = base_state.find_inner_radius()
stress_points = np.arange(1, 2.1, 0.1)
stress_data = [base_state.radial_stress(ri, x) for x in stress_points]

states = [base_state]
prev_state = base_state
num_steps = 8196
for step in range(1, num_steps + 1):  # 2 time steps
    if step % 100 == 0:
        print(f"Time step {step}")
        print("gr min/max:", prev_state.gr.min(), prev_state.gr.max())
        print("gt min/max:", prev_state.gt.min(), prev_state.gt.max())
    next_state = prev_state.update()
    states.append(next_state)
    prev_state = next_state
#endregion

# --- Pre-calculate all data for plotting ---
print("--- Pre-calculating data for plots ---")
plot_data_1d = {
    "radial_stress": [], "hoop_stress": [], "radial_strain": [],
    "hoop_strain": [], "radial_growth": [], "hoop_growth": [], "displacement": [],
    "Ricci": [], "Mandel Trace": []
}
power_data = {"power": [], "entropy": [], "internal_entropy": []}

# Calculate data for each state
number_of_lines = 8
states_to_plot_idx = np.linspace(0, num_steps, num=number_of_lines, dtype=int)
states_to_plot_1d = [states[i] for i in states_to_plot_idx]

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
    plot_data_1d["Mandel Trace"].append(np.array([state.mandel_trace(ri_1d, s) for s in R_range]))

# Calculate data between states (power)
for i in range(number_of_lines - 1):
    state1 = states_to_plot_1d[i]
    state2 = states_to_plot_1d[i+1]
    power_direct = BaseState.power_direct(state1, state2, R_range, dt)
    power_data["power"].append(power_direct)
    entropy = BaseState.entropy(state1, state2, R_range, dt)
    power_data["internal_entropy"].append(entropy)
    power_data["entropy"].append(power_direct - entropy)
print("--- Plotting results ---")

data = {
    "plot_data_1d": plot_data_1d,
    "power_data": power_data,
    "R_range": R_range.tolist(),
    "dt": dt,
    "number_of_lines": number_of_lines,
    "stress_set_point": stress_set_point(R_range).tolist(),
    "gMax": gMax
}

saver.save_data(data, "GCG_var_ODE_data.json")

# # --- Use the Plotter Class ---
# plotter_instance = plotter.ComparisonPlotter(R_range, num_steps, model_name="GCG")

# # Plot spatial data
# plotter_instance.plot_spatial_panel((0, 0), "Radial Stress (Cauchy)", "Stress", plot_data_1d["radial_stress"])
# plotter_instance.plot_spatial_panel((1, 0), "Hoop Stress (Cauchy)", "Stress", plot_data_1d["hoop_stress"])
# plotter_instance.plot_spatial_panel((0, 1), "Radial Strain", "Strain", plot_data_1d["radial_strain"])
# plotter_instance.plot_spatial_panel((1, 1), "Hoop Strain", "Strain", plot_data_1d["hoop_strain"])
# plotter_instance.plot_spatial_panel((0, 2), "Radial Growth", "Growth", plot_data_1d["radial_growth"])
# plotter_instance.plot_spatial_panel((1, 2), "Hoop Growth", "Growth", plot_data_1d["hoop_growth"], set_point=gMax)
# plotter_instance.plot_spatial_panel((2, 0), "Displacement (r)", "Displacement", plot_data_1d["displacement"])

# # Plot special cases
# plotter_instance.plot_spatial_panel((2, 1), "Mandel Trace (r)", "Stress", plot_data_1d["Mandel Trace"], set_point=stress_set_point)
# plotter_instance.plot_dissipation((2, 2), power_data, dt)

# # Finalize and save
# plotter_instance.finalize_and_save("ODE_GCG_results.png")