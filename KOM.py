import FEM_2D.fem_framework as ff
from cylinder_functions import BaseState
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI
from pathlib import Path
import plotter
import saver

''' 1D Solution '''
#region
# Initialize base state
R_range = np.linspace(1.0, 2.0, 64)
initial_gt = np.ones_like(R_range)
initial_gr = np.ones_like(R_range)

dt = 0.01
mu = 1.0
stretch_set_point = 1.1 
# Convert stretch set point to Green-Lagrange strain for KOM model
# s_set = 0.5 * (lambda^2 - 1)
E_set_point = 0.5 * (stretch_set_point**2 - 1)

class KOMState(BaseState):
    """
    Implementation of the Kerckhoffs-Omens-McCulloch (KOM) growth law.
    Reference: Kerckhoffs et al. (2012)
    """
    def __init__(self, R, gr, gt, bc, mu, gMax, set_point, gamma, tau, params=None):
        super().__init__(R, gr, gt, bc, mu, gMax, set_point, gamma, tau)
        
        self.f_cc_max = 0.1  #f_cc,max in paper - Max fiber growth rate
        self.f_ff_max = 0.3  #f_ff,max in paper - Max radial growth rate
        
        self.f_f = 150.0 # f_f in paper - Slope affecting sigmoid for fiber growth
        self.c_c = 75.0  # c_f in paper - Slope affecting sigmoid for radial growth

        self.f_slope = 40.0 # f_length,slope in paper    - Slope affecting sigmoid for *total* fiber growth
        self.c_slope = 60.0 # c_thickness,slope in paper - Slope affecting sigmoif for *total* radial growth
        
        self.sf_setpoint = 0.06 #0.06 # Stimulus at 50% max axial growth
        self.sc_setpoint = 0.07 #0.07 # Stimulus at 50% max radial growth

    def growth_term(self, growth, slope):
        """
        Slope function to adjust steepness based on current growth state.
        """
        return 1 / (1.0 + np.exp(slope * (growth - self.gMax)))
    
    def strain_term(self, height, slope, s, s_50):
        """
        Generic sigmoid function based on Eq 8 and 9
        g_inc = A / (1 + exp(-slope * (s - s50))) + 1
        """
        
        if s >= 0:
            return height / (1.0 + np.exp(-slope * (s - s_50)))
        elif s < 0:
            return -height / (1.0 + np.exp(slope * (s + s_50)))

    def sf(self, ri, s):
        E_ff = self.elastic_hoop_strain(ri, s)
        
        stimulus_l = E_ff - self.set_point
        return stimulus_l
    
    def sr(self, ri, s):
        E_rr = self.elastic_radial_strain(ri, s)
        E_zz = 0.0 # Plane strain assumption
        
        E_cross_max = max(E_rr, E_zz)
        stimulus_t = E_cross_max - self.set_point
        return stimulus_t

    def compute_dgt(self, ri, s):
        """
        Fiber (Hoop) Growth.
        Stimulus: s_l = max(E_ff) - E_set   
        """

        stimulus_f = self.sf(ri, s)
        
        _growth_term = self.growth_term(self.gt_interp(s), self.f_slope)
        _strain_term = self.strain_term(self.f_ff_max, self.f_f, stimulus_f, self.sf_setpoint)
        
        return self.tau * _growth_term * _strain_term + 1

    def compute_dgr(self, ri, s):
        """
        Cross-Fiber (Radial) Growth.
        Stimulus: s_t = max(E_rr, E_zz) - E_set   
        """

        stimulus_r = self.sr(ri, s)
        
        _growth_term = self.growth_term(self.gr_interp(s), self.c_slope)
        _strain_term = self.strain_term(self.f_cc_max, self.c_c, stimulus_r, self.sc_setpoint)

        return np.sqrt(self.tau * _growth_term * _strain_term + 1)
    
    def update(self):
        ri = self.find_inner_radius()

        F_inc_t = np.array([self.compute_dgt(ri, s) for s in self.R])
        new_gt = self.gt * F_inc_t
        
        F_inc_r = np.array([self.compute_dgr(ri, s) for s in self.R])
        new_gr = self.gr * F_inc_r

        return self.__class__(
            self.R, new_gr, new_gt, self.bc, self.mu,
            self.gMax, self.set_point, self.gamma, self.tau
        )

print("Using Green-Lagrange set point:", E_set_point)
base_state = KOMState(
    R=R_range,
    gr=initial_gr,
    gt=initial_gt,
    bc=-0.05,
    mu=mu,
    gMax=1.5, # In the other sims this is 0.5, but this is changed due to the sigmoid
    set_point=E_set_point,
    gamma=1,
    tau=dt
)

print("Initial state:", base_state)

states = [base_state]
prev_state = base_state
num_steps = 200 #32768
for step in range(1, num_steps + 1):
    if step % 10 == 0:
        print(f"Time step {step}")
    next_state = prev_state.update()
    states.append(next_state)
    prev_state = next_state
#endregion

# --- Pre-calculate all data for plotting ---
print("--- Pre-calculating data for plots ---")
plot_data_1d = {
    "radial_stress": [], "hoop_stress": [], "radial_strain": [],
    "hoop_strain": [], "radial_growth": [], "hoop_growth": [], "displacement": [],
    "Ricci": [], "sr": [], "sf": []
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
    plot_data_1d["sr"].append(np.array([state.sr(ri_1d, s) for s in R_range]))
    plot_data_1d["sf"].append(np.array([state.sf(ri_1d, s) for s in R_range]))

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
    "num_steps": num_steps,
    "number_of_lines": number_of_lines,
    "set_point": stretch_set_point,
    }

saver.save_data(data, "KOM_ODE_data.json")

# # --- Use the Plotter Class ---
# plotter_instance = plotter.ComparisonPlotter(R_range, num_steps, model_name="KOM")

# # Plot spatial data
# plotter_instance.plot_spatial_panel((0, 0), "Radial Stress (Cauchy)", "Stress", plot_data_1d["radial_stress"])
# plotter_instance.plot_spatial_panel((1, 0), "Hoop Stress (Cauchy)", "Stress", plot_data_1d["hoop_stress"])
# plotter_instance.plot_spatial_panel((0, 1), "Radial Strain", "Strain", plot_data_1d["radial_strain"])
# plotter_instance.plot_spatial_panel((1, 1), "Hoop Strain", "Strain", plot_data_1d["hoop_strain"])
# plotter_instance.plot_spatial_panel((0, 2), "Radial Growth", "Growth", plot_data_1d["radial_growth"])
# plotter_instance.plot_spatial_panel((1, 2), "Hoop Growth", "Growth", plot_data_1d["hoop_growth"])
# plotter_instance.plot_spatial_panel((2, 0), "Displacement (r)", "Displacement", plot_data_1d["displacement"])

# # Plot special cases
# plotter_instance.plot_spatial_panel((2, 1), "Stimulus L", "Stimulus", plot_data_1d["sl"], set_point=E_set_point)
# plotter_instance.plot_spatial_panel((2, 2), "Stimulus T", "Stimulus", plot_data_1d["st"], set_point=E_set_point)

# # Finalize and save
# plotter_instance.finalize_and_save("ODE_KOM_results.png")
