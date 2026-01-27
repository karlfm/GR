import numpy as np
from cylinder_functions import BaseState
import time
import matplotlib.pyplot as plt
import plotter
import saver
from scipy import interpolate

class LT2State(BaseState):
    
    def compute_dgt(self, ri, s):

        stress_term = self.hoop_cauchy(ri, s)
        dgt = self.tau * (stress_term - self.set_point(s)) / self.set_point(s) + 1

        return dgt

    def compute_dgr(self, ri, s):
        return 1.0

    def compute_homeostasis(self, ri, s):
        """
        Compute homeostatic growth based on the set point stress.
        Solves the cubic equation for g_theta derived from:
        g_theta^3 * sigma = mu * (r/R)^2 + p * g_theta^2
        """
        sigma = self.set_point(s)
        if abs(sigma) < 1e-10:
            return 0.0
            
        r = self.compute_r(ri, s)
        R = s
        p = self.compute_p(ri, s)
        mu = self.mu

        # Terms for S_pm
        term1 = 2 * p**3 * R**2
        term2 = 27 * mu * r**2 * sigma**2
        
        # Discriminant part
        # 27 * mu * r^2 * sigma^2 * (4 * p^3 * R^2 + 27 * mu * r^2 * sigma^2)
        inner_bracket = 4 * p**3 * R**2 + 27 * mu * r**2 * sigma**2
        discriminant_val = 27 * mu * r**2 * sigma**2 * inner_bracket
        
        # Use complex sqrt to handle negative discriminants
        sqrt_disc = np.sqrt(discriminant_val + 0j)
        
        numerator_plus = term1 + term2 + sqrt_disc
        numerator_minus = term1 + term2 - sqrt_disc
        
        denominator = 54 * R**2 * sigma**3
        
        S_plus = numerator_plus / denominator
        S_minus = numerator_minus / denominator
        
        # Calculate cube roots
        g_theta = p / (3 * sigma) + np.power(S_plus, 1/3) + np.power(S_minus, 1/3)
        
        return np.real(g_theta)

def main():
    R_range = np.arange(1, 2 + 1/64, 1/64)
    # Initialize base state
    initial_gr = np.ones_like(R_range)  # No initial growth
    initial_gt = np.ones_like(R_range)

    dt = 0.001
    # stress_set_point = 0.5
    init_set_point = interpolate.interp1d(
            R_range, np.zeros_like(R_range), kind='cubic',
            bounds_error=False, fill_value='extrapolate'
        )
    
    init_state = LT2State(
        R=R_range,
        gr=initial_gr,
        gt=initial_gt,
        bc=-0.025,
        mu=1.0,
        gMax=0,
        set_point=init_set_point,
        gamma=1,
        tau=dt,
        flow_rate = 1.0,
        viscosity_const = 0.05
    )

    # Initial calculations
    ri = init_state.find_inner_radius()
    stress_points = np.arange(1, 2.1, 0.1)
    stress_data = [init_state.angular_stress(ri, x) for x in R_range]
    #interpolate stress data to get set point
    stress_set_point = interpolate.interp1d(
            R_range, stress_data, kind='cubic',
            bounds_error=False, fill_value='extrapolate'
        )

    base_state = LT2State(
        R=R_range,
        gr=initial_gr,
        gt=initial_gt,
        bc=-0.05,
        mu=1.0,
        gMax=1.5,
        set_point=stress_set_point,
        gamma=1,
        tau=dt,
        flow_rate = 1.0,
        viscosity_const = 0.05

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
    
    num_steps = 300 #1500
    for step in range(1, num_steps + 1):  # 2 time steps
        print(f"  Iteration {step}/{num_steps}", end="", flush=True)
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
    

    # --- Pre-calculate all data for plotting ---
    print("--- Pre-calculating data for plots ---")
    plot_data_1d = {
        "radial_stress": [], "hoop_stress": [], "radial_strain": [],
        "hoop_strain": [], "radial_growth": [], "hoop_growth": [], "displacement": [],
        "Ricci": [], "Homeostasis": []
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
        plot_data_1d["Homeostasis"].append(np.array([state.compute_homeostasis(ri_1d, s) for s in R_range]))


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
    
    data = {
        "plot_data_1d": plot_data_1d,
        "power_data": power_data,
        "R_range": R_range.tolist(),
        "dt": dt,
        "number_of_lines": number_of_lines,
        "stress_set_point": stress_set_point(R_range).tolist()
    }

    saver.save_data(data, "LT2_var_ODE_data.json")
    
    # print("--- Plotting results ---")

    # # --- Use the Plotter Class ---
    # plotter_instance = plotter.ComparisonPlotter(R_range, number_of_lines, model_name="LT2")

    # # Plot spatial data
    # plotter_instance.plot_spatial_panel((0, 0), "Radial Stress (Cauchy)", "Stress", plot_data_1d["radial_stress"])
    # plotter_instance.plot_spatial_panel((1, 0), "Hoop Stress (Cauchy)", "Stress", plot_data_1d["hoop_stress"], set_point=stress_set_point)
    # plotter_instance.plot_spatial_panel((0, 1), "Radial Strain", "Strain", plot_data_1d["radial_strain"])
    # plotter_instance.plot_spatial_panel((1, 1), "Hoop Strain", "Strain", plot_data_1d["hoop_strain"])
    # plotter_instance.plot_spatial_panel((0, 2), "Radial Growth", "Growth", plot_data_1d["radial_growth"])
    # plotter_instance.plot_spatial_panel((1, 2), "Hoop Growth", "Growth", plot_data_1d["hoop_growth"], plot_data_1d["Homeostasis"])
    # plotter_instance.plot_spatial_panel((2, 0), "Displacement (r)", "Displacement", plot_data_1d["displacement"])

    # # Plot special cases
    # plotter_instance.plot_ricci((2, 1), plot_data_1d["Ricci"])
    # plotter_instance.plot_dissipation((2, 2), power_data, dt)

    # # Finalize and save
    # plotter_instance.finalize_and_save("ODE_LT2_results.png")

    # # plot_and_save(states[:-1], R_range, time=time_points)

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"\n✓ Total execution time: {total_time:.2f} seconds")

    