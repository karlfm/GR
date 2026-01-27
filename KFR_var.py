import numpy as np
from cylinder_functions import BaseState
import time
from scipy import interpolate
import saver

def main():
    # Initialize base state
    # Initialize base state
    R_range = np.linspace(1.0, 2.0, 64)  # 64 points from 1 to 2 with step 1/63
    initial_gt = np.ones_like(R_range)
    initial_gr = np.ones_like(R_range)  # No initial growth

    class KFRState(BaseState):

        def compute_dgt(self, ri, s):
            """Compute growth rate based on circumferential stress."""
            
            elastic_strain = self.elastic_hoop_strain(ri, s)

            dg = (self.tau*(np.sqrt(2 * elastic_strain + 1) - 1 - self.set_point(s)) + 1)**(1/3)

            return dg
        
        def compute_dgr(self, ri, s):
            """Compute growth rate based on circumferential stress."""
            
            elastic_strain = self.elastic_hoop_strain(ri, s)

            dg = (self.tau*(np.sqrt(2 * elastic_strain + 1) - 1 - self.set_point(s)) + 1)**(1/3)

            return dg
        
        def compute_homeostasis(self, ri, s):
            """Compute homeostatic growth based on circumferential stress."""
            
            # \frac{r}{R(1+s_\mathrm{hom})}

            r = self.compute_r(ri, s)
            g_homeo = r / (s * (1 + self.set_point(s)))

            return g_homeo


    dt = 0.01
    mu=1.0
    strain_set_point = 1.1
    #Green-Lagrange set point
    init_set_point = 0.5*(strain_set_point**2 - 1)

    print("Using Green-Lagrange set point:", init_set_point)

    init_state = KFRState(
        R=R_range,
        gr=initial_gr,
        gt=initial_gt,
        bc=-0.1,
        mu=mu,
        gMax=1.5,
        set_point=0,
        gamma=1,
        tau=dt
    )

    # Initial calculations
    ri = init_state.find_inner_radius()
    strain_data = np.array([init_state.hoop_strain(ri, x) for x in R_range])
    #interpolate stress data to get set point
    GL_set_point = interpolate.interp1d(
        R_range, (strain_data**2 - 1)/2, kind='cubic',
        bounds_error=False, fill_value='extrapolate'
        )
    
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
    
    print(f"\nInitial inner radius: {ri:.15f}")
    print("Initial stress data:")
    print("[" + ",".join(f"{x:.15f}" for x in stress_data) + "]")
    
    # Iterate state updates
    print("\nIterating states...")
    states = [base_state]
    current_state = base_state
    
    num_steps = 16000
    for step in range(1, num_steps + 1):  # 2 time steps
        if step % 100 == 0:
            print(f"  Iteration {step}/{num_steps   }", end="", flush=True)
        start = time.time()
        current_state = current_state.update()
        states.append(current_state)
        # print(f" - {time.time()-start:.3f}s")
    
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
        "Ricci": [], "Elastic Hoop Strain": [], "Homeostasis": []
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
        plot_data_1d["Elastic Hoop Strain"].append(np.array([state.elastic_hoop_strain(ri_1d, s) for s in R_range]))
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
    "num_steps": num_steps,
    "number_of_lines": number_of_lines,
    "set_point": GL_set_point(R_range).tolist(),
    }

    saver.save_data(data, "KFR_var_ODE_data.json")

    
    # # --- Use the Plotter Class ---
    # plotter_instance = plotter.ComparisonPlotter(R_range, num_steps, model_name="KFR")

    # # Plot spatial data
    # plotter_instance.plot_spatial_panel((0, 0), "Radial Stress (Cauchy)", "Stress"      , plot_data_1d["radial_stress"])
    # plotter_instance.plot_spatial_panel((1, 0), "Hoop Stress (Cauchy)"  , "Stress"      , plot_data_1d["hoop_stress"])
    # plotter_instance.plot_spatial_panel((0, 1), "Radial Strain"         , "Strain"      , plot_data_1d["radial_strain"])
    # plotter_instance.plot_spatial_panel((1, 1), "Hoop Strain"           , "Strain"      , plot_data_1d["hoop_strain"])
    # plotter_instance.plot_spatial_panel((0, 2), "Radial Growth"         , "Growth"      , plot_data_1d["radial_growth"])
    # plotter_instance.plot_spatial_panel((1, 2), "Hoop Growth"           , "Growth"      , plot_data_1d["hoop_growth"], plot_data_1d["Homeostasis"])
    # plotter_instance.plot_spatial_panel((2, 0), "Displacement (r)"      , "Displacement", plot_data_1d["displacement"])

    # # Plot special cases
    # plotter_instance.plot_dissipation((2, 2), power_data, dt)
    # plotter_instance.plot_spatial_panel((2, 1), "Elastic Hoop Strain", "Displacement", plot_data_1d["Elastic Hoop Strain"], set_point=GL_set_point)

    # plotter_instance.finalize_and_save("ODE_KFR_results.png")

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"\n✓ Total execution time: {total_time:.2f} seconds")
