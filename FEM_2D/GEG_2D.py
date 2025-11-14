import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
import ufl
import basix
import numpy as np
from collections import defaultdict
from pathlib import Path
import scifem
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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


class DataCollector:

    def __init__(self, comm: MPI.Comm, output_dir: Path, line_points: np.ndarray):
        self.comm = comm
        self.output_dir = output_dir
        self.line_points = line_points
        self.functions: dict[str, dolfinx.fem.Function] = {}
        self.history: dict[str, list[float]] = {}
        self.line_data: dict[str, dolfinx.fem.Function] = {}
        self.line_history: dict[str, list[np.ndarray]] = {}
        self.writers: list[dolfinx.io.VTXWriter] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def register_function(self, name: str, function: dolfinx.fem.Function):
        function.name = name
        self.functions[name] = function
        if name not in self.history:
            self.history[name] = []

    def register_line_data(self, name: str, function: dolfinx.fem.Function, set_point=None):
        """Register a function to collect data along the line points."""
        self.line_data[name] = function
        if name not in self.line_history:
            self.line_history[name] = []

    def setup_writers(self):
        sorted_functions = defaultdict(list)
        for f in self.functions.values():
            f_hash = f.function_space.element.basix_element.hash()
            sorted_functions[f_hash].append(f)

        n = 1
        for funcs in sorted_functions.values():
            filename = (
                self.output_dir / f"{funcs[0].name}.bp"
                if len(funcs) == 1
                else self.output_dir / f"variables_{n}.bp"
            )
            n += 1
            self.writers.append(
                dolfinx.io.VTXWriter(self.comm, filename, funcs, engine="BP4")
            )

    def write(self, t: float):
        for writer in self.writers:
            writer.write(t)

    def update_line_data(self):
        for name, function in self.line_data.items():
            values = scifem.evaluate_function(function, self.line_points)
            self.line_history[name].append(values)

    def close(self):
        for writer in self.writers:
            writer.close()

''' SETUP FUNCTION SPACES AND TRIAL FUNCTIONS '''
def setup_problem(mesh, params):
    
    #region
    QUAD_DEGREE = 8  # The number of points used in the quadrature scheme.

    # Create a second order Lagrange function space for displacement
    # This is basically the the function space to represent the tangent space without the base space
    P_u = basix.ufl.element(
        family="Lagrange",  # Type of functions (Lagrange polynomials)
        cell=str(mesh.ufl_cell()),  # Type of cell (e.g., triangle, square, etc.)
        degree=2,  # Polynomial degree of functions
        shape=(mesh.geometry.dim,),  # Dimension of functions (3D vector in this case)
    )

    # This is basically the union of the tangent space and the base space
    u_space = dolfinx.fem.functionspace(mesh, P_u)

    # Create a first order Lagrange function space for pressure
    P_p = basix.ufl.element(
        family="Lagrange",  # Type of functions (Lagrange polynomials)
        cell=str(mesh.ufl_cell()),  # Type of cell (e.g., triangle, square, etc.)
        degree=1,  # Polynomial degree of functions
        shape=(),  # Dimension of functions (scalar in this case)
    )

    # This creates a union of the base space and the tangent space
    p_space = dolfinx.fem.functionspace(mesh, P_p)

    u = dolfinx.fem.Function(u_space, name="u")
    v = ufl.TestFunction(u_space)
    du = ufl.TrialFunction(u_space)
    p = dolfinx.fem.Function(p_space, name="p")
    q = ufl.TestFunction(p_space)
    dp = ufl.TrialFunction(p_space)

    scalar_element = basix.ufl.element(
        family="CG",
        cell=str(mesh.ufl_cell()),
        degree=5,
        shape=(),
        discontinuous=True,
    )
    scalar_space = dolfinx.fem.functionspace(mesh, scalar_element)

    ''' CREATE FIBERS '''
    P_fiber = basix.ufl.element(
        family="CG",  # Type of functions (Lagrange polynomials)
        cell=str(mesh.ufl_cell()),  # Type of cell (e.g., triangle, square, etc.)
        degree=1,  # Polynomial degree of functions
        shape=(mesh.geometry.dim,),  # Dimension of functions (2D vector in this case)
    )

    fiber_space = dolfinx.fem.functionspace(mesh, P_fiber)

    x, y, z = fiber_space.tabulate_dof_coordinates().T
    r = np.sqrt(x**2 + y**2)
    e_r = np.array([x / r, y / r])
    e_theta = np.array([-e_r[1], e_r[0]])  # 90 degree rotation of e_r


    f0 = dolfinx.fem.Function(fiber_space, name="f0")
    r0 = dolfinx.fem.Function(fiber_space, name="r0")

    f0.x.array[:] = np.array(e_theta).T.reshape(-1)
    r0.x.array[:] = np.array(e_r).T.reshape(-1)

    stress_ff = dolfinx.fem.Function(scalar_space, name="stress_ff")
    stress_nn = dolfinx.fem.Function(scalar_space, name="stress_nn")
    strain_ff = dolfinx.fem.Function(scalar_space, name="strain_ff")
    J = dolfinx.fem.Function(scalar_space, name="J")

    g_r = dolfinx.fem.Function(scalar_space, name="g_r")
    g_t = dolfinx.fem.Function(scalar_space, name="g_t")
    dgt = dolfinx.fem.Function(scalar_space, name="dgt")
    # initial_gt linear profile from 1 to 1.5
    # Set g_t to have a linear profile from 1.0 at the inner radius to 1.5 at the outer radius
    x_coords = ufl.SpatialCoordinate(mesh)
    r_coord = ufl.sqrt(x_coords[0]**2 + x_coords[1]**2)
    R_i = params["R_i"]
    R_o = params["R_o"]
    
    # Linear profile: g_t(r) = 1.0 + 0.5 * (r - R_i) / (R_o - R_i)
    linear_profile = 1.0 + 0.1 * (r_coord - R_i)
    
    initial_expression = dolfinx.fem.Expression(
        linear_profile,
        scalar_space.element.interpolation_points(),
    )
    # g_t.interpolate(initial_expression)
    g_r.x.array[:] = params["g_r"]
    # g_t.interpolate(initial_expression)
    g_t.x.array[:] = params["g_t"]


    problem_variables = {
        "u": u,
        "v": v,
        "du": du,
        "p": p,
        "q": q,
        "dp": dp,
        "f0": f0,
        "r0": r0,
        "stress_ff": stress_ff,
        "strain_ff": strain_ff,
        "stress_nn": stress_nn,
        "J": J,
        "scalar_space": scalar_space,
        "g_r": g_r,
        "g_t": g_t,
        "dgt": dgt,
    }

    return problem_variables


def weak_formulation(mesh, facet_tags, params, problem_variables, QUAD_DEGREE=8):
    u = problem_variables["u"]
    v = problem_variables["v"]
    du = problem_variables["du"]
    p = problem_variables["p"]
    q = problem_variables["q"]
    dp = problem_variables["dp"]
    f0 = problem_variables["f0"]
    r0 = problem_variables["r0"]
    stress_ff = problem_variables["stress_ff"]
    stress_nn = problem_variables["stress_nn"]
    strain_ff = problem_variables["strain_ff"]
    J = problem_variables["J"]
    g_r = problem_variables["g_r"]
    g_t = problem_variables["g_t"]
    dgt = problem_variables["dgt"]
    scalar_space = problem_variables["scalar_space"]
    
    ''' KINEMATICS '''
    #region
    F = ufl.variable(ufl.grad(u) + ufl.Identity(2))
    G = g_r * ufl.outer(r0, r0) + g_t * ufl.outer(f0, f0)
    A = ufl.variable(F * ufl.inv(G))
    #endregion
    
    ''' MATERIAL MODEL '''
    #region
    mu = 1.0  # Shear modulus
    C = A.T * A  # Right Cauchy-Green deformation tensor
    I1 = ufl.tr(C)  # First invariant of the right Cauchy-Green tensor
    psi = (mu / 2) * (I1 - 2)  # Neo-Hookean strain energy function # / 2.0
    stress = ufl.diff(psi, F)

    dx = ufl.dx(metadata={"quadrature_degree": QUAD_DEGREE})

    pressure_term = p * (ufl.det(A) - 1) * dx
    elasticity_term = ufl.inner(stress, ufl.grad(v)) * dx
    # cauchy = (stress + p * ufl.inv(F.T)) * F.T / ufl.det(F)  # This is cauchy stress because stress = dPsi/dF * 
    cauchy = stress*F.T / ufl.det(F) + ufl.Identity(2) * p / ufl.det(F)  # This is cauchy stress because stress = dPsi/dF * 
    #endregion

    ''' BOUNDARY CONDITIONS '''
    #region
    N = ufl.FacetNormal(mesh)  # Normal vector on the boundary of the mesh

    ds = ufl.Measure(
        "ds",  # ???
        domain=mesh,  # Domain of the measure
        subdomain_data=facet_tags,  # Boundary we are interested in
        metadata={"quadrature_degree": QUAD_DEGREE},  # Quadrature degree for the measure
    )

    # Pressure on the inside (Neumann)
    traction = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(-params["c"]))
    # Pressure value on the inside of the cylinder
    # Neumann boundary condition on the inside surface (pulling back the surface element)
    inner_neumann = ufl.inner(v, traction * ufl.det(F) * ufl.inv(F).T * N) * ds(20)

    # Robin on the outside
    N = ufl.FacetNormal(mesh)
    spring = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0001))
    robin_value = ufl.inner(spring * u, N)
    outer_robin = ufl.inner(robin_value * v, ufl.det(F) * ufl.inv(F).T * N) * ds(10)


    dx = ufl.dx(metadata={"quadrature_degree": QUAD_DEGREE})
    #endregion

    expressions = {
        "stress_ff_expr": dolfinx.fem.Expression(  # hoop stress
        ufl.inner(cauchy * f0, f0),
        scalar_space.element.interpolation_points(),
        ),

        "strain_ff_expr": dolfinx.fem.Expression(  # hoop strain
        ufl.inner(A * f0, f0),
        scalar_space.element.interpolation_points(),
        ),

        "stress_nn_expr": dolfinx.fem.Expression(
            ufl.inner(cauchy * r0, r0),
            scalar_space.element.interpolation_points(),
        ),

        "J_expr": dolfinx.fem.Expression(ufl.det(A), scalar_space.element.interpolation_points()),

        "dgt_expr": dolfinx.fem.Expression(params["dt"] * (strain_ff - params["set_point"]), scalar_space.element.interpolation_points()),

        "g2_expr": dolfinx.fem.Expression(g_t + params["dt"] * (strain_ff - params["set_point"]), scalar_space.element.interpolation_points()),
    }

    F0 = (
        elasticity_term + ufl.derivative(pressure_term, u, v) + inner_neumann + outer_robin
    )  #  ufl.derivative(rigid_form, u, v) +ufl.derivative(psi * dx, u, v)

    F1 = ufl.derivative(psi * dx, p, q) + ufl.derivative(
        pressure_term,
        p,
        q,
    )

    R = [F0, F1]  # , F2]
    dR = [
        [ufl.derivative(F0, u, du), ufl.derivative(F0, p, dp)],
        [ufl.derivative(F1, u, du), ufl.derivative(F1, p, dp)],
    ]

    return R, dR, expressions

def run_simulation(params, problem_variables, weak_form_defs, data_collector):

    R, dR, expressions = weak_form_defs
    u, p = problem_variables["u"], problem_variables["p"]
    g_r = problem_variables["g_r"]
    g_t = problem_variables["g_t"]
    stress_ff = problem_variables["stress_ff"]
    stress_nn = problem_variables["stress_nn"]
    strain_ff = problem_variables["strain_ff"]
    dgt = problem_variables["dgt"]
    J = problem_variables["J"]

    petsc_options = {
        "ksp_type": "preonly",  # direct solver
        "pc_type": "lu",  # LU preconditioner
        "pc_factor_mat_solver_type": "mumps",  # paralellization
        "ksp_monitor": None,  # see output during solve
    }

    solver = scifem.NewtonSolver(
        R,
        dR,
        [u, p],
        bcs=[],
        max_iterations=25,
        petsc_options=petsc_options,
    )
    
    time_steps = range(params["num_steps"])
    time = [params["dt"] * i for i in time_steps]

    solver.solve()
    stress_ff.interpolate(expressions["stress_ff_expr"])
    stress_nn.interpolate(expressions["stress_nn_expr"])
    strain_ff.interpolate(expressions["strain_ff_expr"])
    J.interpolate(expressions["J_expr"])
    dgt.interpolate(expressions["dgt_expr"])

    print("updating line data")
    data_collector.update_line_data()
    # data_collector.write(t=time[0])

    for i in time_steps:

        g_t.interpolate(expressions["g2_expr"])

        print(f"Starting growth step {i + 1} of {params['num_steps']}")

        # print g2 values
        print("g_r min/max:", g_r.x.array.min(), g_r.x.array.max())
        print("g_t min/max:", g_t.x.array.min(), g_t.x.array.max())
        solver.solve()

        stress_ff.interpolate(expressions["stress_ff_expr"])
        stress_nn.interpolate(expressions["stress_nn_expr"])
        strain_ff.interpolate(expressions["strain_ff_expr"])
        J.interpolate(expressions["J_expr"])
        # u_mag.interpolate(u_mag_expr)

        dgt.interpolate(expressions["dgt_expr"])
        print("updating line data")
        data_collector.update_line_data()
        data_collector.write(t=time[i])

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

    problem_vars = setup_problem(mesh, params)
    weak_form_defs = weak_formulation(mesh, facet_tags, params, problem_vars)

    data_collector = DataCollector(comm, output_dir, line_points)
    data_collector.register_function("u", problem_vars["u"])
    data_collector.register_function("p", problem_vars["p"])
    data_collector.register_function("Hoop Stress", problem_vars["stress_ff"])
    data_collector.register_function("Radial Stress", problem_vars["stress_nn"])
    data_collector.register_function("Hoop Strain", problem_vars["strain_ff"])
    data_collector.register_line_data("Hoop Stress", problem_vars["stress_ff"])
    data_collector.register_line_data("Radial Stress", problem_vars["stress_nn"])
    data_collector.register_line_data("Hoop Strain", problem_vars["strain_ff"])
    data_collector.register_line_data("Cumulative Hoop Growth", problem_vars["g_t"])
    data_collector.register_line_data("p", problem_vars["p"])
    data_collector.setup_writers()

    time = run_simulation(
        params, problem_vars, weak_form_defs, data_collector
        )
    
    data_collector.close()

    breakpoint()

if __name__ == "__main__":
    main()