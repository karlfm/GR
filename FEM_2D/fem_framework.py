import dolfinx
import ufl
import basix
import numpy as np
from mpi4py import MPI
from pathlib import Path
from collections import defaultdict
import scifem

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

def setup_common_variables(mesh, params):
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

    u_r = dolfinx.fem.Function(scalar_space, name="u_r")
    f0 = dolfinx.fem.Function(fiber_space, name="f0")
    r0 = dolfinx.fem.Function(fiber_space, name="r0")

    f0.x.array[:] = np.array(e_theta).T.reshape(-1)
    r0.x.array[:] = np.array(e_r).T.reshape(-1)

    elastic_strain_ff = dolfinx.fem.Function(scalar_space, name="elastic_strain_ff")
    elastic_strain_rr = dolfinx.fem.Function(scalar_space, name="elastic_strain_rr")
    strain_ff = dolfinx.fem.Function(scalar_space, name="strain_ff")
    strain_nn = dolfinx.fem.Function(scalar_space, name="strain_nn")
    cauchy_ff = dolfinx.fem.Function(scalar_space, name="cauchy_ff")
    cauchy_nn = dolfinx.fem.Function(scalar_space, name="cauchy_nn")
    J = dolfinx.fem.Function(scalar_space, name="J")

    g_r = dolfinx.fem.Function(scalar_space, name="g_r")
    g_t = dolfinx.fem.Function(scalar_space, name="g_t")
    dgr = dolfinx.fem.Function(scalar_space, name="dgr")
    dgt = dolfinx.fem.Function(scalar_space, name="dgt")
    # initial_gt linear profile from 1 to 1.5
    # Set g_t to have a linear profile from 1.0 at the inner radius to 1.5 at the outer radius
    x_coords = ufl.SpatialCoordinate(mesh)
    
    # g_t.interpolate(initial_expression)
    g_r.x.array[:] = params["g_r"]
    # g_t.interpolate(initial_expression)
    g_t.x.array[:] = params["g_t"]


    problem_variables = {
        "r": x_coords,
        "u_r": u_r,
        "u": u,
        "v": v,
        "du": du,
        "p": p,
        "q": q,
        "dp": dp,
        "f0": f0,
        "r0": r0,
        "elastic_strain_ff": elastic_strain_ff,
        "elastic_strain_rr": elastic_strain_rr,
        "strain_ff": strain_ff,
        "strain_nn": strain_nn,
        "cauchy_ff": cauchy_ff,
        "cauchy_nn": cauchy_nn,
        "J": J,
        "scalar_space": scalar_space,
        "g_r": g_r,
        "g_t": g_t,
        "dgt": dgt,
        "dgr": dgr
    }

    return problem_variables

def setup_problem(mesh, facet_tags, params, problem_variables, growth_laws, QUAD_DEGREE=8):
    r = problem_variables["r"]
    u = problem_variables["u"]
    v = problem_variables["v"]
    du = problem_variables["du"]
    p = problem_variables["p"]
    q = problem_variables["q"]
    dp = problem_variables["dp"]
    f0 = problem_variables["f0"]
    r0 = problem_variables["r0"]
    cauchy_ff = problem_variables["cauchy_ff"]
    cauchy_nn = problem_variables["cauchy_nn"]
    elastic_strain_rr = problem_variables["elastic_strain_rr"]
    elastic_strain_ff = problem_variables["elastic_strain_ff"]
    strain_ff = problem_variables["strain_ff"]
    strain_nn = problem_variables["strain_nn"]
    g_r = problem_variables["g_r"]
    g_t = problem_variables["g_t"]
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
        
        ### DISPLACEMENT ###
        "u_r_expr": dolfinx.fem.Expression(ufl.inner(r + u, r0), scalar_space.element.interpolation_points()),
        

        ### STRAIN ###
        "strain_ff_expr": dolfinx.fem.Expression(
            ufl.inner(A * f0, f0),
            scalar_space.element.interpolation_points(),
        ),

        "strain_nn_expr": dolfinx.fem.Expression(
            ufl.inner(A * r0, r0),
            scalar_space.element.interpolation_points(),
        ),

        "elastic_strain_rr_expr": dolfinx.fem.Expression(  # hoop strain
        ufl.inner((A.T * A - ufl.Identity(2)) / 2 * r0, r0),
        scalar_space.element.interpolation_points(),
        ),

        "elastic_strain_ff_expr": dolfinx.fem.Expression(  # hoop strain
        ufl.inner((A.T * A - ufl.Identity(2)) / 2 * f0, f0),
        scalar_space.element.interpolation_points(),
        ),

        ### STRESS ###
        "cauchy_ff_expr": dolfinx.fem.Expression(  # hoop stress
            ufl.inner(cauchy * f0, f0),
            scalar_space.element.interpolation_points(),
        ),

        "cauchy_nn_expr": dolfinx.fem.Expression(
            ufl.inner(cauchy * r0, r0),
            scalar_space.element.interpolation_points(),
        ),

        "PK1_ff_expr": dolfinx.fem.Expression(  # hoop stress
        ufl.inner(stress * f0, f0),
        scalar_space.element.interpolation_points(),
        ),

        "PK1_nn_expr": dolfinx.fem.Expression(
            ufl.inner(stress * r0, r0),
            scalar_space.element.interpolation_points(),
        ),

        "Mandel_ff_expr": dolfinx.fem.Expression(  # hoop stress
            ufl.inner(ufl.det(F) * cauchy * f0, f0),
            scalar_space.element.interpolation_points(),
        ),

        "Mandel_nn_expr": dolfinx.fem.Expression(
            ufl.inner(ufl.det(F) * cauchy * r0, r0),
            scalar_space.element.interpolation_points(),
        ),

        "PK2_ff_expr": dolfinx.fem.Expression(
            ufl.inner(ufl.inv(F) * stress * f0, f0),
            scalar_space.element.interpolation_points(),
        ),

        "PK2_nn_expr": dolfinx.fem.Expression(
            ufl.inner(ufl.inv(F) * stress * r0, r0),
            scalar_space.element.interpolation_points(),
        )
        ,

        # ### GROWTH ###
        # "dgt_expr": dolfinx.fem.Expression(params["dt"] * (cauchy_ff - params["set_point"]) / params["set_point"] + 1, scalar_space.element.interpolation_points()),

        # "gt_expr": dolfinx.fem.Expression(g_t * (params["dt"] * (cauchy_ff - params["set_point"]) / params["set_point"] + 1), scalar_space.element.interpolation_points()),

        # "dgr_expr ": dolfinx.fem.Expression(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), scalar_space.element.interpolation_points()),

        # "gr_expr": dolfinx.fem.Expression(g_r, scalar_space.element.interpolation_points()),
    }

    growth_expressions = growth_laws(problem_variables, params, scalar_space)
    expressions.update(growth_expressions)

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
    u_r = problem_variables["u_r"]
    g_r = problem_variables["g_r"]
    g_t = problem_variables["g_t"]
    elastic_strain_ff = problem_variables["elastic_strain_ff"]
    elastic_strain_rr = problem_variables["elastic_strain_rr"]
    strain_ff = problem_variables["strain_ff"]
    strain_nn = problem_variables["strain_nn"]
    cauchy_ff = problem_variables["cauchy_ff"]
    cauchy_nn = problem_variables["cauchy_nn"]
    dgt = problem_variables["dgt"]
    dgr = problem_variables["dgr"]

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
    elastic_strain_rr.interpolate(expressions["elastic_strain_rr_expr"])
    elastic_strain_ff.interpolate(expressions["elastic_strain_ff_expr"])
    strain_ff.interpolate(expressions["strain_ff_expr"])
    strain_nn.interpolate(expressions["strain_nn_expr"])
    cauchy_ff.interpolate(expressions["cauchy_ff_expr"])
    cauchy_nn.interpolate(expressions["cauchy_nn_expr"])
    dgt.interpolate(expressions["dgt_expr"])
    dgr.interpolate(expressions["dgr_expr"])
    u_r.interpolate(expressions["u_r_expr"])

    print("updating line data")
    data_collector.update_line_data()
    # data_collector.write(t=time[0])

    for i in time_steps:

        g_t.interpolate(expressions["gt_expr"])
        g_r.interpolate(expressions["gr_expr"])

        print(f"Starting growth step {i + 1} of {params['num_steps']}")

        # print g2 values
        print("g_r min/max:", g_r.x.array.min(), g_r.x.array.max())
        print("g_t min/max:", g_t.x.array.min(), g_t.x.array.max())
        solver.solve()

        elastic_strain_rr.interpolate(expressions["elastic_strain_rr_expr"])
        elastic_strain_ff.interpolate(expressions["elastic_strain_ff_expr"])
        strain_ff.interpolate(expressions["strain_ff_expr"])
        strain_nn.interpolate(expressions["strain_nn_expr"])
        cauchy_ff.interpolate(expressions["cauchy_ff_expr"])
        cauchy_nn.interpolate(expressions["cauchy_nn_expr"])
        u_r.interpolate(expressions["u_r_expr"])

        dgt.interpolate(expressions["dgt_expr"])
        dgr.interpolate(expressions["dgr_expr"])
        print("updating line data")
        data_collector.update_line_data()
        data_collector.write(t=time[i])
