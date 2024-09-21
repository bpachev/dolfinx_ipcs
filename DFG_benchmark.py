import argparse
from pathlib import Path

import dolfinx.fem.petsc as petsc_fem
import numpy as np
import ufl
from dolfinx import common, default_scalar_type, fem, io, la, log
from mpi4py import MPI
from petsc4py import PETSc

from create_and_convert_2D_mesh import markers
try:
    import cudolfinx as cufem
except ImportError:
    print("Must install cuda-dolfinx to use CUDA-accelerated assembly.")

has_tqdm = True
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    has_tqdm = False
    print("To view progress with progressbar please install tqdm: `pip3 install tqdm`")

log.set_log_level(log.LogLevel.ERROR)

def print_mat_sum(matrix, name, save=False):
    """Print sum of nonzero entries in PETSc matrix"""
    indptr, indices, data = matrix.getValuesCSR()
    print(f"Sum of {name}: {data.sum()}")
    if save:
        if args.cuda:
            np.savez(name+"-cuda", indptr=indptr, indices=indices, data=data)
        else:
            np.savez(name, indptr=indptr, indices=indices, data=data)

def print_vec(vec, name):
    """Print sum of entries in PETSc vector"""
    print(f"Sum of {name}: {vec.array[:].sum()}")

def IPCS(outdir: Path, filename: str, degree_u: int, cuda=False,
         jit_options: dict = {"cffi_extra_compile_args": ["-Ofast", "-march=native"], "cffi_libraries": ["m"]}):
    assert degree_u >= 2

    mesh_dir = Path("meshes")
    if not mesh_dir.exists():
        raise RuntimeError(f"Could not find {str(mesh_dir)}")
    # Read in mesh
    comm = MPI.COMM_WORLD
    with io.XDMFFile(comm, f"meshes/{filename}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, tdim)
        mesh.topology.create_connectivity(fdim, tdim)

    with io.XDMFFile(comm, f"meshes/{filename}_facets.xdmf", "r") as xdmf:
        mt = xdmf.read_meshtags(mesh, "Facet tags")

    # Define function spaces
    V = fem.functionspace(mesh, ("Lagrange", degree_u, (mesh.geometry.dim, )))
    Q = fem.functionspace(mesh, ("Lagrange", degree_u - 1))

    # Temporal parameters
    t = 0
    dt = default_scalar_type(1e-2)
    T = 8
    #T=2*dt

    # Physical parameters
    nu = 0.001
    f = fem.Constant(mesh, default_scalar_type((0,) * mesh.geometry.dim))
    H = 0.41
    Um = 2.25

    # Define functions for the variational form
    uh = fem.Function(V)
    uh.name = "Velocity"
    u_tent = fem.Function(V)
    u_tent.name = "Tentative_velocity"
    u_old = fem.Function(V)
    ph = fem.Function(Q)
    ph.name = "Pressure"
    phi = fem.Function(Q)
    phi.name = "Phi"
    if comm.rank == 0:
      print("ndofs on rank 0", len(uh.x.array))
    if cuda:
        for _func in [u_tent, phi, uh, ph]:
            _func.vector.setType(PETSc.Vec.Type.CUDA)

    # Define variational forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)

    # ----Step 1: Tentative velocity step----
    w_time = fem.Constant(mesh, 3 / (2 * dt))
    w_diffusion = fem.Constant(mesh, default_scalar_type(nu))
    a_tent = w_time * ufl.inner(u, v) * dx + w_diffusion * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L_tent = (ufl.inner(ph, ufl.div(v)) + ufl.inner(f, v)) * dx
    L_tent += fem.Constant(mesh, 1 / (2 * dt)) * ufl.inner(4 * uh - u_old, v) * dx
    # BDF2 with implicit Adams-Bashforth
    bs = 2 * uh - u_old
    a_tent += ufl.inner(ufl.grad(u) * bs, v) * dx
    # Temam-device
    a_tent += 0.5 * ufl.div(bs) * ufl.inner(u, v) * dx

    # Find boundary facets and create boundary condition
    inlet_facets = mt.indices[mt.values == markers["Inlet"]]
    inlet_dofs = fem.locate_dofs_topological(V, fdim, inlet_facets)
    wall_facets = mt.indices[mt.values == markers["Walls"]]
    wall_dofs = fem.locate_dofs_topological(V, fdim, wall_facets)
    obstacle_facets = mt.indices[mt.values == markers["Obstacle"]]
    obstacle_dofs = fem.locate_dofs_topological(V, fdim, obstacle_facets)

    def inlet_velocity(t):
        if mesh.geometry.dim == 3:
            return lambda x: ((16 * np.sin(np.pi * t / T) * Um * x[1] * x[2] * (H - x[1]) * (H - x[2]) / (H**4),
                               np.zeros(x.shape[1]), np.zeros(x.shape[1])))
        elif mesh.geometry.dim == 2:
            U = 1.5 * np.sin(np.pi * t / T)
            return lambda x: np.row_stack((4 * U * x[1] * (0.41 - x[1]) / (0.41**2), np.zeros(x.shape[1])))

    u_inlet = fem.Function(V)
    u_inlet.interpolate(inlet_velocity(t))
    zero = np.array((0,) * mesh.geometry.dim, dtype=default_scalar_type)
    bcs_tent = [fem.dirichletbc(u_inlet, inlet_dofs), fem.dirichletbc(
        zero, wall_dofs, V), fem.dirichletbc(zero, obstacle_dofs, V)]
    if cuda:
        asm = cufem.CUDAAssembler()
        a_tent = cufem.form(a_tent, jit_options=jit_options)
        device_bcs_tent = asm.pack_bcs(bcs_tent)
        A_tent = asm.assemble_matrix(a_tent, bcs=device_bcs_tent)
        L_tent = cufem.form(L_tent, jit_options=jit_options)
        b_tent = asm.create_vector(L_tent)
    else:
        a_tent = fem.form(a_tent, jit_options=jit_options)
        A_tent = petsc_fem.assemble_matrix(a_tent, bcs=bcs_tent)
        A_tent.assemble()
        L_tent = fem.form(L_tent, jit_options=jit_options)
        b_tent = fem.Function(V)

    # Step 2: Pressure correction step
    outlet_facets = mt.indices[mt.values == markers["Outlet"]]
    outlet_dofs = fem.locate_dofs_topological(Q, fdim, outlet_facets)
    bcs_corr = [fem.dirichletbc(default_scalar_type(0), outlet_dofs, Q)]
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    a_corr = ufl.inner(ufl.grad(p), ufl.grad(q)) * dx
    L_corr = - w_time * ufl.inner(ufl.div(u_tent), q) * dx
    if cuda:
        a_corr = cufem.form(a_corr, jit_options=jit_options)
        device_bcs_corr = asm.pack_bcs(bcs_corr)
        A_corr = asm.assemble_matrix(a_corr, bcs=device_bcs_corr)
        L_corr = cufem.form(L_corr, jit_options=jit_options)
        b_corr = asm.create_vector(L_corr)
        cu_phi = asm.create_vector(L_corr)
    else:
        a_corr = fem.form(a_corr, jit_options=jit_options)
        A_corr = petsc_fem.assemble_matrix(a_corr, bcs=bcs_corr)
        A_corr.assemble()
        b_corr = fem.Function(Q)
        L_corr = fem.form(L_corr, jit_options=jit_options)

    # Step 3: Velocity update
    if cuda:
        a_up = cufem.form(ufl.inner(u, v) * dx, jit_options=jit_options)
        L_up = cufem.form((ufl.inner(u_tent, v) - w_time**(-1) * ufl.inner(ufl.grad(phi), v)) * dx,
                    jit_options=jit_options)
        A_up = asm.assemble_matrix(a_up)
        b_up = asm.create_vector(L_up)
    else:
        a_up = fem.form(ufl.inner(u, v) * dx, jit_options=jit_options)
        L_up = fem.form((ufl.inner(u_tent, v) - w_time**(-1) * ufl.inner(ufl.grad(phi), v)) * dx,
                    jit_options=jit_options)
        A_up = petsc_fem.assemble_matrix(a_up)
        A_up.assemble()
        b_up = fem.Function(V)

    # Setup solvers
    rtol = 1e-8
    atol = 1e-8
    solver_tent = PETSc.KSP().create(comm)  # type: ignore
    if cuda:
        solver_tent.setOperators(A_tent.mat)
    else:
        solver_tent.setOperators(A_tent)
    solver_tent.setTolerances(rtol=rtol, atol=atol)
    solver_tent.rtol = rtol
    solver_tent.setType("bcgs")
    solver_tent.getPC().setType("jacobi")
    # solver_tent.setType("preonly")
    # solver_tent.getPC().setType("lu")
    # solver_tent.getPC().setFactorSolverType("mumps")

    solver_corr = PETSc.KSP().create(comm)  # type: ignore
    if cuda:
        solver_corr.setOperators(A_corr.mat)
    else:
        solver_corr.setOperators(A_corr)
    solver_corr.setTolerances(rtol=rtol, atol=atol)
    # solver_corr.setType("preonly")
    # solver_corr.getPC().setType("lu")
    # solver_corr.getPC().setFactorSolverType("mumps")
    solver_corr.setInitialGuessNonzero(True)
    solver_corr.max_it = 200
    solver_corr.setType("gmres")
    solver_corr.getPC().setType("hypre")
    solver_corr.getPC().setHYPREType("boomeramg")

    solver_up = PETSc.KSP().create(comm)  # type: ignore
    if cuda:
        solver_up.setOperators(A_up.mat)
    else:
        solver_up.setOperators(A_up)
    solver_up.setTolerances(rtol=rtol, atol=atol)
    # solver_up.setType("preonly")
    # solver_up.getPC().setType("lu")
    # solver_up.getPC().setFactorSolverType("mumps")
    solver_up.setInitialGuessNonzero(True)
    solver_up.max_it = 200
    solver_up.setType("cg")
    solver_up.getPC().setType("jacobi")

    # Create output files
    """out_u = io.VTXWriter(comm, outdir / f"u_{dim}D.bp", [uh], engine="BP4")
    out_p = io.VTXWriter(comm, outdir / f"p_{dim}D.bp", [ph], engine="BP4")
    out_u.write(t)
    out_p.write(t)"""

    # Solve problem
    N = int(T / dt)
    if has_tqdm:
        time_range = tqdm(range(N))
    else:
        time_range = range(N)
    for i in time_range:

        t += dt
        # Solve step 1
        with common.Timer("~Step 1"):
            u_inlet.interpolate(inlet_velocity(t))
            with common.Timer("~Assemble 1"):
                if cuda:
                    # Hack to force device-side PETSc vector values to propigate back to dolfinx Vector
                    uh.x.array[:] = uh.vector.array[:]
                    # Account for changing bcs on device
                    device_bcs_tent.update(bcs_tent)
                    asm.assemble_matrix(a_tent, mat=A_tent, bcs=device_bcs_tent)
                    asm.assemble_vector(L_tent, b_tent)
                    asm.apply_lifting(b_tent, [a_tent], [device_bcs_tent])
                    # need to specify the function space where bcs apply
                    asm.set_bc(b_tent, bcs_tent, V)
                else:
                    A_tent.zeroEntries()
                    petsc_fem.assemble_matrix(A_tent, a_tent, bcs=bcs_tent)  # type: ignore
                    A_tent.assemble()
                    b_tent.x.array[:] = 0
                    petsc_fem.assemble_vector(b_tent.vector, L_tent)
                    petsc_fem.apply_lifting(b_tent.vector, [a_tent], [bcs_tent])
                    b_tent.x.scatter_reverse(la.InsertMode.add)
                    petsc_fem.set_bc(b_tent.vector, bcs_tent)
            with common.Timer("~Solve 1"):
                solver_tent.solve(b_tent.vector, u_tent.vector)
            u_tent.x.scatter_forward()

        # Solve step 2
        with common.Timer("~Step 2"):
            if cuda:
                with common.Timer("~Assemble 2"):
                    # TODO: find a better way to do this
                    # It's tricky without access to the dolfinx::Vector internals
                    u_tent.x.array[:] = u_tent.vector.array
                    device_bcs_corr.update(bcs_corr)
                    asm.assemble_vector(L_corr,b_corr)
                    asm.apply_lifting(b_corr, [a_corr], [device_bcs_corr])
                    asm.set_bc(b_corr, bcs_corr, Q)
            else:
                with common.Timer("~Assemble 2"):
                    b_corr.x.array[:] = 0
                    petsc_fem.assemble_vector(b_corr.vector, L_corr)
                    petsc_fem.apply_lifting(b_corr.vector, [a_corr], [bcs_corr])
                    b_corr.x.scatter_reverse(la.InsertMode.add)
                    petsc_fem.set_bc(b_corr.vector, bcs_corr)

            with common.Timer("~Solve 2"):
                solver_corr.solve(b_corr.vector, phi.vector)
            phi.x.scatter_forward()

            # Update p and previous u
            ph.vector.axpy(1.0, phi.vector)
            if cuda:
                ph.x.array[:] = ph.vector.array
            ph.x.scatter_forward()

            u_old.x.array[:] = uh.x.array
            u_old.x.scatter_forward()

        # Solve step 3
        with common.Timer("~Step 3"):
            with common.Timer("~Assemble 3"):
                if cuda:
                    # TODO find a way to avoid this hack
                    phi.x.array[:] = phi.vector.array
                    asm.assemble_vector(L_up, b_up)
                else:
                    b_up.x.array[:] = 0
                    petsc_fem.assemble_vector(b_up.vector, L_up)
                    b_up.x.scatter_reverse(la.InsertMode.add)
            with common.Timer("~Solve 3"):
                solver_up.solve(b_up.vector, uh.vector)
            uh.x.scatter_forward()

        with common.Timer("~IO"):
            pass
            #out_u.write(t)
            #out_p.write(t)

    #out_u.close()
   # out_p.close()
    print_vec(uh.vector, "uh")
    print_vec(ph.vector, "ph")
    tasks = ["~" + name + " " + str(number) for number in range(1,4) for name in ["Step", "Solve", "Assemble"]]
    timing_arrs = {
      task: MPI.COMM_WORLD.gather(common.timing(task), root=0) for task in tasks
    }
    print(timing_arrs)
    if comm.rank == 0:
        max_times = {}
        for task, arr in timing_arrs.items():
          arr = np.asarray(arr)
          max_times[task] = np.max(arr[:,1]/arr[:,0])
        for i in range(1,4):
          total = max_times[f'~Step {i}']
          solve = max_times[f'~Solve {i}']
          assemble = max_times[f'~Assemble {i}']
          print(f"Step {i}: total={total:.3e}, assemble={assemble:.3e}, solve={solve:.3e}")

    """t_step_1 = MPI.COMM_WORLD.gather(common.timing("~Step 1"), root=0)
    t_step_2 = MPI.COMM_WORLD.gather(common.timing("~Step 2"), root=0)
    t_step_3 = MPI.COMM_WORLD.gather(common.timing("~Step 3"), root=0)
    io_time = MPI.COMM_WORLD.gather(common.timing("~IO"), root=0)
    if comm.rank == 0:
        solve_arrs = [solve_1, solve_2, solve_3]
        print("Time-step breakdown")
        for i, step in enumerate([t_step_1, t_step_2, t_step_3]):
            step_arr = np.asarray(step)
            time_per_run = step_arr[:, 1] / step_arr[:, 0]
            print(f"Step {i+1}: Min time: {np.min(time_per_run):.3e}, Max time: {np.max(time_per_run):.3e}")
            solve_arr = np.asarray(solve_arrs[i])
            time_per_solve = solve_arr[:, 1] / solve_arr[:, 0]
            print(f"Min solve time: {np.min(time_per_solve):.3e}, Max solve time: {np.max(time_per_solve):.3e}")
        io_time_arr = np.asarray(io_time)
        time_per_run = io_time_arr[:, 1] / io_time_arr[:, 0]
        print(f"IO {i+1}:   Min time: {np.min(time_per_run):.3e}, Max time: {np.max(time_per_run):.3e}")

    #common.list_timings(comm, [common.TimingType.wall])"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run the DFG 2D-3 benchmark"
        + "http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--degree-u", default=2, type=int, dest="degree", help="Degree of velocity space")
    #_2D = parser.add_mutually_exclusive_group(required=False)
    #_2D.add_argument('--3D', dest='threed', action='store_true', help="Use 3D mesh", default=False)
    parser.add_argument("--outdir", default="results", type=str, dest="outdir", help="Name of output folder")
    parser.add_argument("--cuda", action="store_true", help="Use GPU acceleration", default=False)
    parser.add_argument("--filename", default="channel2D")
    args = parser.parse_args()
   # dim = 3 if args.threed else 2
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    IPCS(outdir, filename=args.filename, degree_u=args.degree, cuda=args.cuda)
