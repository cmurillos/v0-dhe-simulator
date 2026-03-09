"""
DHE Thermal Simulation
"""
import numpy as np
import pandas as pd

from ._cylinder_mesh import CylinderMesh
from ._cylinder_fem_solver import CylinderFEMSolver
from ._generador_campos import GeneradorCamposFisicos


class DHE_simulation:
    """
    Downhole Heat Exchanger thermal simulation.

    Parameters
    ----------
    csv : str
        Path to CSV with physical rock properties.
    rad_borehole : float
        Borehole (inner) radius.
    rad_simulation : float
        Outer radius of simulation domain.
    deep_borehole : float
        Depth where borehole starts (z_min).
    time_on : float
        Start of active (heat exchange) window.
    time_off : float
        End of active window.
    time_final : float
        Final simulation time.
    dt : float
        Time step.
    time_save : float
        Snapshot save interval.
    T_c : float
        Boundary temperature for Robin condition.
    nr, nz, ntheta : int
        Mesh resolution (radial, axial, angular).
    alpha : float
        Robin boundary coefficient.
    """

    def __init__(self, csv, rad_borehole, rad_simulation, deep_borehole,
                 time_on, time_off, time_final,
                 dt, time_save, T_c,
                 nr=10, nz=20, ntheta=15, alpha=0.01):

        self.csv = csv
        self.R_min = rad_borehole
        self.R_max = rad_simulation
        self.z_min = deep_borehole
        self.t_on = time_on
        self.t_off = time_off
        self.tf = time_final
        self.dt = dt
        self.t_save = time_save
        self.T_c_val = T_c
        self.alpha = alpha

        # Extract z_max from CSV
        df = pd.read_csv(csv)
        self.z_max = df['Z'].max() - 1.0

        # Build mesh
        print("[DHE] Building mesh ...", end=" ", flush=True)
        R_range = np.linspace(rad_borehole, rad_simulation, nr)
        Z_range = np.linspace(0, self.z_max, nz)
        self.cylinder = CylinderMesh(R_range, Z_range, self.z_min, ntheta)
        self.nodes = self.cylinder.nodes
        self.elements = self.cylinder.elements
        print(f"({self.nodes.shape[0]} nodes)")

        # Solver
        self._solver = CylinderFEMSolver(
            self.cylinder.nodes,
            self.cylinder.elements,
            self.cylinder.boundary_faces
        )

    def _get_fields(self):
        """Generate random physical fields from CSV."""
        gen = GeneradorCamposFisicos(
            ruta_csv=self.csv,
            nodes=self._solver.skfem_nodes
        )
        p = lambda x: gen.p(*x)
        c = lambda x: gen.c(*x)
        k = lambda x: gen.k(*x)
        T0 = np.array([gen.T(*pt) for pt in self._solver.skfem_nodes])
        return p, c, k, T0

    def run(self, output="simulation"):
        """
        Run the simulation and export to XDMF + HDF5.

        Parameters
        ----------
        output : str
            Base filename for output (creates output.xdmf + output.h5)

        Returns
        -------
        str : path to the .xdmf file
        """
        T_c_func = lambda t, x, y, z: np.full_like(x, self.T_c_val)

        # Generate fields
        print("[DHE] Generating fields ...", end=" ", flush=True)
        p, c, k, T0 = self._get_fields()
        print("done")

        # Assemble FEM system
        print("[DHE] Assembling ...", end=" ", flush=True)
        self._solver.assemble_system(p, c, k, self.alpha, self.R_min, 0.1)
        print("done")

        # Stabilize
        print("[DHE] Stabilizing ...")
        T0_stable, n_iter = self._solver.stabilize(T0, self.dt)

        # Solve
        print("[DHE] Solving ...")
        results = self._solver.solve(
            T0=T0_stable,
            dt=self.dt,
            tf=self.tf,
            t_save=self.t_save,
            T_c_func=T_c_func,
            t_on=self.t_on,
            t_off=self.t_off
        )

        # Export to XDMF
        self._solver.export_xdmf(results, output)

        return f"{output}.xdmf"
