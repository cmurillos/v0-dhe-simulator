"""
DHE Thermal Simulation - Simplified interface.
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
    tries : int
        Number of random field realizations.
    nr, nz, ntheta : int
        Mesh resolution (radial, axial, angular).
    alpha : float
        Robin boundary coefficient.
    """

    def __init__(self, csv, rad_borehole, rad_simulation,
                 time_on, time_off, time_final,
                 dt, time_save, tries,
                 nr=10, nz=20, ntheta=15, alpha=0.01):

        self.csv = csv
        self.R_min = rad_borehole
        self.R_max = rad_simulation
        self.t_on = time_on
        self.t_off = time_off
        self.tf = time_final
        self.dt = dt
        self.t_save = time_save
        self.tries = tries
        self.alpha = alpha
        self.nr = nr
        self.nz = nz
        self.ntheta = ntheta

        # Extract z_max and z_min from CSV
        df = pd.read_csv(csv)
        self.z_max = df['Z'].max() - 1.0  # small offset for compatibility
        self.z_min = df['Z'].min()

        # Build mesh
        print("[DHE] Building mesh ...", end=" ", flush=True)
        R_range = np.linspace(rad_borehole, rad_simulation, nr)
        Z_range = np.linspace(0, self.z_max, nz)
        self.cylinder = CylinderMesh(R_range, Z_range, self.z_min, ntheta)
        print(f"({self.cylinder.nodes.shape[0]} nodes)")

        # Solver
        self._solver = CylinderFEMSolver(
            self.cylinder.nodes,
            self.cylinder.elements,
            self.cylinder.boundary_faces
        )

        # Build borehole node groups by z-level
        self._build_borehole_groups()

    def _build_borehole_groups(self):
        """Identify borehole nodes and group by z-level."""
        nodes = self._solver.skfem_nodes
        r = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
        tol_r = 0.5 * self.R_min
        bh_mask = (np.abs(r - self.R_min) < tol_r) & \
                  (nodes[:, 2] >= self.z_min - 1e-6)
        bh_indices = np.where(bh_mask)[0]

        if len(bh_indices) == 0:
            raise RuntimeError("No borehole nodes found")

        bh_z = nodes[bh_indices, 2]
        bh_z_rounded = np.round(bh_z, 6)
        unique_z = np.unique(bh_z_rounded)

        # groups[i] = array of node indices at z-level i
        self._bh_z_levels = unique_z  # sorted ascending
        self._bh_groups = [bh_indices[bh_z_rounded == zv] for zv in unique_z]
        self._n_layers = len(unique_z)

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

    def _extract_layer_means(self, T_snapshots):
        """
        Extract mean T at each borehole layer for all snapshots.
        Returns list of n_layers arrays, each (n_times,)
        """
        n_times = len(T_snapshots)
        layer_means = [np.empty(n_times) for _ in range(self._n_layers)]

        for t_idx, T in enumerate(T_snapshots):
            T_flat = np.asarray(T).ravel()
            for layer_idx, node_indices in enumerate(self._bh_groups):
                layer_means[layer_idx][t_idx] = T_flat[node_indices].mean()

        return layer_means

    def run(self):
        """
        Run the simulation for all tries.

        Returns
        -------
        times : ndarray (n_times,)
        T_layers : list of n_layers arrays, each with shape (tries, n_times)
            T_layers[n][try_i, :] = mean borehole T at layer n for try_i
        """
        times_ref = None
        # Pre-allocate: list of lists, will become (n_layers, tries, n_times)
        all_layer_data = [[] for _ in range(self._n_layers)]

        T_c = lambda t, x, y, z: np.full_like(x, 300.0)

        for tr in range(self.tries):
            print(f"\n[DHE] === Try {tr+1}/{self.tries} ===")

            # Generate new random fields
            print("[DHE] Generating fields ...", end=" ", flush=True)
            p, c, k, T0 = self._get_fields()
            print("done")

            # Assemble FEM system
            print("[DHE] Assembling ...", end=" ", flush=True)
            self._solver.assemble_system(p, c, k, self.alpha, self.R_min, 0.1)
            print("done")

            # Stabilize
            print("[DHE] Stabilizing ...", end=" ", flush=True)
            T0_stable, n_iter = self._solver.stabilize(T0, self.dt)
            print(f"{n_iter} iters")

            # Solve
            print("[DHE] Solving ...")
            raw = self._solver.solve(
                T0=T0_stable,
                dt=self.dt,
                tf=self.tf,
                t_save=self.t_save,
                T_c_func=T_c,
                t_on=self.t_on,
                t_off=self.t_off
            )

            if times_ref is None:
                times_ref = np.asarray(raw["t"], dtype=float)

            # Extract layer means
            layer_means = self._extract_layer_means(raw["T"])
            for layer_idx in range(self._n_layers):
                all_layer_data[layer_idx].append(layer_means[layer_idx])

        # Convert to arrays: T_layers[layer] has shape (tries, n_times)
        T_layers = [np.array(layer_data) for layer_data in all_layer_data]

        print(f"\n[DHE] Done. {self._n_layers} layers, "
              f"{self.tries} tries, {len(times_ref)} snapshots")

        return times_ref, T_layers
