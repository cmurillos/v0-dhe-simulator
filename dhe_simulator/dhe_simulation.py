import numpy as np

from ._cylinder_mesh import CylinderMesh
from ._cylinder_fem_solver import CylinderFEMSolver
from ._generador_campos import GeneradorCamposFisicos


class DHEResult:
    """
    Container for DHE simulation results.

    Stores the complete mesh geometry and the thermal field at every
    saved time-step in a single, non-redundant object.

    Attributes
    ----------
    nodes : ndarray, shape (N_nodes, 3)
        Mesh node coordinates (x, y, z).
    elements : ndarray, shape (N_tets, 4)
        Tetrahedral connectivity (node indices per tetrahedron).
    boundary_faces : ndarray, shape (N_faces, 3)
        Triangular boundary-face connectivity.
    times : ndarray, shape (N_snapshots,)
        Saved time instants.
    T : ndarray, shape (N_snapshots, N_nodes)
        Temperature at every node for each saved snapshot.
    """

    def __init__(self, nodes, elements, boundary_faces, times, T_snapshots):
        self.nodes = np.asarray(nodes)
        self.elements = np.asarray(elements)
        self.boundary_faces = np.asarray(boundary_faces)
        self.times = np.asarray(times, dtype=float)
        # Guarantee a homogeneous (N_snapshots, N_nodes) float array
        # even if individual snapshots have inconsistent shapes.
        self.T = np.vstack([np.asarray(s, dtype=float).ravel()
                            for s in T_snapshots])

    # dict-like access for backward compatibility
    def __getitem__(self, key):
        if key == "t":
            return list(self.times)
        if key == "T":
            return [self.T[i] for i in range(self.T.shape[0])]
        raise KeyError(key)

    def save(self, path):
        """
        Save the full result to a compressed ``.npz`` file.

        The file contains the keys: ``nodes``, ``elements``,
        ``boundary_faces``, ``times``, ``T``.

        Parameters
        ----------
        path : str
            Output file path (e.g. ``'result.npz'``).
        """
        np.savez_compressed(
            path,
            nodes=self.nodes,
            elements=self.elements,
            boundary_faces=self.boundary_faces,
            times=self.times,
            T=self.T,
        )

    @staticmethod
    def load(path):
        """
        Load a DHEResult from a ``.npz`` file previously saved with
        :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the ``.npz`` file.

        Returns
        -------
        DHEResult
        """
        data = np.load(path)
        return DHEResult(
            nodes=data["nodes"],
            elements=data["elements"],
            boundary_faces=data["boundary_faces"],
            times=data["times"],
            T_snapshots=data["T"],
        )

    def mean_temperature(self, r_max, h):
        """
        Volume-weighted mean temperature inside a sub-cylinder.

        Selects every tetrahedron whose centroid satisfies
        ``r <= r_max`` and ``z <= h``, then computes

        .. math::

            \\langle T \\rangle(t) =
            \\frac{\\sum_e V_e \\, \\bar T_e(t)}{\\sum_e V_e}

        where the sum runs over selected elements, *V_e* is the
        tetrahedron volume, and  *T_e* is the average of T at its
        four vertices.

        Parameters
        ----------
        r_max : float
            Maximum radial distance (must be <= mesh outer radius).
        h : float
            Maximum height (z coordinate).

        Returns
        -------
        times : ndarray, shape (N_snapshots,)
        T_means : ndarray, shape (N_snapshots,)
        """
        pts = self.nodes[self.elements]                  # (N_tet, 4, 3)
        centroids = pts.mean(axis=1)                     # (N_tet, 3)
        r_c = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
        z_c = centroids[:, 2]

        mask = (r_c <= r_max) & (z_c <= h)
        if not mask.any():
            raise ValueError(
                f"No elements found with r<={r_max} and z<={h}."
            )

        sel = self.elements[mask]                        # (n, 4)

        # tetrahedron volumes  |det[v1,v2,v3]| / 6
        v1 = pts[mask, 0] - pts[mask, 3]
        v2 = pts[mask, 1] - pts[mask, 3]
        v3 = pts[mask, 2] - pts[mask, 3]
        vols = np.abs(np.einsum('ij,ij->i', v1, np.cross(v2, v3))) / 6.0

        total_vol = vols.sum()

        # T averaged at the 4 vertices of each selected tet, per snapshot
        T_verts = self.T[:, sel]                         # (snaps, n, 4)
        T_tet   = T_verts.mean(axis=2)                   # (snaps, n)

        T_means = (T_tet * vols[np.newaxis, :]).sum(axis=1) / total_vol

        return self.times.copy(), T_means

    def __repr__(self):
        return (
            f"DHEResult(nodes={self.nodes.shape}, "
            f"elements={self.elements.shape}, "
            f"boundary_faces={self.boundary_faces.shape}, "
            f"snapshots={len(self.times)})"
        )


class DHE_simulation():
    def __init__(self, csv, R_min, R_max, z_min, z_max, alpha=0.01, nz=20, nr=10, nangl=15):
        self.csv = csv
        self._R_range = np.linspace(R_min, R_max, nr)
        self._Z_range = np.linspace(0, z_max, nz)
        self.cylinder = CylinderMesh(self._R_range, self._Z_range, z_min, nangl)
        self._solver = CylinderFEMSolver(self.cylinder.nodes, self.cylinder.elements, self.cylinder.boundary_faces)
        self.p, self.c, self.k, self.T0_array = self._get_fields()
        self._solver.assemble_system(self.p, self.c, self.k, alpha, R_min, 0.1)

    def _get_fields(self):
        gen = GeneradorCamposFisicos(ruta_csv=self.csv, nodes=self._solver.skfem_nodes)
        p_field = lambda x: gen.p(*x)
        c_field = lambda x: gen.c(*x)
        k_field = lambda x: gen.k(*x)
        return p_field, c_field, k_field, np.array([gen.T(*x) for x in self._solver.skfem_nodes])

    def solve(self, dt, t_save, t_on, t_off, tf, T_c):
        raw = self._solver.solve(
            T0=self.T0_array,
            dt=dt,
            tf=tf,
            t_save=t_save,
            T_c_func=T_c,
            t_on=t_on,
            t_off=t_off)

        # Flatten each snapshot to 1D — splu.solve may return (N,) or (N,1)
        T_list = [np.asarray(Ti).ravel() for Ti in raw["T"]]

        return DHEResult(
            nodes=self.cylinder.nodes,
            elements=self.cylinder.elements,
            boundary_faces=self.cylinder.boundary_faces,
            times=raw["t"],
            T_snapshots=np.vstack(T_list),
        )
