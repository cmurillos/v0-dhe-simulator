import sys
import time
import numpy as np

from ._cylinder_mesh import CylinderMesh
from ._cylinder_fem_solver import CylinderFEMSolver
from ._generador_campos import GeneradorCamposFisicos


class DHEResult:
    """
    Container for DHE simulation results.

    Attributes
    ----------
    nodes : ndarray, shape (N_nodes, 3)
        Mesh node coordinates (x, y, z).
    elements : ndarray, shape (N_tets, 4)
        Tetrahedral connectivity (node indices per tetrahedron).
    boundary_faces : ndarray, shape (N_faces, 3)
        Triangular boundary-face connectivity.
    R_min : float
        Inner (borehole) radius.
    z_min : float
        Top of the sealed base / bottom of the open annulus.
    T0_stable : ndarray, shape (N_nodes,)
        Stabilized initial temperature field.
    times : ndarray, shape (N_snapshots,)
        Saved time instants.
    T : ndarray, shape (N_snapshots, N_nodes)
        Temperature at every node for each saved snapshot.
    """

    def __init__(self, nodes, elements, boundary_faces,
                 R_min, z_min, T0_stable, times, T_snapshots):
        self.nodes = np.asarray(nodes)
        self.elements = np.asarray(elements)
        self.boundary_faces = np.asarray(boundary_faces)
        self.R_min = float(R_min)
        self.z_min = float(z_min)
        self.T0_stable = np.asarray(T0_stable, dtype=float).ravel()
        self.times = np.asarray(times, dtype=float)
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
        """Save the full result to a compressed .npz file."""
        np.savez_compressed(
            path,
            nodes=self.nodes,
            elements=self.elements,
            boundary_faces=self.boundary_faces,
            R_min=np.array(self.R_min),
            z_min=np.array(self.z_min),
            T0_stable=self.T0_stable,
            times=self.times,
            T=self.T,
        )

    @staticmethod
    def load(path):
        """Load a DHEResult from a .npz file."""
        data = np.load(path)
        return DHEResult(
            nodes=data["nodes"],
            elements=data["elements"],
            boundary_faces=data["boundary_faces"],
            R_min=float(data["R_min"]),
            z_min=float(data["z_min"]),
            T0_stable=data["T0_stable"],
            times=data["times"],
            T_snapshots=data["T"],
        )

    def delta(self):
        """
        Area-weighted mean perturbation on the borehole wall.

        Automatically identifies the boundary faces of the inner
        cylindrical surface (r ~ R_min, z >= z_min), then computes

            delta(t) = sum_f A_f |T0_f - T_f(t)| / sum_f A_f

        No parameters needed -- R_min and z_min come from the
        simulation geometry.

        Returns
        -------
        times : ndarray, shape (N_snapshots,)
        delta : ndarray, shape (N_snapshots,)
        """
        if self.T0_stable is None:
            raise RuntimeError(
                "T0_stable not available. Run with stabilization first."
            )

        # --- identify borehole faces ---
        face_pts = self.nodes[self.boundary_faces]       # (F, 3, 3)
        r_verts = np.sqrt(face_pts[:, :, 0]**2 +
                          face_pts[:, :, 1]**2)          # (F, 3)
        z_verts = face_pts[:, :, 2]                      # (F, 3)

        tol_r = 0.5 * self.R_min
        inner = (
            np.all(np.abs(r_verts - self.R_min) < tol_r, axis=1) &
            np.all(z_verts >= self.z_min - 1e-6, axis=1)
        )
        if not inner.any():
            raise RuntimeError(
                "No borehole faces found. Check R_min / z_min."
            )

        sel = self.boundary_faces[inner]                 # (n, 3)

        # --- face areas ---
        p = self.nodes[sel]                              # (n, 3, 3)
        v1 = p[:, 1] - p[:, 0]
        v2 = p[:, 2] - p[:, 0]
        areas = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=1)
        total_area = areas.sum()

        # --- T0_stable averaged at 3 vertices of each face ---
        T0_face = self.T0_stable[sel].mean(axis=1)      # (n,)

        # --- T(t) averaged at 3 vertices of each face ---
        T_face = self.T[:, sel].mean(axis=2)             # (snaps, n)

        # --- area-weighted mean of |T0 - T(t)| ---
        diff = np.abs(T0_face[np.newaxis, :] - T_face)  # (snaps, n)
        delta_arr = (diff * areas[np.newaxis, :]).sum(axis=1) / total_area

        return self.times.copy(), delta_arr

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
        self.R_min = R_min
        self.z_min = z_min
        self._R_range = np.linspace(R_min, R_max, nr)
        self._Z_range = np.linspace(0, z_max, nz)

        print("[DHE] Building mesh ...", end=" ", flush=True)
        self.cylinder = CylinderMesh(self._R_range, self._Z_range, z_min, nangl)
        print(f"({self.cylinder.nodes.shape[0]} nodes)")

        self._solver = CylinderFEMSolver(self.cylinder.nodes, self.cylinder.elements, self.cylinder.boundary_faces)

        print("[DHE] Interpolating physical fields ...", end=" ", flush=True)
        self.p, self.c, self.k, self.T0_array = self._get_fields()
        print("done")

        print("[DHE] Assembling FEM matrices ...", end=" ", flush=True)
        self._solver.assemble_system(self.p, self.c, self.k, alpha, R_min, 0.1)
        print("done")

    def _get_fields(self):
        gen = GeneradorCamposFisicos(ruta_csv=self.csv, nodes=self._solver.skfem_nodes)
        p_field = lambda x: gen.p(*x)
        c_field = lambda x: gen.c(*x)
        k_field = lambda x: gen.k(*x)
        return p_field, c_field, k_field, np.array([gen.T(*x) for x in self._solver.skfem_nodes])

    def solve(self, dt, t_save, t_on, t_off, tf, T_c,
              stab_dt=None, stab_tol=1e-6):
        """
        Parameters
        ----------
        dt, t_save, t_on, t_off, tf, T_c :
            Same as before.
        stab_dt : float or None
            Time step for stabilization. Defaults to dt.
        stab_tol : float
            Relative tolerance for stabilization (default 1e-6).
        """
        if stab_dt is None:
            stab_dt = dt

        # 1. Stabilize: insulated boundary until steady state
        print("[DHE] Phase 1/2 : Stabilizing initial condition ...")
        t0 = time.time()
        T0_stable, n_stab = self._solver.stabilize(
            self.T0_array, dt=stab_dt, tol=stab_tol)
        print(f"[DHE] Stabilized in {n_stab} iters ({time.time()-t0:.1f}s)")

        # 2. Run simulation from the stable field
        print("[DHE] Phase 2/2 : Solving thermal evolution ...")
        raw = self._solver.solve(
            T0=T0_stable,
            dt=dt,
            tf=tf,
            t_save=t_save,
            T_c_func=T_c,
            t_on=t_on,
            t_off=t_off)

        T_list = [np.asarray(Ti).ravel() for Ti in raw["T"]]

        return DHEResult(
            nodes=self.cylinder.nodes,
            elements=self.cylinder.elements,
            boundary_faces=self.cylinder.boundary_faces,
            R_min=self.R_min,
            z_min=self.z_min,
            T0_stable=T0_stable,
            times=raw["t"],
            T_snapshots=np.vstack(T_list),
        )
