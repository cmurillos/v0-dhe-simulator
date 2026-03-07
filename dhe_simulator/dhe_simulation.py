import sys
import time
import numpy as np

from ._cylinder_mesh import CylinderMesh
from ._cylinder_fem_solver import CylinderFEMSolver
from ._generador_campos import GeneradorCamposFisicos


def _borehole_geometry(nodes, boundary_faces, R_min, z_min):
    """
    Return (sel, areas, total_area) for the inner-wall faces.

    sel   : (n, 3) index array of selected boundary faces
    areas : (n,)   area of each face
    total : float  sum of areas
    """
    face_pts = nodes[boundary_faces]
    r_v = np.sqrt(face_pts[:, :, 0]**2 + face_pts[:, :, 1]**2)
    z_v = face_pts[:, :, 2]
    tol_r = 0.5 * R_min
    mask = (
        np.all(np.abs(r_v - R_min) < tol_r, axis=1) &
        np.all(z_v >= z_min - 1e-6, axis=1)
    )
    if not mask.any():
        raise RuntimeError("No borehole faces found. Check R_min / z_min.")
    sel = boundary_faces[mask]
    p = nodes[sel]
    v1 = p[:, 1] - p[:, 0]
    v2 = p[:, 2] - p[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=1)
    return sel, areas, areas.sum()


def _compute_T_mean(T_matrix, nodes, boundary_faces, R_min, z_min):
    """
    Area-weighted mean temperature on the borehole wall.

    Parameters
    ----------
    T_matrix  : (N_snaps, N_nodes)
    Returns   : (N_snaps,)
    """
    sel, areas, total_area = _borehole_geometry(nodes, boundary_faces,
                                                R_min, z_min)
    T_face = T_matrix[:, sel].mean(axis=2)           # (snaps, n)
    return (T_face * areas[np.newaxis, :]).sum(axis=1) / total_area


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

    def T_borehole(self):
        """
        Area-weighted mean temperature on the borehole wall.

            T_mean(t) = sum_f A_f * T_f(t) / sum_f A_f

        No parameters needed -- R_min and z_min come from the
        simulation geometry.

        Returns
        -------
        times : ndarray, shape (N_snapshots,)
        T_mean : ndarray, shape (N_snapshots,)
        """
        Tm = _compute_T_mean(self.T, self.nodes, self.boundary_faces,
                             self.R_min, self.z_min)
        return self.times.copy(), Tm

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
        self.alpha = alpha
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

        # Build borehole node groups for profile extraction
        self._solver.build_borehole_groups(R_min, z_min)

    def _get_fields(self):
        gen = GeneradorCamposFisicos(ruta_csv=self.csv, nodes=self._solver.skfem_nodes)
        p_field = lambda x: gen.p(*x)
        c_field = lambda x: gen.c(*x)
        k_field = lambda x: gen.k(*x)
        return p_field, c_field, k_field, np.array([gen.T(*x) for x in self._solver.skfem_nodes])

    def _regenerate_fields(self):
        """Regenerate random fields and reassemble FEM matrices."""
        self.p, self.c, self.k, self.T0_array = self._get_fields()
        self._solver.assemble_system(self.p, self.c, self.k, self.alpha, self.R_min, 0.1)

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

    def scan_toff(self, dt, t_save, t_on, t_off_array, tf, T_c,
                  tries=1, stab_dt=None, stab_tol=1e-6):
        """
        Run simulations for multiple t_off values and multiple random
        field realizations (tries).

        Parameters
        ----------
        dt, t_save, t_on, tf, T_c :
            Same as :meth:`solve`.
        t_off_array : array-like
            Sequence of t_off values to sweep.
        tries : int
            Number of random field realizations to simulate.
        stab_dt, stab_tol :
            Stabilization parameters.

        Returns
        -------
        dict with keys:
            'times'   : ndarray (n_times,)
            'z_levels': ndarray (n_z,)
            't_off'   : ndarray (n_toff,)
            'T'       : ndarray (tries, n_toff, n_times, n_z)
                        Borehole surface T(t,z) for each try and t_off.
        """
        t_off_array = np.asarray(t_off_array, dtype=float)
        n_toff = len(t_off_array)

        if stab_dt is None:
            stab_dt = dt

        z_levels = self._solver._bh_z_levels
        n_z = len(z_levels)
        times_ref = None
        all_results = []  # list of (n_toff, n_times, n_z) per try

        for tr in range(tries):
            print(f"\n[scan] === Try {tr+1}/{tries} ===")

            if tr > 0:
                print("[scan] Regenerating random fields ...")
                self._regenerate_fields()

            # Stabilize for this field realization
            print("[scan] Stabilizing ...")
            T0_stable, n_stab = self._solver.stabilize(
                self.T0_array, dt=stab_dt, tol=stab_tol)
            print(f"[scan] Stabilized in {n_stab} iters")

            try_results = []  # (n_toff, n_times, n_z)

            for i, t_off in enumerate(t_off_array):
                print(f"[scan] t_off {i+1}/{n_toff} = {t_off:.2e}")

                raw = self._solver.solve(
                    T0=T0_stable,
                    dt=dt,
                    tf=tf,
                    t_save=t_save,
                    T_c_func=T_c,
                    t_on=t_on,
                    t_off=t_off)

                times_arr = np.asarray(raw["t"], dtype=float)
                if times_ref is None:
                    times_ref = times_arr

                # Extract borehole profile T(z) at each time snapshot
                T_surface = []  # (n_times, n_z)
                for T_snap in raw["T"]:
                    _, Tr = self._solver.borehole_profile(np.asarray(T_snap).ravel())
                    T_surface.append(Tr)
                try_results.append(np.array(T_surface))  # (n_times, n_z)

            all_results.append(np.array(try_results))  # (n_toff, n_times, n_z)

        T_tensor = np.array(all_results)  # (tries, n_toff, n_times, n_z)
        print(f"\n[scan] Done. T shape: {T_tensor.shape}")

        return {
            'times': times_ref,
            'z_levels': z_levels,
            't_off': t_off_array,
            'T': T_tensor,
        }
