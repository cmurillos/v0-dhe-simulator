import sys
import time
import numpy as np

from ._cylinder_mesh import CylinderMesh
from ._cylinder_fem_solver import CylinderFEMSolver
from ._generador_campos import GeneradorCamposFisicos


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------

class DHEResult:
    """
    Container for a single DHE simulation run.

    Attributes
    ----------
    nodes, elements, boundary_faces : mesh geometry
    R_min, z_min : borehole parameters
    T0_stable    : stabilized initial temperature field (N_nodes,)
    times        : saved time instants (N_snapshots,)
    T            : temperature at every node per snapshot (N_snapshots, N_nodes)
    z_levels     : borehole z-levels (n_z,)
    T_surface    : angular-averaged borehole T(t, z) (N_snapshots, n_z)
    """

    def __init__(self, nodes, elements, boundary_faces,
                 R_min, z_min, T0_stable, times, T_snapshots,
                 z_levels, T_surface):
        self.nodes = np.asarray(nodes)
        self.elements = np.asarray(elements)
        self.boundary_faces = np.asarray(boundary_faces)
        self.R_min = float(R_min)
        self.z_min = float(z_min)
        self.T0_stable = np.asarray(T0_stable, dtype=float).ravel()
        self.times = np.asarray(times, dtype=float)
        self.T = np.vstack([np.asarray(s, dtype=float).ravel()
                            for s in T_snapshots])
        self.z_levels = np.asarray(z_levels, dtype=float)
        self.T_surface = np.asarray(T_surface, dtype=float)

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
            z_levels=self.z_levels,
            T_surface=self.T_surface,
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
            z_levels=data["z_levels"],
            T_surface=data["T_surface"],
        )

    def __repr__(self):
        return (
            f"DHEResult(nodes={self.nodes.shape}, "
            f"snapshots={len(self.times)}, "
            f"T_surface={self.T_surface.shape})"
        )


class ScanResult:
    """
    Container for a t_off sweep (scan_toff).

    Attributes
    ----------
    t_off_array : (n_toff,)
    times       : (n_times,)
    z_levels    : (n_z,)
    T_borehole  : (n_toff, n_times, n_z)  -- borehole surface per t_off
    """

    def __init__(self, t_off_array, times, z_levels, T_borehole):
        self.t_off_array = np.asarray(t_off_array, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.z_levels = np.asarray(z_levels, dtype=float)
        self.T_borehole = np.asarray(T_borehole, dtype=float)

    def save(self, path):
        """Save the scan result to a compressed .npz file."""
        np.savez_compressed(
            path,
            t_off_array=self.t_off_array,
            times=self.times,
            z_levels=self.z_levels,
            T_borehole=self.T_borehole,
        )

    @staticmethod
    def load(path):
        """Load a ScanResult from a .npz file."""
        data = np.load(path)
        return ScanResult(
            t_off_array=data["t_off_array"],
            times=data["times"],
            z_levels=data["z_levels"],
            T_borehole=data["T_borehole"],
        )

    def __repr__(self):
        nt, nz = self.T_borehole.shape[1], self.T_borehole.shape[2]
        return (
            f"ScanResult(t_off=[{len(self.t_off_array)}], "
            f"times={nt}, z_levels={nz})"
        )


# ------------------------------------------------------------------
# Main simulation class
# ------------------------------------------------------------------

class DHE_simulation():
    def __init__(self, csv, R_min, R_max, z_min, z_max,
                 h, rho_f, c_f, v_f, T_in,
                 nz=20, nr=10, nangl=15):
        """
        Parameters
        ----------
        csv   : str   -- path to CSV with physical rock properties
        R_min : float -- borehole (inner) radius [m]
        R_max : float -- outer radius of the domain [m]
        z_min : float -- top of sealed base / bottom of open borehole [m]
        z_max : float -- bottom of the borehole [m]
        h     : float -- convective coefficient [W/(m^2 K)]
        rho_f : float -- fluid density [kg/m^3]
        c_f   : float -- fluid specific heat [J/(kg K)]
        v_f   : float -- axial fluid velocity [m/s]
        T_in  : float -- fluid inlet temperature [K]
        nz, nr, nangl : int -- mesh resolution parameters
        """
        self.csv = csv
        self.R_min = R_min
        self.z_min = z_min
        self._R_range = np.linspace(R_min, R_max, nr)
        self._Z_range = np.linspace(0, z_max, nz)

        print("[DHE] Building mesh ...", end=" ", flush=True)
        self.cylinder = CylinderMesh(self._R_range, self._Z_range, z_min, nangl)
        print(f"({self.cylinder.nodes.shape[0]} nodes)")

        self._solver = CylinderFEMSolver(
            self.cylinder.nodes, self.cylinder.elements,
            self.cylinder.boundary_faces)

        print("[DHE] Interpolating physical fields ...", end=" ", flush=True)
        self.p, self.c, self.k, self.T0_array = self._get_fields()
        print("done")

        print("[DHE] Assembling FEM matrices + fluid model ...", end=" ", flush=True)
        self._solver.assemble_system(
            self.p, self.c, self.k,
            h=h, R_min=R_min, z_min=z_min, eps=0.1,
            rho_f=rho_f, c_f=c_f, v_f=v_f, T_in=T_in)
        print("done")

    def _get_fields(self):
        gen = GeneradorCamposFisicos(
            ruta_csv=self.csv, nodes=self._solver.skfem_nodes)
        # gen.p, gen.c, gen.k expect (x, y, z) with arrays of shape (N,)
        p_field = gen.p
        c_field = gen.c
        k_field = gen.k
        T0 = np.array([gen.T(x, y, z)
                        for x, y, z in self._solver.skfem_nodes])
        return p_field, c_field, k_field, T0

    def _extract_surface(self, raw):
        """
        From a raw solver result dict, build the borehole surface
        T_surface(t, z) by angular-averaging at each z-level.

        Returns
        -------
        z_levels : (n_z,)
        T_surface : (n_snapshots, n_z)
        """
        rows = []
        z_levels = None
        for Ti in raw["T"]:
            Ti_flat = np.asarray(Ti, dtype=float).ravel()
            zl, Tr = self._solver.borehole_profile(Ti_flat)
            if z_levels is None:
                z_levels = zl
            rows.append(Tr)
        return z_levels, np.vstack(rows)

    def solve(self, dt, t_save, t_on, t_off, tf,
              stab_dt=None, stab_tol=1e-6):
        """
        Parameters
        ----------
        dt     : float -- time step
        t_save : float -- snapshot interval
        t_on   : float -- start of active (Robin) window
        t_off  : float -- end of active window
        tf     : float -- final time
        stab_dt : float or None -- stabilization time step (default dt)
        stab_tol : float -- stabilization tolerance (default 1e-6)
        """
        if stab_dt is None:
            stab_dt = dt

        # 1. Stabilize
        print("[DHE] Phase 1/2 : Stabilizing initial condition ...")
        t0 = time.time()
        T0_stable, n_stab = self._solver.stabilize(
            self.T0_array, dt=stab_dt, tol=stab_tol)
        print(f"[DHE] Stabilized in {n_stab} iters ({time.time()-t0:.1f}s)")

        # 2. Solve
        print("[DHE] Phase 2/2 : Solving thermal evolution ...")
        raw = self._solver.solve(
            T0=T0_stable, dt=dt, tf=tf,
            t_save=t_save, t_on=t_on, t_off=t_off)

        T_list = [np.asarray(Ti).ravel() for Ti in raw["T"]]
        z_levels, T_surface = self._extract_surface(raw)

        return DHEResult(
            nodes=self.cylinder.nodes,
            elements=self.cylinder.elements,
            boundary_faces=self.cylinder.boundary_faces,
            R_min=self.R_min,
            z_min=self.z_min,
            T0_stable=T0_stable,
            times=raw["t"],
            T_snapshots=np.vstack(T_list),
            z_levels=z_levels,
            T_surface=T_surface,
        )

    def scan_toff(self, dt, t_save, t_on, t_off_array, tf,
                  stab_dt=None, stab_tol=1e-6):
        """
        Run one simulation per t_off value, sharing a single
        stabilization. Returns a ScanResult with the full
        borehole surface T(t, z) for each t_off.

        Parameters
        ----------
        dt, t_save, t_on, tf :
            Same as solve().
        t_off_array : array-like
            Sequence of t_off values to sweep.
        stab_dt, stab_tol :
            Stabilization parameters.

        Returns
        -------
        ScanResult
        """
        t_off_array = np.asarray(t_off_array, dtype=float)
        n_runs = len(t_off_array)

        if stab_dt is None:
            stab_dt = dt

        # 1. Stabilize once
        print(f"[scan] Phase 1/{n_runs+1} : Stabilizing ...")
        t0w = time.time()
        T0_stable, n_stab = self._solver.stabilize(
            self.T0_array, dt=stab_dt, tol=stab_tol)
        print(f"[scan] Stabilized in {n_stab} iters "
              f"({time.time()-t0w:.1f}s)")

        times_ref = None
        z_ref = None
        surfaces = []

        for i, t_off in enumerate(t_off_array):
            print(f"[scan] Run {i+1}/{n_runs}  t_off={t_off:.0f}")

            raw = self._solver.solve(
                T0=T0_stable, dt=dt, tf=tf,
                t_save=t_save, t_on=t_on, t_off=t_off)

            z_levels, T_surf = self._extract_surface(raw)

            if times_ref is None:
                times_ref = np.asarray(raw["t"], dtype=float)
                z_ref = z_levels
            surfaces.append(T_surf)

        T_borehole = np.stack(surfaces, axis=0)  # (n_toff, n_times, n_z)

        result = ScanResult(t_off_array, times_ref, z_ref, T_borehole)
        print(f"[scan] Done. {result}")
        return result
