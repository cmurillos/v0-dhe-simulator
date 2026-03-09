import sys
import time
import numpy as np
from skfem import MeshTet, ElementTetP1, Basis, asm, BilinearForm, LinearForm, helpers
from scipy.sparse.linalg import splu


def _fmt_time(seconds):
    """Format seconds into a compact human string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _bar(fraction, width=30):
    """Return a compact progress bar string."""
    filled = int(width * fraction)
    return "\u2588" * filled + "\u2591" * (width - filled)


class CylinderFEMSolver:
    """
    Solver FEM para la ecuación de calor en mallas cilíndricas con condición
    de frontera tipo Robin:

        -k * grad(T) · n = alpha(t,x) * (T - T_c(t,x))

    donde alpha(t,x) = alpha_val  si x1²+x2² <= (R_min+eps)² y t_on < t < t_off,
                      = 0          en otro caso.

    Compatible con funciones lambda f(x) (x array (3,N))
    y con funciones de GeneradorCamposFisicos f(x, y, z).
    """

    def __init__(self, nodes, elements, boundary_faces):
        nodes_c = np.ascontiguousarray(nodes.T)
        elements_c = np.ascontiguousarray(elements.T)

        self.mesh = MeshTet(nodes_c, elements_c)
        self.basis = Basis(self.mesh, ElementTetP1())
        self.boundary_basis = self.basis.boundary()
        self.skfem_nodes = self.mesh.p.T

        self.M = None
        self.K = None
        self.R = None          # Matriz Robin de frontera
        self.alpha_val = None
        self.R_min = None
        self.eps = None

    # ------------------------------------------------------------------
    # Wrappers para compatibilidad con GeneradorCamposFisicos
    # ------------------------------------------------------------------
    def _wrap_field_function(self, func):
        """
        Envuelve una función de campo para que siempre acepte w.x de skfem
        (array de forma (3, N)).

        Soporta:
          - f(x, y, z)       con x, y, z arrays de forma (N,)  [GeneradorCamposFisicos]
          - f(coords)         con coords de forma (3, N)
        """
        def wrapped(coords):
            N = coords.shape[1]
            x, y, z = coords[0], coords[1], coords[2]
            # Intentar primero f(x, y, z) — patron de GeneradorCamposFisicos
            try:
                result = func(x, y, z)
            except TypeError:
                # Fallback: f(coords) con coords de forma (3, N)
                result = func(coords)

            if np.isscalar(result):
                return np.full((N, 1), float(result)) # Reshape to (N, 1) for broadcasting
            result = np.asarray(result, dtype=float)
            if result.ndim == 1: # If it's a 1D array (N,), reshape to (N, 1)
                return result.reshape(-1, 1)
            elif result.ndim == 2 and result.shape[1] == 1:
                return result # Already (N, 1)
            elif result.ndim == 2 and result.shape[0] == N:
                # If it's (N, M) where M > 1, this implies the func returns multiple values per point.
                # For scalar coefficients, it should be (N, 1).
                # Assuming the first column is the intended scalar value if such a case occurs.
                return result[:, 0].reshape(-1, 1)
            else:
                raise ValueError(f"Wrapped function returned incompatible shape {result.shape} for N={N}. Expected (N,) or (N,1).")
        return wrapped

    def _wrap_Tc_function(self, T_c_func, t):
        """
        Envuelve T_c(t, …) en una función solo espacial para un instante t fijo.

        Soporta:
          - T_c(t, coords)      con coords de forma (3, N)
          - T_c(t, x, y, z)     con x, y, z arrays de forma (N,)
        """
        def wrapped(coords):
            try:
                result = T_c_func(t, coords)
                if np.isscalar(result):
                    return np.full(coords.shape[1], float(result))
                result = np.asarray(result, dtype=float)
                if result.shape == (coords.shape[1],):
                    return result
                raise ValueError
            except (TypeError, IndexError, ValueError):
                x, y, z = coords[0], coords[1], coords[2]
                result = T_c_func(t, x, y, z)
                if np.isscalar(result):
                    return np.full(len(x), float(result))
                return np.asarray(result, dtype=float)
        return wrapped

    # ------------------------------------------------------------------
    # Ensamblaje
    # ------------------------------------------------------------------
    def assemble_system(self, p_func, c_func, k_func, alpha_val, R_min, eps):
        """
        Ensambla las matrices de masa M, rigidez K, y Robin R.

        Parámetros
        ----------
        p_func : callable   – densidad ρ(x)
        c_func : callable   – capacidad calorífica c(x)
        k_func : callable   – conductividad térmica k(x)
        alpha_val : float    – valor constante de α en la región activa
        R_min : float        – radio mínimo (frontera interior)
        eps : float          – tolerancia para seleccionar caras interiores
        """
        self.alpha_val = alpha_val
        self.R_min = R_min
        self.eps = eps

        p_w = self._wrap_field_function(p_func)
        c_w = self._wrap_field_function(c_func)
        k_w = self._wrap_field_function(k_func)

        r_threshold_sq = (R_min + eps) ** 2

        @BilinearForm
        def mass(u, v, w):
            return p_w(w.x) * c_w(w.x) * u * v

        @BilinearForm
        def stiffness(u, v, w):
            return k_w(w.x) * helpers.dot(u.grad, v.grad)

        @BilinearForm
        def robin(u, v, w):
            r2 = w.x[0] ** 2 + w.x[1] ** 2
            mask = (r2 <= r_threshold_sq).astype(float)
            return alpha_val * mask * u * v

        self.M = asm(mass, self.basis).tocsc()
        self.K = asm(stiffness, self.basis).tocsc()
        self.R = asm(robin, self.boundary_basis).tocsc()

    def _assemble_robin_load(self, t, T_c_func):
        """
        Ensambla el vector de carga Robin en el instante t:

            F_robin = ∫_∂Ω α(x) · T_c(t, x) · v  dS
        """
        Tc_w = self._wrap_Tc_function(T_c_func, t)
        r_threshold_sq = (self.R_min + self.eps) ** 2
        alpha_val = self.alpha_val

        @LinearForm
        def robin_load(v, w):
            r2 = w.x[0] ** 2 + w.x[1] ** 2
            mask = (r2 <= r_threshold_sq).astype(float)
            return alpha_val * mask * Tc_w(w.x) * v

        return asm(robin_load, self.boundary_basis)

    # ------------------------------------------------------------------
    # Borehole profile extraction
    # ------------------------------------------------------------------
    def build_borehole_groups(self, R_min, z_min):
        """
        Identify borehole nodes (r ~ R_min, z >= z_min) and group by z-level.
        Must be called before borehole_profile().
        """
        nodes = self.skfem_nodes
        r = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
        tol_r = 0.5 * R_min
        bh_mask = (np.abs(r - R_min) < tol_r) & (nodes[:, 2] >= z_min - 1e-6)
        bh_indices = np.where(bh_mask)[0]
        if len(bh_indices) == 0:
            raise RuntimeError(f"No borehole nodes found (R_min={R_min}, z_min={z_min})")
        bh_z = nodes[bh_indices, 2]
        bh_z_rounded = np.round(bh_z, 6)
        unique_z = np.unique(bh_z_rounded)
        groups = {zv: bh_indices[bh_z_rounded == zv] for zv in unique_z}
        self._bh_z_levels = unique_z
        self._bh_node_groups = groups

    def borehole_profile(self, T_field):
        """
        Angular-average temperature on the borehole at each z-level.
        Returns (z_levels, T_r) both 1D arrays.
        """
        z = self._bh_z_levels
        Tr = np.array([T_field[self._bh_node_groups[zv]].mean() for zv in z])
        return z, Tr

    # ------------------------------------------------------------------
    # Estabilización (frontera aislada)
    # ------------------------------------------------------------------
    def stabilize(self, T0, dt, tol=1e-6, max_iter=100000):
        """
        Evoluciona el sistema con gradiente de temperatura cero en la
        frontera (condicion aislada) hasta que la temperatura ya no
        cambie apreciablemente entre pasos.

        Reutiliza las matrices ``M`` y ``K`` ya ensambladas (no re-
        ensambla nada).

        Criterio de parada:

        .. math::

            \\frac{\\|T^{n+1} - T^n\\|}{\\|T^n\\| + \\epsilon} < \\text{tol}

        Parameters
        ----------
        T0 : ndarray, shape (N_nodes,)
            Temperatura inicial (p.ej. del CSV experimental).
        dt : float
            Paso de tiempo para la relajacion.
        tol : float, optional
            Tolerancia relativa de convergencia (default 1e-6).
        max_iter : int, optional
            Maximo de iteraciones (default 100000).

        Returns
        -------
        T_stable : ndarray, shape (N_nodes,)
            Campo de temperatura estabilizado.
        n_iter : int
            Numero de iteraciones realizadas.
        """
        if self.M is None or self.K is None:
            raise RuntimeError(
                "Debe llamar assemble_system() antes de stabilize()."
            )

        A = (self.M + dt * self.K).tocsc()
        solve = splu(A).solve

        T = T0.copy().ravel()
        t0_wall = time.time()
        rel_change = 1.0

        for n in range(1, max_iter + 1):
            b = self.M @ T
            T_new = np.asarray(solve(b)).ravel()

            norm_T = np.linalg.norm(T)
            diff = np.linalg.norm(T_new - T)
            rel_change = diff / (norm_T + 1e-30)

            if n % 20 == 0 or rel_change < tol:
                # log-scale progress: tol .. 1 mapped to 0% .. 100%
                import math
                log_rc = math.log10(max(rel_change, tol))
                log_tol = math.log10(tol)
                frac = max(0.0, min(1.0, 1.0 - log_rc / log_tol))
                elapsed = time.time() - t0_wall
                eta = (elapsed / max(frac, 1e-6)) * (1 - frac) if frac > 0.01 else 0
                sys.stdout.write(
                    f"\r  Stabilizing {_bar(frac)} {frac*100:5.1f}%"
                    f"  iter {n}  rel={rel_change:.1e}"
                    f"  [{_fmt_time(elapsed)}<{_fmt_time(eta)}]   "
                )
                sys.stdout.flush()

            if rel_change < tol:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return T_new, n

            T = T_new

        sys.stdout.write("\n")
        sys.stdout.flush()
        print(f"  [stabilize] Warning: max_iter={max_iter} reached "
              f"(last rel change = {rel_change:.2e})")
        return T, max_iter

    # ------------------------------------------------------------------
    # Resolución temporal
    # ------------------------------------------------------------------
    def solve(self, T0, dt, tf, t_save, T_c_func, t_on, t_off):
        """
        Resuelve la ecuación de calor con condición Robin activa en [t_on, t_off]:

            ρ c ∂T/∂t = div(k grad T)
            -k grad(T)·n = α(t,x)(T - T_c(t,x))   en ∂Ω

        Euler implícito:
          ● Activo  (t_on < t < t_off):
              (M + dt·K + dt·R) T^{n+1} = M·T^n + dt·F_robin(t^{n+1})
          ● Inactivo (frontera aislada):
              (M + dt·K) T^{n+1} = M·T^n

        Parámetros
        ----------
        T0       : ndarray  – temperatura inicial en cada nodo
        dt       : float    – paso de tiempo
        Tf       : float    – tiempo final
        t_save   : float    – intervalo de guardado de resultados
        T_c_func : callable – T_c(t, x, y, z)  o  T_c(t, coords)
        t_on     : float    – inicio de la ventana activa
        t_off    : float    – fin de la ventana activa

        Retorna
        -------
        dict  con claves 't' (lista de tiempos) y 'T' (lista de arrays de temperatura).
        """
        if self.M is None or self.K is None or self.R is None:
            raise RuntimeError(
                "Debe llamar assemble_system() antes de solve()."
            )

        # Pre-factorizar ambos sistemas
        A_inactive = (self.M + dt * self.K).tocsc()
        A_active = (self.M + dt * self.K + dt * self.R).tocsc()

        solve_inactive = splu(A_inactive).solve
        solve_active = splu(A_active).solve

        T = T0.copy()
        results = {"t": [0.0], "T": [T.copy()]}
        t = 0.0
        next_save = t_save

        total_steps = int(np.ceil(tf / dt))
        step = 0
        t0_wall = time.time()

        while t < tf - 1e-12:
            t_next = t + dt
            step += 1

            if t_on < t_next < t_off:
                F_robin = self._assemble_robin_load(t_next, T_c_func)
                b = self.M @ T + dt * F_robin
                T = np.asarray(solve_active(b)).ravel()
            else:
                b = self.M @ T
                T = np.asarray(solve_inactive(b)).ravel()

            t = t_next
            if t >= next_save - 1e-12:
                results["t"].append(t)
                results["T"].append(T.copy())
                next_save += t_save

            # progress every 10 steps or last step
            if step % 10 == 0 or step == total_steps:
                frac = min(step / total_steps, 1.0)
                elapsed = time.time() - t0_wall
                eta = (elapsed / max(frac, 1e-6)) * (1 - frac) if frac > 0.01 else 0
                phase = "Robin" if t_on < t < t_off else "Adiab"
                sys.stdout.write(
                    f"\r  Solving     {_bar(frac)} {frac*100:5.1f}%"
                    f"  step {step}/{total_steps}  t={t:.0f}"
                    f"  [{phase}]"
                    f"  [{_fmt_time(elapsed)}<{_fmt_time(eta)}]   "
                )
                sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()
        return results

    def export_xdmf(self, results, filename="output"):
        """
        Export simulation results to XDMF + HDF5 format.
        Creates filename.xdmf and filename.h5
        """
        import h5py

        times = results["t"]
        T_list = results["T"]
        n_times = len(times)
        n_nodes = len(T_list[0])
        n_cells = self.mesh.nelements

        h5_path = f"{filename}.h5"
        xdmf_path = f"{filename}.xdmf"

        # Write HDF5
        with h5py.File(h5_path, "w") as h5:
            # Mesh data
            h5.create_dataset("nodes", data=self.mesh.p.T)  # (n_nodes, 3)
            h5.create_dataset("cells", data=self.mesh.t.T)  # (n_cells, 4)
            # Temperature at each time
            for i, T in enumerate(T_list):
                h5.create_dataset(f"T_{i}", data=np.asarray(T).ravel())

        # Write XDMF
        xdmf = ['<?xml version="1.0"?>',
                '<Xdmf Version="3.0">',
                '<Domain>',
                '<Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">']

        for i, t in enumerate(times):
            xdmf.append(f'  <Grid Name="mesh" GridType="Uniform">')
            xdmf.append(f'    <Time Value="{t}"/>')
            xdmf.append(f'    <Topology TopologyType="Tetrahedron" NumberOfElements="{n_cells}">')
            xdmf.append(f'      <DataItem Dimensions="{n_cells} 4" Format="HDF">{h5_path}:/cells</DataItem>')
            xdmf.append(f'    </Topology>')
            xdmf.append(f'    <Geometry GeometryType="XYZ">')
            xdmf.append(f'      <DataItem Dimensions="{n_nodes} 3" Format="HDF">{h5_path}:/nodes</DataItem>')
            xdmf.append(f'    </Geometry>')
            xdmf.append(f'    <Attribute Name="Temperature" AttributeType="Scalar" Center="Node">')
            xdmf.append(f'      <DataItem Dimensions="{n_nodes}" Format="HDF">{h5_path}:/T_{i}</DataItem>')
            xdmf.append(f'    </Attribute>')
            xdmf.append(f'  </Grid>')

        xdmf.extend(['</Grid>', '</Domain>', '</Xdmf>'])

        with open(xdmf_path, "w") as f:
            f.write("\n".join(xdmf))

        print(f"[export] Wrote {xdmf_path} + {h5_path}")
