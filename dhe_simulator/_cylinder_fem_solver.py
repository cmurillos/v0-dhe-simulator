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
    Solver FEM para la ecuacion de calor en mallas cilindricas con condicion
    de frontera tipo Robin y modelo de fluido convectivo interno:

        -k grad(T).n = h (T_rock - T_f)

    donde T_f(t,z) se calcula a partir de la temperatura promedio de
    la roca en la pared del pozo usando:

        T_f(z) = e^{-beta*z} T_in + beta int_0^z e^{-beta(z-s)} T_r(s) ds

    con beta = 2h / (rho_f * R * v_f * c_f).
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
        self.h = None          # coeficiente convectivo
        self.R_min = None
        self.eps = None

        # Fluid model
        self.beta = None
        self.T_in = None
        self._bh_z_levels = None   # sorted unique z-levels on borehole
        self._bh_node_groups = None  # dict: z_level -> list of node indices

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

    # ------------------------------------------------------------------
    # Ensamblaje
    # ------------------------------------------------------------------
    def assemble_system(self, p_func, c_func, k_func,
                        h, R_min, eps,
                        rho_f, c_f, v_f, T_in):
        """
        Ensambla las matrices de masa M, rigidez K, y Robin R,
        y configura el modelo de fluido convectivo.

        Parametros
        ----------
        p_func : callable   -- densidad roca rho(x)
        c_func : callable   -- capacidad calorifica roca c(x)
        k_func : callable   -- conductividad termica roca k(x)
        h      : float      -- coeficiente convectivo interno [W/(m^2 K)]
        R_min  : float      -- radio del pozo (frontera interior)
        eps    : float      -- tolerancia geometrica para caras interiores
        rho_f  : float      -- densidad del fluido [kg/m^3]
        c_f    : float      -- calor especifico del fluido [J/(kg K)]
        v_f    : float      -- velocidad axial del fluido [m/s]
        T_in   : float      -- temperatura de entrada del fluido [K]
        """
        self.h = h
        self.R_min = R_min
        self.eps = eps
        self.T_in = T_in
        self.beta = 2.0 * h / (rho_f * R_min * v_f * c_f)

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
            return h * mask * u * v

        self.M = asm(mass, self.basis).tocsc()
        self.K = asm(stiffness, self.basis).tocsc()
        self.R = asm(robin, self.boundary_basis).tocsc()

        # Pre-compute borehole node groups by z-level
        self._build_borehole_groups()

    def _build_borehole_groups(self):
        """
        Identify mesh nodes on the borehole wall (r ~ R_min, z >= z_min
        where z_min is inferred as the smallest z among those nodes)
        and group them by z-level for fast angular averaging.
        """
        nodes = self.skfem_nodes
        r = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
        tol_r = 0.5 * self.R_min
        bh_mask = np.abs(r - self.R_min) < tol_r

        bh_indices = np.where(bh_mask)[0]
        bh_z = nodes[bh_indices, 2]

        # round to avoid floating-point duplicates
        decimals = 6
        bh_z_rounded = np.round(bh_z, decimals)
        unique_z = np.unique(bh_z_rounded)

        groups = {}
        for z_val in unique_z:
            groups[z_val] = bh_indices[bh_z_rounded == z_val]

        self._bh_z_levels = unique_z          # sorted
        self._bh_node_groups = groups

    def _compute_Tf(self, T_current):
        """
        Compute the fluid temperature T_f(z) at each borehole z-level
        given the current rock temperature field T_current.

        Uses trapezoidal integration of:
            T_f(z) = e^{-beta z} T_in + beta int_0^z e^{-beta(z-s)} T_r(s) ds

        Returns
        -------
        z_levels : ndarray (n_levels,)
        Tf_vals  : ndarray (n_levels,)
        """
        z = self._bh_z_levels
        n = len(z)
        beta = self.beta

        # T_r(z) = simple average of T at borehole nodes at each z
        Tr = np.empty(n)
        for i, zv in enumerate(z):
            idx = self._bh_node_groups[zv]
            Tr[i] = T_current[idx].mean()

        # Trapezoidal integration of  beta * int_0^z e^{-beta(z-s)} Tr(s) ds
        Tf = np.empty(n)
        Tf[0] = np.exp(-beta * z[0]) * self.T_in
        # add contribution from integrand at s=0: Tr(z[0]) weighted
        if n > 0:
            integral = 0.0
            for i in range(n):
                exp_z = np.exp(-beta * z[i])
                if i == 0:
                    integral = 0.0
                else:
                    dz = z[i] - z[i - 1]
                    # trapezoidal: f(s_{i-1}) and f(s_i)
                    # integrand at s: e^{beta*s} * Tr(s)
                    f_prev = np.exp(beta * z[i - 1]) * Tr[i - 1]
                    f_curr = np.exp(beta * z[i]) * Tr[i]
                    integral += 0.5 * dz * (f_prev + f_curr)
                Tf[i] = exp_z * self.T_in + beta * exp_z * integral

        return z, Tf

    def _assemble_robin_load_fluid(self, T_current):
        """
        Ensambla el vector de carga Robin usando el modelo de fluido:

            F_robin = integral_dOmega h * T_f(x) * v  dS

        donde T_f se calcula internamente a partir de T_current.
        """
        z_levels, Tf_vals = self._compute_Tf(T_current)

        # Build interpolator: for any z, get T_f by nearest z-level
        from scipy.interpolate import interp1d
        if len(z_levels) > 1:
            Tf_interp = interp1d(z_levels, Tf_vals,
                                 kind='linear', fill_value='extrapolate')
        else:
            Tf_interp = lambda z: np.full_like(np.asarray(z, dtype=float),
                                                Tf_vals[0])

        r_threshold_sq = (self.R_min + self.eps) ** 2
        h_val = self.h

        @LinearForm
        def robin_load(v, w):
            r2 = w.x[0] ** 2 + w.x[1] ** 2
            mask = (r2 <= r_threshold_sq).astype(float)
            Tf_at_z = Tf_interp(w.x[2])
            return h_val * mask * Tf_at_z * v

        return asm(robin_load, self.boundary_basis)

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
    def solve(self, T0, dt, tf, t_save, t_on, t_off):
        """
        Resuelve la ecuacion de calor con condicion Robin activa en
        [t_on, t_off] usando el modelo de fluido convectivo interno:

            rho c dT/dt = div(k grad T)
            -k grad(T).n = h (T_rock - T_f(z))   en la pared del pozo

        donde T_f(z) se calcula internamente a partir de T_rock.

        Euler implicito:
          - Activo  (t_on < t < t_off):
              (M + dt*K + dt*R) T^{n+1} = M*T^n + dt*F_robin(T^n)
          - Inactivo (frontera aislada):
              (M + dt*K) T^{n+1} = M*T^n

        Parametros
        ----------
        T0     : ndarray  -- temperatura inicial en cada nodo
        dt     : float    -- paso de tiempo
        tf     : float    -- tiempo final
        t_save : float    -- intervalo de guardado
        t_on   : float    -- inicio de la ventana activa
        t_off  : float    -- fin de la ventana activa

        Retorna
        -------
        dict con claves 't' (lista de tiempos) y 'T' (lista de arrays).
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
                F_robin = self._assemble_robin_load_fluid(T)
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
