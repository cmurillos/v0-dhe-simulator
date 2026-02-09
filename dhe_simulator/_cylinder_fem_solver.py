import numpy as np
from skfem import MeshTet, ElementTetP1, Basis, asm, BilinearForm, LinearForm, helpers
from scipy.sparse.linalg import splu


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

        while t < tf - 1e-12:
            t_next = t + dt

            if t_on < t_next < t_off:
                # Robin activo: re-ensamblar carga porque T_c depende de t
                F_robin = self._assemble_robin_load(t_next, T_c_func)
                b = self.M @ T + dt * F_robin
                T = np.asarray(solve_active(b)).ravel()
            else:
                # Frontera aislada (α = 0)
                b = self.M @ T
                T = np.asarray(solve_inactive(b)).ravel()

            t = t_next
            if t >= next_save - 1e-12:
                results["t"].append(t)
                results["T"].append(T.copy())
                next_save += t_save

        return results
