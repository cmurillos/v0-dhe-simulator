import numpy as np

from ._cylinder_mesh import CylinderMesh
from ._cylinder_fem_solver import CylinderFEMSolver
from ._generador_campos import GeneradorCamposFisicos


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
        res = self._solver.solve(
            T0=self.T0_array,
            dt=dt,
            tf=tf,
            t_save=t_save,
            T_c_func=T_c,
            t_on=t_on,
            t_off=t_off)
        return res
