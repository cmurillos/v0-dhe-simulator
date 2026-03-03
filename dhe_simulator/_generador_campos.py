import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.integrate import simpson


class GeneradorCamposFisicos:
    def __init__(self, ruta_csv, nodes):
        self._datos = self._cargar_datos(ruta_csv)
        self._perfiles = self._generar_perfiles_fisicos()
        self.G = self._generate_G(nodes)
        self.p, self.c, self.k, self.T = self._configurar_funciones_finales()

    def _cargar_datos(self, ruta):
        df = pd.read_csv(ruta)
        return [df[col].values for col in df.columns]

    def _piecewise_generator(self,X,Y):
        X=np.asarray(np.unique([0,*X]))
        Y=np.asarray(Y)
        return lambda x: Y[np.clip(np.searchsorted(X,np.asarray(x),side="right")-1, 0, len(Y)-1)]


    def _generar_perfiles_fisicos(self):
        Z, T, vT, p, vp, c, vc, k, vk = self._datos
        return {
            'T': (self._piecewise_generator(Z, T), self._piecewise_generator(Z, vT)),
            'p': (self._piecewise_generator(Z, p), self._piecewise_generator(Z, vp)),
            'c': (self._piecewise_generator(Z, c), self._piecewise_generator(Z, vc)),
            'k': (self._piecewise_generator(Z, k), self._piecewise_generator(Z, vk))
        }

    def _stats_z(self, f, z):
        r, th = np.linspace(0, self._R, 50), np.linspace(0, 2*np.pi, 50)
        rr, tth = np.meshgrid(r, th)
        v = f(rr * np.cos(tth), rr * np.sin(tth), z)
        m = simpson(simpson(v * r, x=r), x=th) / (np.pi * self._R**2)
        s = np.sqrt(simpson(simpson(((v - m)**2) * r, x=r), x=th) / (np.pi * self._R**2))
        return m, s

    def _generate_G(self,nodes):
        tree=cKDTree(nodes)
        Y=np.random.normal(size=len(nodes))
        return np.vectorize(lambda x,y,z: Y[tree.query(np.column_stack(np.broadcast_arrays(x,y,z)))[1]])


    def _configurar_funciones_finales(self):
        def gen(key):
            m_s, v_s = self._perfiles[key]
            return lambda x, y, z: m_s(z) + v_s(z) * self.G(x, y, z)
        return gen('p'), gen('c'), gen('k'), gen('T')

    def stats_campos_fisicos(self, z):
        res = {}
        fields = {'p': self.p, 'c': self.c, 'k': self.k, 'T': self.T}
        for n, f in fields.items():
            m, s = self._stats_z(f, z)
            res[n] = {'media': m, 'std': s}
        return res
