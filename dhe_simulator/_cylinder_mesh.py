import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CylinderMesh:
    def __init__(self, R_vals, Z_vals, z_min, N_theta=30):
        self.R_vals = np.asarray(R_vals)
        self.Z_vals = np.asarray(Z_vals)
        self.N_theta = N_theta
        self.z_min = z_min
        self.nodes = None
        self.elements = None
        self.boundary_faces = None
        self.generate_nodes()
        self.regularize_nodes()
        self.build_simplices()
        self.clean_mesh()


    # --------------------------------------------------
    # 1. Generación de nodos (determinista)
    # --------------------------------------------------
    def generate_nodes(self):
        box = []
        dtheta = 2 * np.pi / self.N_theta

        for i, R in enumerate(self.R_vals):
            for j, Z in enumerate(self.Z_vals):
                # Alternancia angular (staggering determinista)
                phase = 0.5 * dtheta if j % 2 else 0.0

                for k in range(self.N_theta):
                    theta = dtheta * k + phase
                    box.append([
                        R * np.cos(theta),
                        R * np.sin(theta),
                        Z
                    ])

                # Cierre sólido en la base
                if Z <= self.z_min:
                    box.append([0.0, 0.0, Z])

        self.nodes = np.array(box)
        return self.nodes

    # --------------------------------------------------
    # 2. Filtrado de simplejos
    # --------------------------------------------------
    def build_simplices(self):
        tri = Delaunay(self.nodes)
        elements = tri.simplices

        centers = np.mean(self.nodes[elements], axis=1)
        radii = np.linalg.norm(centers[:, :2], axis=1)
        z_c = centers[:, 2]

        tol = 1e-2
        mask = (
            (z_c <= self.z_min + tol) |
            ((radii >= self.R_vals.min() - tol) &
             (radii <= self.R_vals.max() + tol))
        )

        self.elements = elements[mask]

        # frontera
        face_count = {}
        for tet in self.elements:
            faces = [
                tuple(sorted(tet[[0, 1, 2]])),
                tuple(sorted(tet[[0, 1, 3]])),
                tuple(sorted(tet[[0, 2, 3]])),
                tuple(sorted(tet[[1, 2, 3]]))
            ]
            for f in faces:
                face_count[f] = face_count.get(f, 0) + 1

        self.boundary_faces = np.array(
            [f for f, c in face_count.items() if c == 1]
        )

        return self.elements, self.boundary_faces

    # --------------------------------------------------
    # 3. Visualización
    # --------------------------------------------------
    def plot(self,figsize=(10,8),facecolor="tomato",alpha=0.6,edgecolor="k",lw=0.3,title=None,show_axes=False,view=None):
        fig=plt.figure(figsize=figsize)
        ax=fig.add_subplot(111,projection="3d")
        poly=Poly3DCollection(self.nodes[self.boundary_faces],alpha=alpha,edgecolor=edgecolor,linewidths=lw)
        poly.set_facecolor(facecolor)
        ax.add_collection3d(poly)
        ext=np.array([self.nodes.min(0),self.nodes.max(0)])
        ax.set_xlim(ext[:,0]); ax.set_ylim(ext[:,1]); ax.set_zlim(ext[:,2])
        if view: ax.view_init(*view)
        if title: ax.set_title(title)
        if not show_axes: ax.set_axis_off()
        plt.tight_layout()
        plt.show()


            # --------------------------------------------------
    # 4. Regularización determinista de nodos
    # --------------------------------------------------
    def regularize_nodes(self):
        """
        Redistribución determinista para mejorar isotropía
        de volúmenes (Δr ≈ rΔθ ≈ Δz)
        """
        nodes = self.nodes.copy()

        # reconstruimos por capas
        new_nodes = []

        dtheta = 2 * np.pi / self.N_theta
        dz = np.mean(np.diff(self.Z_vals))

        for Z in self.Z_vals:
            # radios óptimos según métrica cilíndrica
            r0 = self.R_vals.min()
            r1 = self.R_vals.max()

            Nr = int((r1 - r0) / dz) + 1
            R_opt = np.linspace(r0, r1, Nr)

            for r in R_opt:
                # fase angular determinista dependiente de z
                phase = (Z / dz) * 0.5 * dtheta

                for k in range(self.N_theta):
                    theta = k * dtheta + phase
                    new_nodes.append([
                        r * np.cos(theta),
                        r * np.sin(theta),
                        Z
                    ])

            if Z <= self.z_min:
                new_nodes.append([0.0, 0.0, Z])

        self.nodes = np.array(new_nodes)
        return self.nodes

    def clean_mesh(self, min_volume=1e-10):
        """
        Elimina tetraedros degenerados y nodos huérfanos.
        Re-indexa los elementos y las caras de la frontera.
        """
        if self.elements is None or len(self.elements) == 0:
            return

        # --- 1. Eliminar tetraedros con volumen casi nulo (degenerados) ---
        pts = self.nodes[self.elements]
        # Fórmula del volumen: |(a-d) . ((b-d) x (c-d))| / 6
        v1 = pts[:, 0] - pts[:, 3]
        v2 = pts[:, 1] - pts[:, 3]
        v3 = pts[:, 2] - pts[:, 3]
        volumes = np.abs(np.einsum('ij,ij->i', v1, np.cross(v2, v3))) / 6.0

        valid_tets_mask = volumes > min_volume
        self.elements = self.elements[valid_tets_mask]

        # --- 2. Eliminar nodos que no pertenecen a ningún tetraedro ---
        # Encontrar índices de nodos únicos que sí se usan
        used_nodes_indices = np.unique(self.elements)

        # Crear un array de mapeo: viejo_indice -> nuevo_indice
        # Inicializamos con -1 para detectar errores
        mapping = np.full(len(self.nodes), -1, dtype=int)
        mapping[used_nodes_indices] = np.arange(len(used_nodes_indices))

        # --- 3. Actualizar la malla ---
        # Re-indexar elementos
        self.elements = mapping[self.elements]

        # Re-indexar caras de frontera (si existen)
        if self.boundary_faces is not None:
            # Solo conservamos caras cuyos nodos existan en el mapeo
            self.boundary_faces = mapping[self.boundary_faces]
            # Si alguna cara quedó con un -1 (aunque no debería pasar), la filtramos
            mask_faces = np.all(self.boundary_faces != -1, axis=1)
            self.boundary_faces = self.boundary_faces[mask_faces]

        # Finalmente, recortar la lista de nodos
        self.nodes = self.nodes[used_nodes_indices]
