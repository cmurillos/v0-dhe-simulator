# DHE-simulator

<div align="justify">

Este repositorio implementa una **libreria computacional data-driven** para la estimacion de *tiempos de recuperacion termica* en sistemas de **Downhole Heat Exchanger (DHE)**, basada en simulaciones numericas de conduccion de calor en medios geologicos heterogeneos. El nucleo del codigo esta disenado para ser **reproducible, extensible y fisicamente consistente**, y se apoya explicitamente en una **tabla de datos experimentales** suministrada por el usuario en formato CSV.

La libreria esta pensada como soporte metodologico para trabajos de **nivel de maestria** y proyectos de investigacion en geotermia somera y profunda, con especial enfasis en:

* el uso explicito de incertidumbre experimental,
* la reconstruccion de campos fisicos aleatorios,
* y la definicion operativa del tiempo de recuperacion termica.

---

### Entrada principal: tabla de propiedades del subsuelo

El **insumo central del codigo** es una tabla que contiene estimaciones estadisticas de propiedades termicas del subsuelo a distintas profundidades. Esta tabla se suministra en formato CSV y tiene la estructura

| z  | T  | var(T)  | rho  | var(rho)  | c  | var(c)  | k  | var(k)  |
| -- | -- | ------- | ---- | --------- | -- | ------- | -- | ------- |
| z1 | T1 | var(T1) | rho1 | var(rho1) | c1 | var(c1) | k1 | var(k1) |
| ...| ...| ...     | ...  | ...       | ...| ...     | ...| ...     |
| zn | Tn | var(Tn) | rhon | var(rhon) | cn | var(cn) | kn | var(kn) |

Cada fila corresponde a una profundidad (z_i) y reporta valores medios y varianzas asociadas a:

* densidad (rho),
* calor especifico (c),
* conductividad termica (k),
* temperatura estacionaria (T).

Estas cantidades reflejan la **incertidumbre inherente a la caracterizacion geologica** y constituyen la base para la generacion de campos espaciales heterogeneos.

---

### Dominio geometrico

El dominio de simulacion es un **cilindro tridimensional perforado**, definido como

$$\Omega = \Omega_{\mathrm{ext}} \setminus \Omega_{\mathrm{int}}$$

con

$$\Omega_{\mathrm{ext}} = \{(r,\theta,z) \in \mathbb{R}^3 : r < R_1,\; -L < z < 0\}$$

$$\Omega_{\mathrm{int}} = \{(r,\theta,z) \in \mathbb{R}^3 : r < R_2,\; -d < z < 0\}, \quad R_2 \ll R_1$$

Los parametros geometricos (R1, R2, L, d) son provistos por el usuario y permiten representar la region de subsuelo afectada por un pozo DHE.

---

### Reconstruccion de campos fisicos aleatorios

A partir de la tabla CSV se definen funciones **constantes a trozos** en la direccion vertical:

$$\rho_0(z),\; \mathrm{var}(\rho_0)(z),\; c_0(z),\; \mathrm{var}(c_0)(z),\; k_0(z),\; \mathrm{var}(k_0)(z),\; T_0(z),\; \mathrm{var}(T_0)(z)$$

Sea G un **campo aleatorio adimensional**, con media nula y varianza unitaria en cada seccion horizontal:

$$\langle G(\cdot,\cdot,z) \rangle_{xy} \approx 0, \qquad \mathrm{var}_{xy}(G(\cdot,\cdot,z)) \approx 1$$

Los campos fisicos se construyen entonces como

$$\rho(x,y,z) = \rho_0(z) + \mathrm{var}(\rho_0)(z)\,G(x,y,z)$$

$$c(x,y,z) = c_0(z) + \mathrm{var}(c_0)(z)\,G(x,y,z)$$

$$k(x,y,z) = k_0(z) + \mathrm{var}(k_0)(z)\,G(x,y,z)$$

$$T_0(x,y,z) = T_0(z) + \mathrm{var}(T_0)(z)\,G(x,y,z)$$

Esta construccion garantiza que los **promedios y varianzas en profundidad** de los campos reconstruidos reproduzcan los valores reportados en la tabla experimental. En la implementacion, el campo aleatorio G se maneja como un **objeto interno del generador**, lo que permite reutilizar realizaciones coherentes entre distintos coeficientes fisicos.

---

### Modelo termico

La evolucion de la temperatura se modela mediante la ecuacion de conduccion de calor con coeficientes heterogeneos

$$\rho(x)c(x)\,\partial_t T(x,t) = \nabla \cdot \big(k(x)\nabla T(x,t)\big), \quad (x,t) \in \Omega \times (0,t_f]$$

con condicion inicial

$$T(x,0) = T_0(x)$$

y condicion de frontera tipo Robin no homogenea sobre la perforacion:

$$\nabla T \cdot \eta = -\frac{\alpha(x,t)}{k(x)}\big(T(x,t)-T_{\mathrm{int}}(x,t)\big), \quad (x,t) \in \partial\Omega_{\mathrm{int}} \times (t_{on}, t_{off})$$

En el resto de la frontera se impone flujo nulo.

---

### Formato de salida

El metodo `solve()` retorna un objeto `DHEResult` que encapsula toda la geometria y la dinamica termica de la simulacion. El metodo `save(path)` exporta los resultados a un archivo `.npz` comprimido con la siguiente estructura:

| Clave             | Forma               | Descripcion                                           |
|-------------------|---------------------|-------------------------------------------------------|
| `nodes`           | `(N_nodes, 3)`      | Coordenadas (x, y, z) de cada nodo de la malla       |
| `elements`        | `(N_tets, 4)`       | Indices de los 4 nodos de cada tetraedro              |
| `boundary_faces`  | `(N_faces, 3)`      | Indices de los 3 nodos de cada triangulo de frontera  |
| `times`           | `(N_snapshots,)`    | Instantes de tiempo guardados                         |
| `T`               | `(N_snapshots, N_nodes)` | Temperatura en cada nodo para cada instante      |

Este formato es **no redundante** (la geometria se guarda una sola vez), **ligero** (compresion zip nativa de NumPy), y **completo** (permite reconstruir el campo T(t,x) sobre toda la malla sin perdida de informacion).

---

### Objetivo del codigo

El proposito de esta libreria es permitir:

1. Generar multiples realizaciones coherentes de campos fisicos aleatorios a partir de datos reales.
2. Ejecutar simulaciones termicas consistentes con la fisica del subsuelo.
3. Extraer funciones delta(t) y estimar estadisticamente los tiempos de recuperacion.
4. Servir como base para modelos reducidos y leyes empiricas de recuperacion termica en pozos DHE.

---

### Instalacion

```bash
pip install git+https://github.com/cmurillos/v0-dhe-simulator.git@thermal-simulation-library
```

### Dependencias

- Python >= 3.9
- numpy >= 1.22
- scipy >= 1.9
- pandas >= 1.5
- matplotlib >= 3.5
- scikit-fem >= 8.0

### Uso rapido

```python
import numpy as np
from dhe_simulator import DHE_simulation

dhe = DHE_simulation('/content/perfiles_5_capas.csv', 1.0, 100.0, 150, 200)
T_c = lambda t, x, y, z: np.full_like(x, 300.0)
res = dhe.solve(dt=20000, t_save=80000, t_on=600000, t_off=1000000, tf=2000000, T_c=T_c)

# Guardar resultados completos
res.save('simulation_output.npz')

# Cargar resultados
data = np.load('simulation_output.npz')
print(data['nodes'].shape, data['T'].shape)
```

---

### Referencias

1. G. Culver & J. W. Lund, *Downhole Heat Exchangers*, 1999.
2. Y. Zhang et al., *Applied Thermal Engineering* **163** (2019).
3. P. Chi et al., *Energies* **18** (2025).
4. D. R. Poirier & G. H. Geiger, *Transport Phenomena in Materials Processing*, Springer (2016).
5. Y. Zhao et al., *Sustainability* **12** (2020).

</div>
