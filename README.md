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

$$\Omega_{\mathrm{ext}} = \{(r,\theta,z) \in \mathbb{R}^3 : r < R_{\max},\; 0 < z < z_{\max}\}$$

$$\Omega_{\mathrm{int}} = \{(r,\theta,z) \in \mathbb{R}^3 : r < R_{\min},\; z_{\min} < z < z_{\max}\}$$

donde $R_{\min} \ll R_{\max}$. Los parametros geometricos son provistos por el usuario y permiten representar la region de subsuelo afectada por un pozo DHE.

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

---

### Modelo termico

La evolucion de la temperatura en la roca se modela mediante la ecuacion de conduccion de calor con coeficientes heterogeneos

$$\rho(x)c(x)\,\partial_t T(x,t) = \nabla \cdot \big(k(x)\nabla T(x,t)\big), \quad (x,t) \in \Omega \times (0,t_f]$$

con condicion inicial estabilizada y condicion de frontera tipo Robin sobre la perforacion:

$$-k\,\nabla T \cdot \hat{n} = h\big(T_{\mathrm{rock}} - T_f(t,z)\big), \quad (x,t) \in \partial\Omega_{\mathrm{int}} \times (t_{\mathrm{on}}, t_{\mathrm{off}})$$

En el resto de la frontera se impone flujo nulo (adiabatica).

---

### Modelo de fluido convectivo

La temperatura del fluido $T_f(t,z)$ dentro del pozo se calcula internamente a partir de la temperatura de la roca en la pared, usando un balance de energia en el flujo axial:

$$T_r(t,z) = \frac{1}{2\pi}\int_0^{2\pi} T(t, R_{\min}\sin\theta, R_{\min}\cos\theta, z)\,d\theta$$

$$\beta = \frac{2h}{\rho_f\, R_{\min}\, v_f\, c_f}$$

$$T_f(t,z) = e^{-\beta(z-z_{\min})}\,T_{\mathrm{in}} + \beta\int_{z_{\min}}^{z} e^{-\beta(z-s)}\,T_r(t,s)\,ds$$

donde:
* $h$ es el coeficiente convectivo interno [W/(m^2 K)],
* $\rho_f$ es la densidad del fluido [kg/m^3],
* $c_f$ es el calor especifico del fluido [J/(kg K)],
* $v_f$ es la velocidad axial del fluido [m/s],
* $T_{\mathrm{in}}$ es la temperatura de entrada del fluido [K].

En la implementacion, $T_r(z)$ se calcula como el **promedio aritmetico** de la temperatura en los nodos de la malla sobre la pared del pozo a cada nivel $z$, y la integral se evalua con **cuadratura trapezoidal**.

---

### Formato de salida

#### Simulacion individual (`DHEResult`)

El metodo `solve()` retorna un objeto `DHEResult` con:

| Clave             | Forma                    | Descripcion                                      |
|-------------------|--------------------------|--------------------------------------------------|
| `nodes`           | `(N_nodes, 3)`           | Coordenadas (x, y, z) de cada nodo               |
| `elements`        | `(N_tets, 4)`            | Conectividad tetraedrica                          |
| `boundary_faces`  | `(N_faces, 3)`           | Caras triangulares de frontera                    |
| `T0_stable`       | `(N_nodes,)`             | Campo de temperatura estabilizado                 |
| `times`           | `(N_snapshots,)`         | Instantes de tiempo guardados                     |
| `T`               | `(N_snapshots, N_nodes)` | Temperatura en cada nodo por snapshot             |
| `z_levels`        | `(n_z,)`                 | Niveles z en la pared del pozo                    |
| `T_surface`       | `(N_snapshots, n_z)`     | Temperatura angular-promediada T(t, z) en el pozo |

#### Barrido de t_off (`ScanResult`)

El metodo `scan_toff()` retorna un `ScanResult` con:

| Clave         | Forma                       | Descripcion                                       |
|---------------|-----------------------------|----------------------------------------------------|
| `t_off_array` | `(n_toff,)`                 | Valores de t_off barridos                          |
| `times`       | `(n_times,)`                | Instantes de tiempo                                |
| `z_levels`    | `(n_z,)`                    | Niveles z en la pared del pozo                     |
| `T_borehole`  | `(n_toff, n_times, n_z)`    | Superficie T(t, z) por cada t_off                  |

Ambos se guardan en formato `.npz` comprimido.

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

dhe = DHE_simulation(
    'perfiles_5_capas.csv',
    R_min=1.0, R_max=300.0, z_min=100, z_max=400,
    h=500, rho_f=972, c_f=4195, v_f=0.5, T_in=293,
    nr=12,
)

res = dhe.solve(
    dt=20000, t_save=100000,
    t_on=500000, t_off=50000000,
    tf=1000000000,
    stab_tol=1e-9, stab_dt=100000,
)

# Borehole surface T(t, z)
print(res.T_surface.shape)

# Save
res.save('result.npz')
```

### Barrido de t_off

```python
import numpy as np

t_off_values = np.linspace(1e6, 1e8, 20)
scan = dhe.scan_toff(
    dt=20000, t_save=100000,
    t_on=500000, t_off_array=t_off_values,
    tf=1000000000,
    stab_tol=1e-9, stab_dt=100000,
)

# scan.T_borehole has shape (20, n_times, n_z)
scan.save('scan_result.npz')
```

---

### Referencias

1. G. Culver & J. W. Lund, *Downhole Heat Exchangers*, 1999.
2. Y. Zhang et al., *Applied Thermal Engineering* **163** (2019).
3. P. Chi et al., *Energies* **18** (2025).
4. D. R. Poirier & G. H. Geiger, *Transport Phenomena in Materials Processing*, Springer (2016).
5. Y. Zhao et al., *Sustainability* **12** (2020).

</div>
