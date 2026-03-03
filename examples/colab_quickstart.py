# =============================================================
# DHE Simulator - Google Colab Quick Start
# =============================================================
# Copy each section below into separate Colab cells.
# =============================================================

# ---- Cell 1: Install ----------------------------------------
# !pip install --force-reinstall git+https://github.com/cmurillos/v0-dhe-simulator.git@thermal-simulation-library

# ---- Cell 2: Generate the 5-layer CSV -----------------------
import pandas as pd
import numpy as np

alpha_diff = 1e-6  # thermal diffusivity [m^2/s]

Z  = np.array([100, 200, 300, 400, 500], dtype=float)
T  = np.array([291, 297, 303, 309, 315], dtype=float)
vT = np.array([1.0, 1.2, 1.5, 1.8, 2.0], dtype=float)
c  = np.array([850, 830, 800, 780, 760], dtype=float)
vc = np.array([40, 35, 30, 25, 25], dtype=float)
k  = np.array([2.2, 2.25, 2.29, 2.34, 2.38], dtype=float)
vk = np.array([0.10, 0.12, 0.15, 0.18, 0.20], dtype=float)
p  = k / (alpha_diff * c)
vp = np.array([120, 130, 140, 150, 160], dtype=float)

df = pd.DataFrame({'Z': Z, 'T': T, 'vT': vT,
                    'p': p, 'vp': vp,
                    'c': c, 'vc': vc,
                    'k': k, 'vk': vk})
csv_path = '/content/perfiles_5_capas.csv'
df.to_csv(csv_path, index=False)
print(df)

# ---- Cell 3: Import -----------------------------------------
from dhe_simulator import DHE_simulation, DHEResult, ScanResult

# ---- Cell 4: Define geometry and fluid parameters ------------
R_min  = 1.0       # borehole radius [m]
R_max  = 300.0     # outer domain radius [m]
z_min  = 100.0     # top of the open borehole [m]
z_max  = 400.0     # borehole depth [m]

h     = 500.0      # convective coefficient [W/(m^2 K)]
rho_f = 972.0      # water density at ~80 C [kg/m^3]
c_f   = 4195.0     # water specific heat [J/(kg K)]
v_f   = 0.5        # axial velocity [m/s]
T_in  = 293.0      # inlet temperature [K] (~20 C)

# ---- Cell 5: Build simulation object ------------------------
dhe = DHE_simulation(
    csv_path, R_min, R_max, z_min, z_max,
    h=h, rho_f=rho_f, c_f=c_f, v_f=v_f, T_in=T_in,
    nr=12,
)

# ---- Cell 6: Single run -------------------------------------
res = dhe.solve(
    dt=20000,
    t_save=100000,
    t_on=500000,
    t_off=50000000,
    tf=1000000000,
    stab_tol=1e-9,
    stab_dt=100000,
)
print(res)

# ---- Cell 7: Heatmap T(t, z) for a single run ---------------
import matplotlib.pyplot as plt
from matplotlib import ticker

days = res.times / 86400.0

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.pcolormesh(
    days, res.z_levels, res.T_surface.T,
    shading='auto', cmap='inferno',
)
ax.set_xlabel('Time [days]', fontsize=12)
ax.set_ylabel('Depth z [m]', fontsize=12)
ax.set_title('Borehole wall temperature  T(t, z)', fontsize=14)
cb = fig.colorbar(im, ax=ax, label='T [K]')
plt.tight_layout()
plt.show()

# ---- Cell 8: Save / Load single run -------------------------
res.save('simulation_output.npz')
loaded = DHEResult.load('simulation_output.npz')
print(f"Loaded: {loaded}")

# ---- Cell 9: scan_toff  -- sweep multiple t_off values ------
t_off_values = np.linspace(1e6, 1e8, 20)

scan = dhe.scan_toff(
    dt=20000,
    t_save=100000,
    t_on=500000,
    t_off_array=t_off_values,
    tf=1000000000,
    stab_tol=1e-9,
    stab_dt=100000,
)
print(scan)

# ---- Cell 10: Save / Load scan result -----------------------
scan.save('scan_output.npz')
scan_loaded = ScanResult.load('scan_output.npz')
print(f"Loaded: {scan_loaded}")

# ---- Cell 11: Overview heatmap grid (4 x 5) -----------------
# Pick a subset or show all 20 t_off in a 4x5 grid.
n_toff = len(t_off_values)
ncols = 5
nrows = int(np.ceil(n_toff / ncols))

days_scan = scan.times / 86400.0
toff_days = (t_off_values - t_off_values[0]) / 86400.0

vmin = scan.T_borehole.min()
vmax = scan.T_borehole.max()

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                         sharex=True, sharey=True)
axes_flat = axes.ravel()

for i in range(n_toff):
    ax = axes_flat[i]
    im = ax.pcolormesh(
        days_scan, scan.z_levels, scan.T_borehole[i].T,
        shading='auto', cmap='inferno',
        vmin=vmin, vmax=vmax,
    )
    ax.set_title(f't_off = {t_off_values[i]/86400:.0f} d', fontsize=9)
    if i % ncols == 0:
        ax.set_ylabel('z [m]', fontsize=9)
    if i >= (nrows - 1) * ncols:
        ax.set_xlabel('t [days]', fontsize=9)
    ax.tick_params(labelsize=7)

# hide unused panels
for j in range(n_toff, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.colorbar(im, ax=axes, label='T [K]', shrink=0.6, pad=0.02)
fig.suptitle('Borehole wall T(t, z) for different t_off', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# ---- Cell 12: Mean borehole T(t) per t_off ------------------
# Average T across all z-levels at each time for each t_off.

fig, ax = plt.subplots(figsize=(12, 5))
cmap = plt.cm.viridis(np.linspace(0, 1, n_toff))

for i in range(n_toff):
    T_mean_z = scan.T_borehole[i].mean(axis=1)   # (n_times,)
    ax.plot(days_scan, T_mean_z, color=cmap[i], lw=1.2)

sm = plt.cm.ScalarMappable(
    cmap='viridis',
    norm=plt.Normalize(t_off_values[0]/86400, t_off_values[-1]/86400))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, label='t_off [days]')
ax.set_xlabel('Time [days]', fontsize=12)
ax.set_ylabel('Mean borehole T [K]', fontsize=12)
ax.set_title('Depth-averaged borehole temperature per t_off', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---- Cell 13: T(z) profiles at a fixed time -----------------
t_snapshot_days = 5000    # pick a time in days
t_snapshot_s = t_snapshot_days * 86400.0
idx_t = np.argmin(np.abs(scan.times - t_snapshot_s))

fig, ax = plt.subplots(figsize=(6, 7))
cmap = plt.cm.viridis(np.linspace(0, 1, n_toff))

for i in range(n_toff):
    ax.plot(scan.T_borehole[i, idx_t, :], scan.z_levels,
            color=cmap[i], lw=1.2)

sm = plt.cm.ScalarMappable(
    cmap='viridis',
    norm=plt.Normalize(t_off_values[0]/86400, t_off_values[-1]/86400))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, label='t_off [days]')
ax.set_xlabel('T [K]', fontsize=12)
ax.set_ylabel('Depth z [m]', fontsize=12)
ax.set_title(f'Borehole T(z) at t = {scan.times[idx_t]/86400:.0f} days',
             fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
