# =============================================================
# DHE Simulator - Google Colab Quick Start
# =============================================================
# Copy each section below into separate Colab cells.
# =============================================================

# ---- Cell 1: Install ----------------------------------------
# !pip install git+https://github.com/cmurillos/v0-dhe-simulator.git@thermal-simulation-library

# ---- Cell 2: Generate the 10-layer CSV ----------------------
import pandas as pd
import numpy as np

alpha_diff = 1e-6  # thermal diffusivity [m^2/s]

Z  = np.array([100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900], dtype=float)
T  = np.array([291, 297, 303, 309, 315, 321, 327, 333, 339, 345], dtype=float)
vT = np.array([1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.2, 3.5, 4.0], dtype=float)
c  = np.array([850, 830, 800, 780, 760, 740, 720, 700, 700, 680], dtype=float)
vc = np.array([40, 35, 30, 25, 25, 30, 35, 40, 45, 50], dtype=float)
k  = np.array([2.2, 2.25, 2.29, 2.34, 2.38, 2.41, 2.43, 2.45, 2.50, 2.50], dtype=float)
vk = np.array([0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.23, 0.25, 0.25, 0.25], dtype=float)
p  = k / (alpha_diff * c)
vp = np.array([120, 130, 140, 150, 160, 170, 180, 190, 195, 200], dtype=float)

df = pd.DataFrame({'Z': Z, 'T': T, 'vT': vT,
                    'p': p, 'vp': vp,
                    'c': c, 'vc': vc,
                    'k': k, 'vk': vk})
csv_path = '/content/perfiles_10_capas.csv'
df.to_csv(csv_path, index=False)
print(df)

# ---- Cell 3: Import -----------------------------------------
from dhe_simulator import DHE_simulation, DHEResult, ScanResult

# ---- Cell 4: Define geometry and fluid parameters ------------
R_min  = 0.1       # borehole radius [m]
R_max  = 100.0     # outer domain radius [m]
z_min  = 150.0     # top of sealed base [m]
z_max  = 1900.0    # borehole depth [m]

h     = 500.0      # convective coefficient [W/(m^2 K)]
rho_f = 972.0      # water density at ~80 C [kg/m^3]
c_f   = 4195.0     # water specific heat [J/(kg K)]
v_f   = 0.5        # axial velocity [m/s]
T_in  = 293.0      # inlet temperature [K] (~20 C)

# ---- Cell 5: Build simulation object ------------------------
dhe = DHE_simulation(
    csv_path, R_min, R_max, z_min, z_max,
    h=h, rho_f=rho_f, c_f=c_f, v_f=v_f, T_in=T_in,
)

# ---- Cell 6: Single run -------------------------------------
res = dhe.solve(
    dt=20000,
    t_save=80000,
    t_on=600000,
    t_off=1000000,
    tf=2000000,
)
print(res)

# ---- Cell 7: Heatmap of T_borehole(t, z) for a single run ---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.pcolormesh(
    res.times / 1e6,          # x-axis: time in M-seconds
    res.z_levels,              # y-axis: depth z
    res.T_surface.T,           # transpose so shape = (n_z, n_times)
    shading='auto', cmap='inferno',
)
ax.set_xlabel('Time [Ms]')
ax.set_ylabel('Depth z [m]')
ax.set_title('Borehole wall temperature T(t, z)')
fig.colorbar(im, ax=ax, label='T [K]')
plt.tight_layout()
plt.show()

# ---- Cell 8: Save / Load single run -------------------------
res.save('simulation_output.npz')
loaded = DHEResult.load('simulation_output.npz')
print(f"Loaded: {loaded}")

# ---- Cell 9: scan_toff  -- sweep multiple t_off values ------
t_off_values = [800000, 1000000, 1200000, 1500000]

scan = dhe.scan_toff(
    dt=20000,
    t_save=80000,
    t_on=600000,
    t_off_array=t_off_values,
    tf=2000000,
)
print(scan)

# ---- Cell 10: Save / Load scan result -----------------------
scan.save('scan_output.npz')
scan_loaded = ScanResult.load('scan_output.npz')
print(f"Loaded: {scan_loaded}")

# ---- Cell 11: Heatmaps -- one subplot per t_off -------------
fig, axes = plt.subplots(1, len(t_off_values), figsize=(5*len(t_off_values), 5),
                         sharey=True)
if len(t_off_values) == 1:
    axes = [axes]

vmin = scan.T_borehole.min()
vmax = scan.T_borehole.max()

for i, (ax, toff) in enumerate(zip(axes, t_off_values)):
    im = ax.pcolormesh(
        scan.times / 1e6,
        scan.z_levels,
        scan.T_borehole[i].T,   # (n_z, n_times)
        shading='auto', cmap='inferno',
        vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel('Time [Ms]')
    ax.set_title(f't_off = {toff/1e6:.1f} Ms')
    if i == 0:
        ax.set_ylabel('Depth z [m]')

fig.colorbar(im, ax=axes, label='T [K]', shrink=0.8)
fig.suptitle('Borehole wall T(t, z) for different t_off', fontsize=14)
plt.tight_layout()
plt.show()

# ---- Cell 12: T(z) profiles at a fixed time -----------------
t_snapshot = 1200000   # pick a time [s]
idx_t = np.argmin(np.abs(scan.times - t_snapshot))

fig, ax = plt.subplots(figsize=(6, 6))
for i, toff in enumerate(t_off_values):
    ax.plot(scan.T_borehole[i, idx_t, :], scan.z_levels,
            '-o', label=f't_off={toff/1e6:.1f} Ms')
ax.set_xlabel('T [K]')
ax.set_ylabel('Depth z [m]')
ax.set_title(f'Borehole T(z) at t = {scan.times[idx_t]/1e6:.2f} Ms')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
