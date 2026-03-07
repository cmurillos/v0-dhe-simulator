# =============================================================
# DHE Simulator - Google Colab Quick Start
# =============================================================
# Copy each section below into separate Colab cells.
# =============================================================

# ---- Cell 1: Install ----------------------------------------
# !pip install git+https://github.com/cmurillos/v0-dhe-simulator.git@thermal-simulation-library

# ---- Cell 2: Import -----------------------------------------
import numpy as np
from dhe_simulator import DHE_simulation, DHEResult

# ---- Cell 3: Upload CSV -------------------------------------
# from google.colab import files
# uploaded = files.upload()
# csv_path = list(uploaded.keys())[0]
csv_path = '/content/perfiles_5_capas.csv'

# ---- Cell 4: Run simulation ---------------------------------
R_min = 1.0
R_max = 100.0
z_min = 150
z_max = 200

dhe = DHE_simulation(csv_path, R_min, R_max, z_min, z_max)
T_c = lambda t, x, y, z: np.full_like(x, 300.0)

res = dhe.solve(
    dt=20000,
    t_save=80000,
    t_on=600000,
    t_off=1000000,
    tf=2000000,
    T_c=T_c,
)

print(res)

# ---- Cell 5: Save / Load ------------------------------------
res.save('simulation_output.npz')
loaded = DHEResult.load('simulation_output.npz')
print(f"Nodes: {loaded.nodes.shape}  |  Snapshots: {loaded.T.shape}")

# ---- Cell 6: Mean T on the borehole wall --------------------
import matplotlib.pyplot as plt

# T_borehole() computes the area-weighted mean temperature on the
# inner cylindrical surface (r ~ R_min, z >= z_min).  No args needed.
times, T_mean = res.T_borehole()

print("times :", times)
print("T_mean:", T_mean)

plt.figure()
plt.plot(times, T_mean, '-o')
plt.xlabel('Time [s]')
plt.ylabel('Mean T [K]')
plt.title('Mean borehole-wall temperature')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Cell 7: scan_toff  -- sweep t_off with multiple tries --
# Returns a dict with 'times', 'z_levels', 't_off', and 'T' tensor
# T shape: (tries, n_toff, n_times, n_z)

t_off_values = [800000, 1000000, 1200000, 1500000]

result = dhe.scan_toff(
    dt=20000,
    t_save=80000,
    t_on=600000,
    t_off_array=t_off_values,
    tf=2000000,
    T_c=T_c,
    tries=3,  # number of random field realizations
)

print("times shape:", result['times'].shape)
print("z_levels shape:", result['z_levels'].shape)
print("T tensor shape:", result['T'].shape)  # (tries, n_toff, n_times, n_z)

# ---- Cell 8: Save scan results ------------------------------
np.savez_compressed('scan_results.npz', **result)

# ---- Cell 9: Visualize one realization ----------------------
# Plot T(t,z) heatmap for first try, first t_off
T_surface = result['T'][0, 0]  # (n_times, n_z)
times = result['times']
z = result['z_levels']

plt.figure(figsize=(10, 6))
plt.pcolormesh(times, z, T_surface.T, shading='auto', cmap='hot')
plt.colorbar(label='T [K]')
plt.xlabel('Time [s]')
plt.ylabel('z [m]')
plt.title(f'Borehole T(t,z) - try 1, t_off={t_off_values[0]:.0f}')
plt.tight_layout()
plt.show()
