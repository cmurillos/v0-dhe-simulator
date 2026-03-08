# =============================================================
# DHE Simulator - Google Colab Quick Start
# =============================================================
# Copy each section below into separate Colab cells.
# =============================================================

# ---- Cell 1: Install ----------------------------------------
# !pip install git+https://github.com/cmurillos/v0-dhe-simulator.git@thermal-simulation-library

# ---- Cell 2: Import -----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from dhe_simulator import DHE_simulation

# ---- Cell 3: Upload CSV -------------------------------------
# from google.colab import files
# uploaded = files.upload()
# csv_path = list(uploaded.keys())[0]
csv_path = '/content/physical_properties.csv'

# ---- Cell 4: Run simulation ---------------------------------
sim = DHE_simulation(
    csv=csv_path,
    rad_borehole=1.0,
    rad_simulation=100.0,
    time_on=600000,
    time_off=1000000,
    time_final=2000000,
    dt=20000,
    time_save=80000,
    tries=3,
    nr=10,
    nz=20,
    ntheta=15,
)

times, T_layers = sim.run()

print(f"times shape: {times.shape}")
print(f"Number of layers: {len(T_layers)}")
print(f"Each layer shape: {T_layers[0].shape}")  # (tries, n_times)

# ---- Cell 5: Save results -----------------------------------
np.savez_compressed(
    'simulation_results.npz',
    times=times,
    T_layers=np.array(T_layers),  # (n_layers, tries, n_times)
    z_levels=sim._bh_z_levels,
)

# ---- Cell 6: Visualize one layer ----------------------------
layer_idx = 5  # choose a layer
z_val = sim._bh_z_levels[layer_idx]

plt.figure(figsize=(10, 5))
for tr in range(sim.tries):
    plt.plot(times, T_layers[layer_idx][tr], label=f'Try {tr+1}')
plt.xlabel('Time [s]')
plt.ylabel('T [K]')
plt.title(f'Borehole T(t) at layer {layer_idx} (z={z_val:.1f} m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Cell 7: Heatmap of mean across tries -------------------
# Average T across all tries for each layer
T_mean = np.array([T_layers[n].mean(axis=0) for n in range(len(T_layers))])
# T_mean shape: (n_layers, n_times)

plt.figure(figsize=(10, 6))
plt.pcolormesh(times, sim._bh_z_levels, T_mean, shading='auto', cmap='hot')
plt.colorbar(label='T [K]')
plt.xlabel('Time [s]')
plt.ylabel('z [m]')
plt.title('Mean borehole T(t,z) averaged over tries')
plt.tight_layout()
plt.show()
