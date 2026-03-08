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
    deep_borehole=150.0,       # z_min: where borehole starts
    time_on=600000,
    time_off=1000000,
    time_final=2000000,
    dt=20000,
    time_save=80000,
    T_c=300.0,                 # boundary temperature [K]
    tries=3,
    nr=10,
    nz=20,
    ntheta=15,
)

times, T = sim.run()

print(f"times shape: {times.shape}")           # (n_times,)
print(f"T shape: {T.shape}")                   # (tries, n_times, n_nodes)
print(f"nodes shape: {sim.nodes.shape}")       # (n_nodes, 3)

# ---- Cell 5: Save results -----------------------------------
np.savez_compressed(
    'simulation_results.npz',
    times=times,
    T=T,                       # (tries, n_times, n_nodes)
    nodes=sim.nodes,           # (n_nodes, 3) for reconstruction
    elements=sim.elements,     # mesh connectivity
)

# ---- Cell 6: Reconstruct T at a specific time ---------------
try_idx = 0
time_idx = 10   # snapshot index

T_snapshot = T[try_idx, time_idx, :]  # (n_nodes,)
coords = sim.nodes                     # (n_nodes, 3) -> [x, y, z]

print(f"T at t={times[time_idx]:.0f}s, try {try_idx+1}:")
print(f"  min={T_snapshot.min():.2f} K, max={T_snapshot.max():.2f} K")

# ---- Cell 7: Plot T vs z at borehole (r ~ R_min) ------------
# Select nodes near the borehole wall
r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
bh_mask = np.abs(r - sim.R_min) < 0.5 * sim.R_min
bh_nodes = np.where(bh_mask)[0]

z_bh = coords[bh_nodes, 2]
T_bh = T_snapshot[bh_nodes]

# Sort by z for plotting
order = np.argsort(z_bh)
plt.figure(figsize=(8, 5))
plt.plot(T_bh[order], z_bh[order], 'o-')
plt.xlabel('T [K]')
plt.ylabel('z [m]')
plt.title(f'Borehole T(z) at t={times[time_idx]:.0f}s')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Cell 8: Evolution at one node --------------------------
node_idx = bh_nodes[len(bh_nodes)//2]  # middle borehole node
plt.figure(figsize=(10, 5))
for tr in range(sim.tries):
    plt.plot(times, T[tr, :, node_idx], label=f'Try {tr+1}')
plt.xlabel('Time [s]')
plt.ylabel('T [K]')
plt.title(f'T(t) at node {node_idx} (z={coords[node_idx, 2]:.1f} m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
