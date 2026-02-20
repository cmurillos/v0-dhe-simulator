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

# ---- Cell 6: delta(t) on the borehole wall ------------------
import matplotlib.pyplot as plt

# delta(t) is computed automatically on the inner cylindrical
# surface (r ~ R_min, z >= z_min).  No arguments needed.
times, delta = res.delta()

print("times:", times)
print("delta:", delta)

plt.figure()
plt.plot(times, delta, '-o')
plt.xlabel('Time [s]')
plt.ylabel(r'$\delta(t)$ [K]')
plt.title('Mean perturbation on the borehole wall')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Cell 7: scan_toff  -- sweep multiple t_off values ------
# Stabilizes only once, then runs one simulation per t_off.
# Returns a matrix: column 0 = times, columns 1..n = delta curves.

t_off_values = [800000, 1000000, 1200000, 1500000]

table = dhe.scan_toff(
    dt=20000,
    t_save=80000,
    t_on=600000,
    t_off_array=t_off_values,
    tf=2000000,
    T_c=T_c,
)

# table[:, 0] = times,  table[:, i+1] = delta for t_off_values[i]
plt.figure()
for i, toff in enumerate(t_off_values):
    plt.plot(table[:, 0], table[:, i + 1], '-o',
             label=f't_off={toff:.0f}')
plt.xlabel('Time [s]')
plt.ylabel(r'$\delta(t)$ [K]')
plt.title('Borehole perturbation for different t_off')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
