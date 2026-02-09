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
# Option A - upload manually:
#   from google.colab import files
#   uploaded = files.upload()
#   csv_path = list(uploaded.keys())[0]
#
# Option B - Google Drive:
#   from google.colab import drive
#   drive.mount('/content/drive')
#   csv_path = '/content/drive/MyDrive/perfiles_5_capas.csv'

csv_path = '/content/perfiles_5_capas.csv'   # <-- adjust as needed

# ---- Cell 4: Run simulation ---------------------------------
# The solver first stabilizes the initial temperature field
# (insulated boundary, until steady state), then runs the
# DHE simulation starting from that stable condition.

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
    # stab_dt=20000,   # optional: stabilization time step (defaults to dt)
    # stab_tol=1e-6,   # optional: stabilization tolerance
)

print(res)

# ---- Cell 5: Save results -----------------------------------
res.save('simulation_output.npz')
print("Saved to simulation_output.npz")

# ---- Cell 6: Load and inspect (can be done later) -----------
loaded = DHEResult.load('simulation_output.npz')
print(f"Nodes:      {loaded.nodes.shape}")
print(f"Elements:   {loaded.elements.shape}")
print(f"Faces:      {loaded.boundary_faces.shape}")
print(f"T0_stable:  {loaded.T0_stable.shape}")
print(f"Times:      {loaded.times.shape}  -> {loaded.times}")
print(f"T matrix:   {loaded.T.shape}")

# ---- Cell 7: delta(t) in a sub-cylinder ---------------------
import matplotlib.pyplot as plt

# delta(t) = volume-weighted mean of |T0_stable(x) - T(t,x)|
# inside the sub-cylinder defined by r <= r_max and z_low <= z <= z_high.
#
# z_low, z_high are ABSOLUTE z-coordinates (from z=0).
# Example on a mesh with z_min=150, z_max=200:
#   Upper half  -> z_low=175, z_high=200
#   Lower half  -> z_low=150, z_high=175
#   Full height -> z_low=150, z_high=200

r_sub  = 50.0    # max radius of sub-cylinder
z_lo   = 175.0   # lower z-bound  (upper half of the reservoir)
z_hi   = 200.0   # upper z-bound

times, delta = res.delta(r_max=r_sub, z_low=z_lo, z_high=z_hi)

print("times:", times)
print("delta:", delta)

plt.figure()
plt.plot(times, delta, '-o')
plt.xlabel('Time [s]')
plt.ylabel(r'$\delta(t)$ [K]')
plt.title(f'Mean |T0_stable - T(t)|  (r<={r_sub}, {z_lo}<=z<={z_hi})')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Cell 8: Backward-compatible dict access -----------------
# res['t'] and res['T'] still work as before:
print(f"Snapshots via dict: {len(res['t'])}")
