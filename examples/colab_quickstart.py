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
# DHEResult(nodes=(N,3), elements=(M,4), boundary_faces=(F,3), snapshots=25)

# ---- Cell 5: Save results -----------------------------------
res.save('simulation_output.npz')
print("Saved to simulation_output.npz")

# ---- Cell 6: Load and inspect (can be done later) -----------
loaded = DHEResult.load('simulation_output.npz')
print(f"Nodes:     {loaded.nodes.shape}")
print(f"Elements:  {loaded.elements.shape}")
print(f"Faces:     {loaded.boundary_faces.shape}")
print(f"Times:     {loaded.times.shape}  -> {loaded.times}")
print(f"T matrix:  {loaded.T.shape}")
print(f"T final:   min={loaded.T[-1].min():.2f}  max={loaded.T[-1].max():.2f}")

# ---- Cell 7: Backward-compatible dict access -----------------
# res['t'] and res['T'] still work as before:
print(f"Snapshots via dict: {len(res['t'])}")
