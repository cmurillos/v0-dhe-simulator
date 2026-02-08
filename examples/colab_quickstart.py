# =============================================================
# DHE Simulator - Google Colab Quick Start
# =============================================================
# Copy this entire cell into a Colab notebook to run it.
#
# Prerequisites:
#   - A CSV file with the 5-layer experimental profiles
#     (upload it to Colab or mount Google Drive).
# =============================================================

# --- 1. Install the library directly from the GitHub repo ----
# !pip install git+https://github.com/cmurillos/v0-dhe-simulator.git@thermal-simulation-library

# --- 2. Import -----------------------------------------------
import numpy as np
from dhe_simulator import DHE_simulation

# --- 3. Upload or specify CSV path ----------------------------
# If your CSV is in Google Drive:
#   from google.colab import drive
#   drive.mount('/content/drive')
#   csv_path = '/content/drive/MyDrive/perfiles_5_capas.csv'
#
# If you upload manually to Colab:
#   from google.colab import files
#   uploaded = files.upload()           # select your CSV
#   csv_path = list(uploaded.keys())[0]

csv_path = '/content/perfiles_5_capas.csv'   # <-- adjust as needed

# --- 4. Define geometry (metres) ------------------------------
R_min = 1.0
R_max = 100.0
z_min = 150
z_max = 200

# --- 5. Create simulation ------------------------------------
dhe = DHE_simulation(csv_path, R_min, R_max, z_min, z_max)

# --- 6. Define coolant temperature (constant 300 K here) ------
T_c = lambda t, x, y, z: np.full_like(x, 300.0)

# --- 7. Run ---------------------------------------------------
res = dhe.solve(
    dt=20000,
    t_save=80000,
    t_on=600000,
    t_off=1000000,
    tf=2000000,
    T_c=T_c,
)

# --- 8. Inspect results ---------------------------------------
print(f"Saved {len(res['t'])} snapshots")
print(f"Time steps : {res['t']}")
print(f"Final T min: {res['T'][-1].min():.2f}  max: {res['T'][-1].max():.2f}")
