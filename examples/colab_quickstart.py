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

