"""
dhe_simulator
=============

Thermal simulation of Downhole Heat Exchangers (DHE) using FEM
on cylindrical meshes with experimentally-derived physical fields.

Public API
----------
DHE_simulation : Main simulation class.

Usage
-----
    from dhe_simulator import DHE_simulation

    sim = DHE_simulation(
        csv="physical_properties.csv",
        R_min=0.1, R_max=100, z_min=150, z_max=1900,
        h=500, rho_f=972, c_f=4195, v_f=0.5, T_in=293,
    )
    results = sim.solve(dt=20000, t_save=80000, t_on=600000, t_off=1000000, tf=2000000)
"""

from .dhe_simulation import DHE_simulation, DHEResult, ScanResult

__all__ = ["DHE_simulation", "DHEResult", "ScanResult"]
__version__ = "0.1.0"
