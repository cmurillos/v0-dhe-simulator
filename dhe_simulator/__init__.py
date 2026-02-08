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
        R_min=0.05, R_max=0.10,
        z_min=50, z_max=500,
    )
    results = sim.solve(dt=1.0, t_save=10, t_on=0, t_off=3600, tf=7200, T_c=...)
"""

from .dhe_simulation import DHE_simulation

__all__ = ["DHE_simulation"]
__version__ = "0.1.0"
