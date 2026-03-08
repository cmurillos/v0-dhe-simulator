"""
dhe_simulator
=============

Thermal simulation of Downhole Heat Exchangers (DHE) using FEM
on cylindrical meshes with experimentally-derived physical fields.

Usage
-----
    from dhe_simulator import DHE_simulation

    sim = DHE_simulation(
        csv="physical_properties.csv",
        rad_borehole=0.1, rad_simulation=100,
        time_on=600000, time_off=1000000, time_final=2000000,
        dt=20000, time_save=80000, tries=5,
    )
    times, T_layers = sim.run()
    # T_layers[n] has shape (tries, n_times) for layer n
"""

from .dhe_simulation import DHE_simulation

__all__ = ["DHE_simulation"]
__version__ = "0.1.0"
