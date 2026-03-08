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
        rad_borehole=1.0, rad_simulation=100,
        deep_borehole=150,
        time_on=600000, time_off=1000000, time_final=2000000,
        dt=20000, time_save=80000, T_c=300, tries=5,
    )
    times, T = sim.run()
    # times: (n_times,)
    # T: (tries, n_times, n_nodes) - full temperature field
    # sim.nodes: (n_nodes, 3) - mesh coordinates for reconstruction
"""

from .dhe_simulation import DHE_simulation

__all__ = ["DHE_simulation"]
__version__ = "0.1.0"
