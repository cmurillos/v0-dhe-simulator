from setuptools import setup, find_packages

setup(
    name="dhe_simulator",
    version="0.1.0",
    description=(
        "Thermal simulation of Downhole Heat Exchangers (DHE) "
        "using FEM on cylindrical meshes with experimentally-derived "
        "physical fields."
    ),
    author="cmurillos",
    url="https://github.com/cmurillos/v0-dhe-simulator",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22,<2.2",
        "scipy>=1.9,<1.17",
        "pandas>=1.5,<3.0",
        "matplotlib>=3.5",
        "scikit-fem>=8.0",
    ],
)
