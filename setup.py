from setuptools import setup

setup(
    name="bdld",
    version="0.2.0",
    description="Simulation algorithm for Langevin dynamics with additional birth/death steps",
    author="Benjamin Pampel",
    packages=["bdld", "bdld.actions", "bdld.potential", "bdld.helpers"],
    scripts=["bin/bdld_run"],
    python_requires=">=3.6",
    install_requires=["numpy>=1.19", "scipy>=1.5", "matplotlib>=3.3"],
    extras_require={"kde-statsmodels": ["statsmodels"]},
)
