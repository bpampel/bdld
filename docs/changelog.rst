.. _changelog:

Changelog
***************************************
This file documents the bigger changes between versions


[0.3.1] - 2021-07-06
^^^^^^^^^^^^^^^^^^^^^^^

- added option to output momentum to trajectory files
- added optional factor for birth-death probabilities
- tuning to use gitlab CI, cleanup of some old code

[0.3.0] - 2021-03-23
^^^^^^^^^^^^^^^^^^^^^^^

- Birth-death: Density estimate includes diagonal term in Kernel matrix (i=j)
- simple overdamped Langevin integrator (Euler-Maruyama) added
- mass of particles can now be specified


[0.2.0] - 2021-01-26
^^^^^^^^^^^^^^^^^^^^^^^

- complete refactor of code
- input of simulation is now via much more flexible input files instead of fixed scripts
- other major changes like periodic output of different statistics / analysis


[0.1.0] - 2020-08-02
^^^^^^^^^^^^^^^^^^^^^^^

- first working version
