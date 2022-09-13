.. _changelog:

Changelog
***************************************
This file documents the bigger changes between versions

[0.3.2] - upcoming
^^^^^^^^^^^^^^^^^^^^^^

- add `recalculate-probabilities` option to birth-death action for permanent usage
- rename `correction-variant` to `approximation-variant` in code and allow this in the input
- deprecate the `correction-variant` option (but currently still supported)
- fixed periodic boundary conditions
- fix crash in DeltaF and ParticleDistribution if no filename was specified
- FES is no longer data member of Histogram class but stored as separate grid in the FesAction
- helper functions required only by single action moved to the respective modules
- a lot of added / changed code documentation
- improved / extended tests (BirthDeathAction)
- cosmetic improvements of code


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
