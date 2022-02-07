.. _histogram:

Histogram the particle positions [histogram]
********************************************

Action to store a histogram of the particle positions

This requires that the trajectories are at least saved in memory, i.e. the :ref:`[trajectories] <trajectories>` section has to exist in the input file.
The number of time steps between updates of the histogram is set by the number of positions stored in memory by the :ref:`[trajectories] <trajectories>` action.

Note that when printing out the histogram the midpoint of the bins is given as coordinate

**min**: *float* or *list of floats*
  Minimum position of the histogram per dimension

**max**: *float* or *list of floats*
  Maximum position of the histogram per dimension

**bins**: *int* or *list of ints*
  Number of histogram bins per direction

**stride**: *int*, defaults to 1
  Time steps between saved positions of particles

**reset**: *int* or *list of ints*
  Reset all histogram counts at the specified time steps.
  Will be done after writing the current histogram, but before any dependent actions are executed

**filename**: *string*, optional
  Filename to write the histogram to.

**write-stride**: *int*, optional
  Write histogram to file every n time steps (filename will be appended by :code:`_t` (with the current time step t).
  If not specified the histogram will only be written once at the end of the simulation.

**fmt**: *string*, defaults to "%14.9"
  c-style format specifier used when writing data to file


Example
^^^^^^^

::

[histogram]
bins: 200
min: -2.5
max: 2.5
filename: histo
