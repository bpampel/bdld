.. _delta-f:

Free energy difference between states [delta-f]
***********************************************

Calculate the difference between states from the :ref:`[fes] action <fes>`


**stride**: *int*, optional
  Time steps between calculations.
  If not specified, it will only be calculated at the end of the simulation.

States are specified as rectangular areas of the coordinates, defined by min and max values.

**state1-min**: *list of floats*
  Minimum coordinates of state 1 (one value per dimension)

**state1-max**: *list of floats*
  Maximum coordinates of state 1 (one value per dimension)

Analoguous for all states 2, 3, 4 etc.
At least 2 states need to be specified.
All free energy differences are given with respect to the first state.

**filename**: *string*, optional
  Filename to write the values to.

**write-stride**: *int*, defaults to 100*stride
  Write calculated values en block to file every n time steps

**fmt**: *string*, defaults to "%14.9"
  c-style format specifier used when writing data to file


Example
^^^^^^^

Calculate the differences between two states every 100 time steps
::

  [delta-f]
  stride: 100
  filename: delta-f
  state1-min: 0, 0
  state1-max: 1, 1
  state2-min: 0, 1
  state2-max: 0, 2
