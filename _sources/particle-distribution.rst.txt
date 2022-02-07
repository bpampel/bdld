.. _particle-distribution:

Analyze distribution of particles [particle-distribution]
*****************************************************

Count the number of particles in each of the given states.


**stride**: *int*
  Time steps between calculations.

States are specified as rectangular areas of the coordinates, defined by min and max values.

**state1-min**: *list of floats*
  Minimum coordinates of state 1 (one value per dimension)

**state1-max**: *list of floats*
  Maximum coordinates of state 1 (one value per dimension)

Analoguous for all states 2, 3, 4 etc.

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

  [particle-distribution]
  stride: 100
  filename: particle_dist
  state1-min: 0, 0
  state1-max: 1, 1
  state2-min: 0, 1
  state2-max: 0, 2
