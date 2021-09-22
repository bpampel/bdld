.. _birth-death:

Birth-death algorithm [birth-death]
***************************************

Perform birth-death events during the simulation.

**stride**: *int*
  Number of time steps between attempts of birth and death events

**kernel-bandwidth**: *float* or *list of floats*
  Bandwidth of the Gaussian kernel per dimension

**recalculate-probabilities**: *bool*, optional
  Recalculate the probabilities after each succesful birth-death event

**correction-variant**: *string*, optional
  use a correction to deviate from the original algorithm. Possible values are:

  * *additive*: the first proposed correction
  * *multiplicative*: the second proposed correction

**seed**: *int*, optional
  Starting seed for the random number generator used to accept/decline the birth-death probabilities

**stats-stride**: *int*, optional
  Write statistics about the birth-death events every n time steps.
  Will not write any statistics if not specified

**stats-filename**: *string*, optional
  Write statistics to specified file instead of screen

Example
^^^^^^^

::

  [birth-death]
  stride: 100
  kernel-bandwidth: 0.3
  correction-variant: multiplicative
