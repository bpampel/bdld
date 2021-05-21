.. _birth-death:

Birth-death algorithm [birth-death]
***************************************

Perform birth-death events during the simulation.

**stride**: *int*
  Number of time steps between attempts of birth and death events

**kernel-bandwidth**: *float* or *list of floats*
  Bandwidth of the Gaussian kernel per dimension

**correction-variant**: *string*, optional
  use a correction to deviate from the original algorithm. Possible values are:

  * *additive*: the first proposed correction
  * *multiplicative*: the second proposed correction
    p

**equilibrium-density-method**: *string*, optional
  Specify the method to calculate the equilibrium density. Must be one of:

  * *potential*: the true equilibrium density from the potential (default)
  * *histogram*: estimate from sampling via an histogram. Requires specifying the associated histogram with **density-estimate-histogram**
  * *uniform*: a flat distribution

**density-estimate-histogram**: *string*, optional
  Specify histogram action to use for calculating the equilibrium density

**density-estimate-stride**: *int*, optional
  Number of time steps between updates of the equilibrium density
  Will only be used if **density-estimate-histogram** was specified

**seed**: *int*, optional
  Starting seed for the random number generator used to accept/decline the birth-death probabilities
  Internally 1000 is added to this seed value to have different seeds for the LD and birth-death when specifying a seed in DEFAULT

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
