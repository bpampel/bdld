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

**aproximation-variant** or **correction-variant**: *string*, optional
  Specify the approximation (Lambda) to use. Defaults to *original*
  **correction-variant** is currently still accepted to support old input files but is deprecated and will be removed in a later version.
  If both are specified, no error is thrown and the **approximation-variant** value is used.
  Possible values are:

  * *original* or *orig*: the original approximation by Lu et al. Default value
  * *additive* or *add*
  * *multiplicative* or *mult*

**exponential-factor**: *float*, optional
  factor in the exponential of the birth-death probabilities, also referred to as *rate factor*

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
