.. _potential:

Potential [potential]
*********************

Specify the potential the particles should "feel".

Depending on the chosen `type` there are different options available / compulsory, which are explained in the following sections.

Besides specific hardcoded potentials, there are options to enter custom potentials.

**type**: *string*
  Specify the type of the potential. Currently available values

  * `polynomial`_: specify polynomial coefficients for potential
  * `mueller-brown`_: 2D potential with 3 metastable states seperate by barriers.
  * `entropic-double-well`_: 2D potential with 3 metastable states seperate by barriers.

                InputOption("n_dim", int, False, default=2),
            ]

**boundary-condition**: *string*
  What should happen if particles go outside the specified range. If not specified, nothing happens. Currently available values:

  * *periodic*: move outside particles to other side of potential
  * *reflective*: move outside particles to the boundary and reverse their momentum

.. _polynomial:

Polynomial potential
^^^^^^^^^^^^^^^^^^^^

Custom potential build by a polynomial in up to 3 dimensions.

**n_dim**: *int*
  Number of dimensions of the potential

**min**: *float* or *list of floats*
  Minimum point(s) of the potential (per dimension)

**min**: *float* or *list of floats*
  Maximum point(s) of the potential (per dimension)

In 1D the coefficients can be directly specified with

**coeffs**: *list of floats*
  Coefficients of polynomial in increasing order, starting with constant order

Alternatively (and compulsory in more than one dimension) the path to the file holding the coefficients can be specified:

**coeffs-file**: *string*
  Path to the file holding the coefficients
  The first :code:`n_dim` columns hold the polynomial orders while column N+1 holds the coefficients.
  Remaining columns or lines starting with :code:`#` are ignored.
  The syntax is compatible with coefficient files used by the :code:`ves_md_linearexpansion` tool of plumed.


.. _mueller-brown:

Müller-Brown potential
^^^^^^^^^^^^^^^^^^^^^^

2D potential created by a sum of four exponential terms.

More information about this potential can be found in [1]_.

The potential ranges are hardcoded to :code:`[(-1.5, 1.5), (-0.5, 2.5)]`

**scaling-factor**: *float*, optional
  Scale the potential by the given factor



Entropic double-well potential
^^^^^^^^^^^^^^^^^^^^^^

Symmetric 2D potential with two (equipotential) states separated by an entropic barrier at x=0.

A special version of this was used in [1]_ (see Eq. 30 there)

The potential ranges are hardcoded to :code:`[(-1.5, 1.5), (-1.5, 1.5)]`

**sigma-x**: *float*, optional
  Defines the width of the barrier in x direction, default 0.1
    - sigma_y 
  Scale the potential by the given factor
**sigma-y**: *float*, optional
  Defines the width of the opening of the barrier in y direction
**scaling-factor**: *float*, optional
  Scale the potential by the given factor, default 1.0


Examples
^^^^^^^^

The following input directly sets up a potential of the form :math:`0.2*x - 4*x^2 + x^4`
::

  [potential]
  type: polynomial
  n_dim: 1
  coeffs: 0, 0.2, -4, 0, 1
  min: -2.5
  max: 2.5

When using a potential of higher dimensionality, the coefficients are given in an extra file.
This is the exemplary input for the Wolfe-Quapp potential
::

  [potential]
  type: polynomial
  n_dim: 2
  coeffs-file: wolfe_quapp.coeffs
  min: -2.5, -2.5
  max: 2.5, 2.5

The `wolfe_quapp.coeffs` file specifying the coefficients might look like this:
::

  #! FIELDS idx_dim1 idx_dim2 pot.coeffs index description
  #! SET type LinearBasisSet
  #! SET ndimensions  2
  #! SET ncoeffs_total  25
  #! SET shape_dim1  5
  #! SET shape_dim2  5
         0       0         0.0000000000000000e+00       0  1*1
         1       0         0.3000000000000000e+00       1  s^1*1
         2       0        -2.0000000000000000e+00       2  s^2*1
         4       0         1.0000000000000000e+00       4  s^4*1
         0       1         0.1000000000000000e+00       5  1*s^1
         1       1        +1.0000000000000000e+00       6  s^1*s^1
         0       2        -4.0000000000000000e+00      10  1*s^2
         0       4         1.0000000000000000e+00      20  1*s^4
  #!-------------------

Note that the parser actually ignores all header comments as well as the index and description column.
Using a file with just the first three columns gives the same result.


When using the Müller-Brown or entropic double-well potentials most of the properties are hardcoded, so only few options remain. Here we choose to employ reflective boundary conditions to avoid particles outside the range due to the low scaling factor.
::

  [potential]
  type: mueller-brown
  scaling-factor: 0.1
  boundary-condition: reflective


References
^^^^^^^^^^

.. [1] Klaus Müller and Leo D. Brown. Location of saddle points and minimum energy paths by a constrained simplex optimization procedure. Theoretica Chimica Acta, 53(1), 1979.

.. [2] Eq. 30 of Faradjian & Elber, J. Chem. Phys. 120, 10880 (2004), https://doi.org/10.1063/1.1738640
