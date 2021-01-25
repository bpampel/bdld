.. _potential:

Potential [potential]
*********************

Specify the potential the particles should "feel".

Depending on the chosen `type` there are different options available / compulsory, which are explained in the following sections.

Besides specific hardcoded potentials, there are options to enter custom potentials.

**type**: *string*
  Specify the type of the potential. Currently available values

  * :ref:`polynomial`: specify polynomial coefficients for potential
  * :ref:`mueller-brown`: 2D potential with 3 metastable states seperate by barriers.


.. _polynomial:
Polynomial potential
^^^^^^^^^^^^^^^^^^^^

Custom potential build by a polynomial in up to 3 dimensions.


.. _mueller-brown:
Müller-Brown potential
^^^^^^^^^^^^^^^^^^^^^^

More information about this potential can be found in [#]_.


References
^^^^^^^^^^


.. [#] Klaus Müller and Leo D. Brown. Location of saddle points and minimum energy paths by a constrained simplex optimization procedure. Theoretica Chimica Acta, 53(1), 1979.
