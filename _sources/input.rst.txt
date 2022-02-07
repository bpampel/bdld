.. _input:

Input file
**********

This is an explanation of the syntax of the input file and an overview over the possible actions that can be specified.


Syntax
^^^^^^

The input files are read with the `configparser package <https://docs.python.org/3/library/configparser.html>`_ .
In short, the file is structured into sections, which are marked with square brackets, e.g.::

  [potential]

Each setting for the section is then specified with a line containing::

  option: value

to set the option *option* to *value*.

The configparser format allows to specify options that should be shared among all sections under a `[DEFAULT]` section, such as::

  [DEFAULT]
  kt: 1

which sets the thermal energy to 1 for all sections.


.. warning::
  Unknown options do not result in errors, they are simply ignored.

  Therefore, beware of typos in non-essential sections or options:
  the program will run without error, but might not do what is expected!

For more details of the file syntax see the `configparser documentation <https://docs.python.org/3/library/configparser.html>`_ .


Option types
^^^^^^^^^^^^

The option values have different types.
The desired type for each option is given in the specific section.
These are not strictly checked, but python has to be able to convert the entered value into the desired type.

Possible types are:

* *int*, e.g. `2`
* *float*, e.g. `5.83`
* *string*, e.g. `my_filename`
* *bool*, e.g `true` or `false`

Some options allow or require to enter lists. Simply seperate multiple values by a comma, like::

  list-allowing-option: 3, 5

(whitespace is ignored)


Overview of actions
^^^^^^^^^^^^^^^^^^^

Each action is listed here shortly, for more details visit the specific pages.


:ref:`[ld]<ld>`
  Langevin dynamics options, always required. Specify e.g. timestep, friction, temperature and number of steps here.


:ref:`[potential]<potential>`
  The potential to use for the Langevin Dynamics, always required. Either one of the hardcoded ones, or specify polynomial coefficients.

:ref:`[particles]<particles>`
  Number of independent particles/walkers and the initial conditions, always required.

:ref:`[birth-death]<birth-death>`
  Options for the birth-death events. Can be omitted for pure Langevin Dynamics.


The following actions are for receiving output from the simulation, some of them after performing analysis.
They will not alter the movement of the particles.

Note that while all of them are optional, some are required for other actions, e.g. you need a histogram to calculate the FES.


:ref:`[trajectories]<trajectories>`
  Store trajectories of the particles, optional.

:ref:`[histogram]<histogram>`
  Save positions of walkers in histogram, optional. Requires the [trajectories] action.

:ref:`[fes]<fes>`
  Write estimates of the free energy surface to files, optional. Requires the [histogram] action.

:ref:`[delta-f]<delta-f>`
  Calculate the free energy difference between specified states, optional. Requires the [fes] action.

:ref:`[particle-distribution]<particle-distribution>`
  Count the number of particles in given states, optional.
