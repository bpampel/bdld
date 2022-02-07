.. _ld:

Langevin Dynamics [ld]
******************************************************

This section sets the option for the Langevin Dynamics.


Integrators
^^^^^^^^^^^
Currently two different integrators are implemented, which are specified with the **type** option:

**type**: *string*
  the integrator to use

The available options are *"bussi-parinello"* and *"overdamped"*.

The Bussi-Parinello option uses the algorithm and the thermostat described by Bussi and Parinello [#1]_.

The respective Langevin equation is

.. math::
  \mathop{}\!\mathrm{d} p(t) = - \nabla U(q) \mathop{}\!\mathrm{d} t - \gamma p(t) \mathop{}\!\mathrm{d} t + \sqrt{2 m \gamma \beta^{-1} } \mathop{}\!\mathrm{d} W (t)


The update rule for the overdamped integrator is a Euler-Maruyama scheme [#2]_ given by the simpler equation that was e.g. used in [#3]_:

.. math::
  x(t+\Delta t) = x(t) - \Delta t \nabla U(x(t)) + \sqrt{2 \Delta t} W (t+\Delta t)


Options
^^^^^^^

**timestep**: *float*
  integration time step in dimensionless units

**n_steps**: *int*
  number of time steps the simulation should run

**seed**: *int*, optional
  starting seed for the random number generator of the noise term

**kt**: *float*, required only for bussi-parinello
  thermal energy of the simulation in energy units (:math:`k_B T = \beta^{-1}`)

**friction**: *float*, required only for bussi-parinello
  friction parameter :math:`\gamma`

Example
^^^^^^^

::

  [ld]
  type: bussi-parinello
  timestep: 0.005
  kt: 1.0
  friction: 10.0
  n_steps: 10000


References
^^^^^^^^^^

.. [#1] Giovanni Bussi and Michele Parrinello. Accurate sampling using Langevin dynamics. Physical Review E, 75(5):056707, May 2007. doi: 10.1103/PhysRevE.75.056707
.. [#2] Gisiro Maruyama. Continuous Markov Processes and Stochastic Equations. Rendiconti del Circolo Matematico di Palermo 4(1):48–90. 1955. doi: 10.1007/10.1007/BF02846028
.. [#3] Yulong Lu, Jianfeng Lu, and James Nolen. Accelerated Langevin Sampling with Birth-Death. arXiv:1905.09863v1
