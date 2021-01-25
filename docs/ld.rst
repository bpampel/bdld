.. _ld:

Langevin Dynamics with Bussi-Parinello Thermostat [ld]
******************************************************

This sets the option for running Langevin Dynamics with the thermostat described by Bussi and Parinello [#]_.

The respective Langevin equation is

.. math::
  \mathop{}\!\mathrm{d} p(t) = - \nabla U(q) \mathop{}\!\mathrm{d} t - \gamma p(t) \mathop{}\!\mathrm{d} t + \sqrt{2 m \gamma \beta^{-1} } \mathop{}\!\mathrm{d} W (t)



**timestep**: *float*
  integration time step in dimensionless units

**kt**: *float*
  thermal energy of the simulation in energy units (:math:`k_B T = \beta^{-1}`)

**friction**: *float*
  friction parameter :math:`\gamma`

**n_steps**: *int*
  number of time steps the simulation should run

**seed**: *int*, optional
  starting seed for the random number generator of the thermostat


References
^^^^^^^^^^


.. [#] Giovanni Bussi and Michele Parrinello. Accurate sampling using Langevin dynamics. Physical Review E, 75(5):056707, May 2007.
