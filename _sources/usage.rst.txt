.. _usage:

Usage
*****

After installing, the package can be executed and passed an input file::

  python -m bdld input

The input file uses the syntax of the configparser package.
Documentation of the structure and options can be found under :ref:`input`.


If you didn't install the code, you can use the ``bdld_run`` executable in the ``bin`` subdirectory
::

  bdld_run input

which behaves exactly the same way. If you installed the code globally, this executable should now also be in your ``$PATH``.
