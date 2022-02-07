.. _install:

Installation
************

Clone the git repository or download either the ``bdld-$version.tar.gz`` or ``bdld-$version.whl`` archive to a directory of your liking.

It is not actually required to install the code.
If you want to run the code without installing, you just need to put the source code somewhere (by e.g. extracting the tar.gz file) and can proceed with `usage`_.
You then need to manually make sure that all the dependencies are installed.

If you want to install it, there are different possibilities:

1. use pip or wheel to install it globally
2. use the Makefile to install it in a virtual environment

Install globally with pip/wheel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have pip installed you can directly install the package to your global python library from the archive file.
Using wheel is the newer (and faster) way, but requires additionally the `wheel` package.

Both (`.tar.gz` and `.whl`) can be installed with::

  pip install $archive

Install in local environment with Makefile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use the Makefile to install it to a virtual environment with all the required packages.
Just running::

  make install

in the root directory of the code should be enough to install it to a virtual environment under ``.venv``.

It can be activated by typing ``source .venv/bin/activated`` after which you should be able to use the bdld package.

More details about venv can be found in the `official python docs <https://docs.python.org/3/library/venv.html>`_.
