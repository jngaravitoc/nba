************
Installation
************

``nba`` needs Python 3.10 or later. Install it from the GitHub repository::

    git clone https://github.com/jngaravitoc/nba.git
    cd nba
    python -m pip install .

Use ``python -m pip install -e .`` instead for an editable install, to work on
the code.

Optional dependencies are installed with extras:

=============  =====================================================================
Extra          Installs
=============  =====================================================================
``extra``      healpy, pynbody and the FIRE tools (gizmo_analysis, halo_analysis)
``dev``        pytest and flake8, to run the tests
``docs``       Sphinx and the extensions used to build this documentation
=============  =====================================================================

For example::

    python -m pip install -e ".[dev,docs]"

Building the documentation
==========================

From the ``docs/`` folder::

    make html

The pages are written to ``docs/_build/html``; open ``index.html`` in a browser.
The examples in the pages are tested with::

    pytest --doctest-rst docs/
