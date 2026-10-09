# Sphinx configuration for nba, following gala's docs (sphinx-astropy defaults,
# pydata-sphinx-theme, automodapi API pages). Build with `make html`.
import pathlib
import shutil
import sys

try:
    from sphinx_astropy.conf.v1 import *  # noqa: F401,F403
except ImportError:
    print("ERROR: building the nba documentation requires sphinx-astropy: "
          "pip install -e '.[docs]'")
    sys.exit(1)

import nba

docs_root = pathlib.Path(__file__).parent.resolve()

project = "nba"
author = "Nicolas Garavito-Camargo"
copyright = "2026, Nicolas Garavito-Camargo"
version = release = nba.__version__

highlight_language = "python3"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
templates_path = ["_templates"]
default_role = "obj"

# API pages
numpydoc_show_class_members = False
numpydoc_xref_param_type = True
automodapi_toctreedirnm = "api"
automodsumm_inherited_members = False

intersphinx_mapping.update({  # noqa: F405
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
})

# Notebooks and Markdown pages with myst-nb. The notebooks are run on the
# cluster (they need the simulation snapshots) and shown as they are.
extensions += ["myst_nb"]  # noqa: F405
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath"]

# The tutorials live in tutorials/ at the top of the repository; copy the ones
# shown in the docs next to tutorials.rst at build time (ignored by git).
TUTORIALS = ["Reading_GC21_MWLMC_snapshots.ipynb", "lmc_centering.ipynb"]
(docs_root / "tutorials").mkdir(exist_ok=True)
for name in TUTORIALS:
    shutil.copy(docs_root.parent / "tutorials" / name, docs_root / "tutorials" / name)

# HTML
html_theme = "pydata_sphinx_theme"
html_title = f"nba v{version}"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/jngaravitoc/nba",
    "navbar_align": "left",
    "show_toc_level": 2,
}
html_sidebars = {"install": [], "getting_started": [], "changes": []}
