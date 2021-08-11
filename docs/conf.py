# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath("../src/fairlens"))

# for x in os.walk('../../src'):
#     sys.path.insert(0, x[0])

# -- Project information -----------------------------------------------------

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"


project = "FairLens"
copyright = "2021, Synthesized Ltd."
author = "Synthesized Ltd."

# The full version, including alpha/beta/rc tags
release = get_version("fairlens")


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "m2r2",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx_panels",
    "sphinxcontrib.bibtex",
]

autosummary_generate = True
add_module_names = False
autodoc_typehints = "description"

autodoc_default_options = {"exclude-members": "__weakref__,__dict__,__init_subclass__"}

panels_add_bootstrap_css = False

bibtex_bibfiles = ["refs.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/synthesized-io/fairlens",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/synthesizedio",
            "icon": "fab fa-twitter-square",
        },
    ],
    "external_links": [
        {"name": "Synthesized", "url": "https://synthesized.io"},
    ],
    "google_analytics_id": "UA-130210493-1",
    "navbar_start": ["navbar-logo", "version.html"],
}

html_sidebars = {"**": ["search-field.html", "sidebar-nav-bs.html"]}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
]

mathjax_path = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

# Customization
html_logo = "_static/FairLens_196x51.png"
