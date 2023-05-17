# Configuration file for the Sphinx documentation builder.
import os
import sys

# === Project Config ====

import logging
logging.getLogger('sphinx').setLevel(logging.DEBUG)


# Point ReadTheDocs to the directory
sys.path.insert(0, os.path.abspath('../../'))
                
project = 'SpuCo'
copyright = '2023, Siddharth Joshi, Yu Yang, Baharan Mirzasoleiman'
author = 'Siddharth Joshi, Yu Yang, Baharan Mirzasoleiman'

# === General Config ===

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_rtd_theme",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

autodoc_mock_imports = [
    "xarray",
]

# How to represents typehints
autodoc_typehints = "signature"

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

html_theme_options = {"collapse_navigation": False}

# -- Options for EPUB output
epub_show_urls = "footnote"

# Make sure __init__ is added to docstrings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}