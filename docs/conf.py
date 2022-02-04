# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import sphinx_book_theme


# -- Project information -----------------------------------------------------

project = 'APPFL'
copyright = '2021, Argonne National Laboratory'
author = 'Argonne National Laboratory'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
        'sphinx.ext.viewcode',
        'sphinx.ext.autosummary',
        'sphinx.ext.todo',
        'sphinx.ext.doctest',
        'sphinx.ext.intersphinx',
        'sphinx.ext.coverage',
        'sphinx.ext.mathjax',
        'sphinx.ext.ifconfig',
        'sphinx.ext.autosectionlabel',
        'myst_parser',
        'sphinx_book_theme',
        'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = [
        os.path.join(sphinx_book_theme.get_html_theme_path(), "_templates"),
        '_templates'
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_data']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
html_theme_path = [sphinx_book_theme.get_html_theme_path()]

# html_logo = '_static/logo/appfl.png'

html_theme_options = {
    # header settings
    "repository_url": "https://github.com/APPFL/APPFL",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_download_button": False,
    # sidebar settings
    "show_navbar_depth": 1,
#     "logo_only": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_copy_source = False

nbsphinx_execute = 'never'
