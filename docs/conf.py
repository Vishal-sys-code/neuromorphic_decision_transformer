# Configuration file for the Sphinx documentation builder.

import os
import sys
# Insert the source path for autodoc
sys.path.insert(0, os.path.abspath('../snn-dt/src'))

project = 'Spiking Decision Transformer'
copyright = '2025, DeepBrain Labs'
author = 'Vishal Pandey & Debasmita Biswas / DeepBrain Labs'
version = '1.0'
release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google/NumPy style docstrings
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'myst_parser',          # Markdown support
    'sphinx_copybutton',    # Copy buttons for code blocks
    'sphinx_design',        # For badges, cards, tabs
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- MyST Parser Configuration -----------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
]
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_title = "Spiking Decision Transformer"

html_theme_options = {
    "repository_url": "https://github.com/Vishal-sys-code/neuromorphic_decision_transformer",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "home_page_in_toc": True,
    "show_navbar_depth": 3,
    "toc_title": "Research Contents",
    "extra_footer": "Based on <i>Spiking Decision Transformers: Local Plasticity, Phase-Coding, and Dendritic Routing for Low-Power Sequence Control</i> (2025).",
}