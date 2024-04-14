# Sources:
# https://itnext.io/daily-bit-e-of-c-modern-documentation-tools-9b96ba283732
# https://stackoverflow.com/questions/59990484/doxygensphinxbreatheexhale

from textwrap3 import dedent

# Basic configuration
project = 'sharded_map'
copyright = '2024, Skadic'
author = 'Skadic'

# Extensions to use
extensions = [ "breathe", "exhale", 'myst_parser' ]

# Configuration for the breathe extension
# Which directory to read the Doxygen output from
breathe_projects = {"sharded_map":"xml"}
breathe_default_project = "sharded_map"

exhale_args = {
    "containmentFolder": "./api",
    "doxygenStripFromPath": "../include",
    #"doxygenStripFromPath": "../src",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "sharded_map API",
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    "treeViewIsBootstrap": True,
}

# Configuration for the theme
html_theme = "furo"
html_theme_options = {
    "repository_url": "https://github.com/Skadic/sharded_map",
    "use_repository_button": True,
}
