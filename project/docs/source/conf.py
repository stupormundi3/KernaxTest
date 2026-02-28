
import os
import sys
racine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
kernel_path = os.path.join(racine_path, 'Kernelification')
sys.path.insert(0, kernel_path)

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
project = 'Kernax'
copyright = '2026, Ziad Iziz'
author = 'Ziad Iziz'
release = '2026'



extensions = [
    'sphinx.ext.autodoc',   
    'sphinx.ext.napoleon',   
    'sphinx.ext.viewcode',  
]

templates_path = ['_templates']
exclude_patterns = []




html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
