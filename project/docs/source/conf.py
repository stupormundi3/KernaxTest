
import os
import sys
autodoc_mock_imports = ["jax", "jaxlib", "optax", "chex", "absl", "ml_dtypes"]
racine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, racine_path)

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
    'sphinx_togglebutton',
    'nbsphinx'
]
nbsphinx_execute = 'never'


templates_path = ['_templates']
exclude_patterns = []




html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
