from distutils.core import Extension
from Cython.Build import cythonize

# from https://stackoverflow.com/questions/31043774/customize-location-of-so-file-generated-by-cython
ext_modules = [
    Extension("hictools.utils.cnumtools",
              ['hictools/utils/cnumtools.pyx'], )
]


def build(setup_kwargs):
    setup_kwargs.update({
        'ext_modules': cythonize(ext_modules),
    })
