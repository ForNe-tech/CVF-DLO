from setuptools import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize("CVF/process/skeletonize/skeletonize.pyx"),
    include_dirs = [np.get_include()]
)