from distutils.core import setup
from distutils.extension import Extension
#from Cython.Distutils import build_ext

# Get numpy path
# import os, numpy
# numpy_base_path = os.path.split(numpy.__file__)[0]
# numpy_incl_path = os.path.join(numpy_base_path, "core", "include")

# lm10_acc = Extension("streams.potential._lm10_acceleration",
#                       ["streams/potential/_lm10_acceleration.pyx"],
#                      include_dirs=[numpy_incl_path])

setup(
    name="Nonlinear Dynamics",
    version="0.0",
    author="Adrian M. Price-Whelan",
    author_email="adrn@astro.columbia.edu",
    license="BSD",
    packages=['nonlineardynamics'],
    ext_modules=[]
)
