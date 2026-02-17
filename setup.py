# from pybind11 import get_cmake_dir
# Available at setup time due to pyproject.toml
# from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

# ext_modules = [
#     Pybind11Extension("python_example",
#         ["src/main.cpp"],
#         # Example: passing in the version to the compiled code
#         define_macros = [('VERSION_INFO', __version__)],
#         ),
# ]

setup(
    name="Warehouse_RL",
    version=__version__,
    author="",
    author_email="",
    url="",
    description="",
    long_description="",
    packages=[
        "Warehouse_RL",
    ],
)
