# setup.py
# Builds the pybind11 C++ extension module.
#
# Usage (from project root, with venv active):
#   python setup.py build_ext --inplace
#
# This compiles bindings/pybind_module.cpp + all C++ source files into
# a shared library called pathplanning.so (or pathplanning.cpython-312-darwin.so)
# that Python can import directly.

from setuptools import setup, Extension
import pybind11

cpp_sources = [
    "bindings/pybind_module.cpp",
    "core/Position.cpp",
    "core/Types.cpp",
    "environment/Cell.cpp",
    "environment/Environment.cpp",
    "environment/DynamicEnvironment.cpp",
    "planning/algorithms/AStar.cpp",
    "planning/algorithms/Dijkstra.cpp",
    "planning/algorithms/BFS.cpp",
    "planning/algorithms/BidirectionalAStar.cpp",
    "planning/algorithms/ThetaStar.cpp",
    "planning/algorithms/JPS.cpp",
    "planning/algorithms/DStarLite.cpp",
    "planning/algorithms/CBS.cpp",
    "environment/SensorModel.cpp",
    "rl/RLAgent.cpp",
    "rl/RLEnvironment.cpp",
    "rl/QTable.cpp",
    "rl/QLearningAgent.cpp",
    "utils/ProbabilityUtils.cpp",
    "planning/hybrid/HeuristicNetwork.cpp",
    "planning/hybrid/NeuralAStar.cpp",
]

extension = Extension(
    name="pathplanning",
    sources=cpp_sources,
    include_dirs=[
        pybind11.get_include(),
        ".",
    ],
    extra_compile_args=["-std=c++17", "-O2", "-Wall"],
    language="c++",
)

setup(
    name="pathplanning",
    ext_modules=[extension],
)
