# setup.py
# Builds the pybind11 C++ extension module.
#
# Usage (from project root, with venv active):
#   python setup.py build_ext --inplace
#   — or from the menu: ./pathplanning → Build Python module
#
# Keep cpp_sources aligned with bindings/pybind_module.cpp includes.

from setuptools import setup, Extension
import pybind11

cpp_sources = [
    "bindings/pybind_module.cpp",
    "core/Position.cpp",
    "core/Types.cpp",
    "environment/Cell.cpp",
    "environment/Environment.cpp",
    "environment/DynamicEnvironment.cpp",
    "environment/SensorModel.cpp",
    "planning/algorithms/AStar.cpp",
    "planning/algorithms/Dijkstra.cpp",
    "planning/algorithms/BFS.cpp",
    "planning/algorithms/BidirectionalAStar.cpp",
    "planning/algorithms/ThetaStar.cpp",
    "planning/algorithms/JPS.cpp",
    "planning/algorithms/DStarLite.cpp",
    "planning/algorithms/RRT.cpp",
    "planning/algorithms/CBS.cpp",
    "planning/hybrid/HeuristicNetwork.cpp",
    "planning/hybrid/NeuralAStar.cpp",
    "rl/RLAgent.cpp",
    "rl/RLEnvironment.cpp",
    "rl/QTable.cpp",
    "rl/QLearningAgent.cpp",
    "rl/DynaQAgent.cpp",
    "utils/ProbabilityUtils.cpp",
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
