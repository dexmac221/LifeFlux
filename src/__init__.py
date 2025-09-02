"""
LifeFlow - Advanced Cellular Automaton with Flow Dynamics

A sophisticated implementation of Conway's Game of Life extended with 
continuous potential fields and fluid-like dye advection.

Modules:
--------
life2 : 2D Life simulator with flow dynamics
life3d : 3D Life simulator with volumetric visualization
"""

__version__ = "1.0.0"
__author__ = "LifeFlow Contributors"
__license__ = "MIT"

# Import main classes for convenient access
try:
    from .lifeflow2d import LifeFlow2DSimulator, RealtimeViewer
    __all__ = ['LifeFlow2DSimulator', 'RealtimeViewer']
except ImportError:
    __all__ = []

try:
    from .lifeflow3d import LifeFlow3DSimulator, LifeFlow3DViewer
    __all__.extend(['LifeFlow3DSimulator', 'LifeFlow3DViewer'])
except ImportError:
    pass

# Package metadata
DESCRIPTION = "Advanced Cellular Automaton with Flow Dynamics - Conway's Life meets fluid simulation"
URL = "https://github.com/yourusername/lifeflow"
