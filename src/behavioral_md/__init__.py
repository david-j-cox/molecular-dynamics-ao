"""Behavioral Molecular Dynamics Simulation Engine.

Models an organism as a distributed dynamical system of coupled behavioral
"atoms" whose activations evolve under history-dependent stimulus forces via
Verlet integration. Behavior emerges from the coupled dynamics rather than from
a reinforcement-learning policy.
"""

from behavioral_md.config import SimulationConfig

__all__ = ["SimulationConfig"]
__version__ = "0.1.0"
