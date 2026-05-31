"""Shared JAX primitives for the vectorized engines.

Small, exactly-shared pieces of the sensory geometry that jax_engine, forage, and
matching had each written inline: the distance->intensity falloff and the
divide-by-zero-safe unit vector toward a source. Extracted so the formula lives in
one place. Both broadcast over any leading batch shape (the trailing axis of
``diff`` is the 2-D spatial vector; ``dist`` is its norm with that axis removed).
"""

from __future__ import annotations

import jax.numpy as jnp


def exp_falloff(dist: jnp.ndarray, sensor_range: float) -> jnp.ndarray:
    """Stimulus intensity decaying with distance: exp(-dist / sensor_range)."""
    return jnp.exp(-dist / sensor_range)


def safe_unit(diff: jnp.ndarray, dist: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    """Unit vector diff/||diff||, zero where the distance is ~0 (avoids 0/0)."""
    return jnp.where(dist[..., None] > eps, diff / jnp.clip(dist[..., None], eps, None), 0.0)
