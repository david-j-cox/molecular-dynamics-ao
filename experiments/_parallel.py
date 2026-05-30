"""Parallel sweep helper for experiments (re-exported from the package).

The implementation lives in ``behavioral_md.parallel`` so scripts and the
package can share it; experiments import it from here for convenience.
"""

from behavioral_md.parallel import run_sweep

__all__ = ["run_sweep"]
