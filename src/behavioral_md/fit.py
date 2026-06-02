"""Fit organism parameters to target matching-law sensitivities (derivative-free).

This is the payoff of the JAX engine: searching organism parameters so the
generalized-matching-law sensitivities hit chosen targets, while keeping those
sensitivities **emergent** -- they fall out of the full stochastic dynamics (sampling
noise + travel/changeover cost + cue learning), never imposed by a chosen functional
form.

Why derivative-free rather than autodiff. The roadmap framed autodiff as the enabler
for fitting, and we built the differentiable Gumbel-softmax surrogate (``matching_diff``)
for exactly that. But reverse-mode gradients through the ~1000-step recurrent rollout
**explode**: per-step sensitivities compound multiplicatively through the fed-back
state (learned weights, armed-probability, position), so the measured gradient of a
sensitivity w.r.t. a parameter is ~1000x too large, sign-unstable across noise seeds,
and useless for optimization (default ``d a_rate/d beta`` came out at -310..+10 across
seeds vs a true finite-difference of ~+0.03). The one place autodiff *would* be clean
is a molar closed-form value->allocation model, but reproducing the matching law in
closed form requires assuming ``B_k proportional to v_k**a`` -- i.e. writing the
sensitivity in as a parameter, which forces the very result the engine is supposed to
let emerge. So we keep the emergent forward model and search it derivative-free.

What we search. The Gumbel forward model is deterministic and smooth in the parameters
under fixed common-random-number keys (the relaxed softmax is differentiable; only its
*gradient through the long rollout* is unusable -- the *function* is well-behaved, as
the per-parameter probe confirms). So Nelder-Mead on ``soft_sensitivities`` is a clean,
robust search. The raw categorical sim, by contrast, is piecewise-constant under fixed
keys (argmax thresholds), which is why we search the relaxed model and then re-plug the
fitted parameters into the true stochastic engine to confirm transfer (exp023).

Free parameters are ``temperature``, ``approach_gain``, ``beta`` (see
``matching_diff.FREE_PARAMS``; ``lr_cue`` is excluded because its effect does not
transfer). The loss is the squared error to the (a_rate, a_amt) targets, so a single
fit moves both sensitivities jointly (the multi-dimension case).
"""

from __future__ import annotations

import jax
import numpy as np
from scipy.optimize import minimize

from behavioral_md.matching import MatchConfig
from behavioral_md.matching_diff import (
    FREE_PARAMS,
    TUNABLE,
    default_params,
    soft_sensitivities,
    soft_sensitivities_all,
)

# Lower bound keeping the positive scale parameters away from 0 during the search.
_MIN_PARAM = 1e-2


def _to_params(x, free) -> dict:
    """Vector (in ``free`` order) -> constrained free-parameter dict."""
    return {k: float(max(x[i], _MIN_PARAM)) for i, k in enumerate(free)}


def fitted_config(params: dict, mcfg: MatchConfig) -> MatchConfig:
    """A MatchConfig with the fitted free parameters substituted in."""
    return mcfg._replace(**{k: float(v) for k, v in params.items()})


def fit(targets, mcfg: MatchConfig | None = None, free=FREE_PARAMS, n_steps: int = 1500,
        n_org: int = 128, key=None, maxiter: int = 120, verbose: bool = False):
    """Derivative-free (Nelder-Mead) fit of the free parameters to target sensitivities.

    ``targets`` is the desired ``(a_rate, a_amt)`` pair (in the surrogate's sensitivity
    units; note the surrogate's reachable range is compressed relative to the
    stochastic engine, so targets should sit within it -- see exp023). ``free`` is the
    tuple of parameters to search (default the three discriminability levers; add
    ``"amount_exponent"`` to decouple a_amt -- see matching_diff.TUNABLE). Returns
    ``(fitted_mcfg, history)`` where ``history`` is a per-evaluation list of dicts
    (eval, loss, a_rate, a_amt, params). The forward model uses common-random-number
    Gumbel/arming noise (fixed ``key``), so the objective is a deterministic smooth
    function of the parameters.
    """
    mcfg = MatchConfig() if mcfg is None else mcfg
    if key is None:
        key = jax.random.key(0)
    t_rate, t_amt = targets

    # Compile the forward sensitivities once; each evaluation just runs it.
    sens = jax.jit(lambda p: soft_sensitivities(p, mcfg, n_steps, n_org, key))

    history = []

    def objective(x):
        params = _to_params(x, free)
        a_rate, a_amt = sens(params)
        a_rate, a_amt = float(a_rate), float(a_amt)
        loss = (a_rate - t_rate) ** 2 + (a_amt - t_amt) ** 2
        history.append({
            "eval": len(history),
            "loss": loss,
            "a_rate": a_rate,
            "a_amt": a_amt,
            "params": params,
        })
        if verbose and len(history) % 10 == 1:
            ps = "  ".join(f"{k}={params[k]:.3f}" for k in free)
            print(f"  eval {len(history):3d}  loss={loss:.4f}  "
                  f"a_rate={a_rate:.3f}  a_amt={a_amt:.3f}  {ps}")
        return loss

    x0 = np.array([default_params(mcfg, free)[k] for k in free], float)
    bounds = [(_MIN_PARAM, None)] * len(free)
    minimize(objective, x0, method="Nelder-Mead", bounds=bounds,
             options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5})

    best = min(history, key=lambda h: h["loss"])
    if verbose:
        print(f"best loss={best['loss']:.4f} at "
              + "  ".join(f"{k}={best['params'][k]:.3f}" for k in free))
    return fitted_config(best["params"], mcfg), history


def fit_dims(targets: dict, mcfg: MatchConfig | None = None, free=TUNABLE,
             n_steps: int = 1500, n_org: int = 128, key=None, maxiter: int = 160,
             verbose: bool = False):
    """Derivative-free fit targeting any subset of the four sensitivities.

    ``targets`` is a dict over {"rate","amt","prob","delay"} giving the desired value
    of each targeted sensitivity (surrogate units). ``free`` is the tuple of parameters
    to search -- typically the relevant per-dimension curvature levers plus ``beta``
    (the rate anchor); e.g. fitting prob/delay uses
    ``("beta", "probability_exponent", "delay_k")``. Uses the four-dimension surrogate
    (soft_sensitivities_all). Returns ``(fitted_mcfg, history)`` where each history
    entry carries all four sensitivities and the params. See exp025.
    """
    mcfg = MatchConfig() if mcfg is None else mcfg
    if key is None:
        key = jax.random.key(0)
    dims = list(targets)

    sens = jax.jit(lambda p: soft_sensitivities_all(p, mcfg, n_steps, n_org, key))

    history = []

    def objective(x):
        params = _to_params(x, free)
        d = {k: float(v) for k, v in sens(params).items()}
        loss = sum((d[k] - targets[k]) ** 2 for k in dims)
        history.append({"eval": len(history), "loss": loss, **d, "params": params})
        if verbose and len(history) % 10 == 1:
            ds = "  ".join(f"{k}={d[k]:.3f}" for k in dims)
            print(f"  eval {len(history):3d}  loss={loss:.4f}  {ds}")
        return loss

    x0 = np.array([default_params(mcfg, free)[k] for k in free], float)
    bounds = [(_MIN_PARAM, None)] * len(free)
    minimize(objective, x0, method="Nelder-Mead", bounds=bounds,
             options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5})

    best = min(history, key=lambda h: h["loss"])
    return fitted_config(best["params"], mcfg), history


if __name__ == "__main__":
    import sys
    tr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.47
    ta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.60
    print(f"fitting to targets a_rate={tr}, a_amt={ta}")
    fitted, hist = fit((tr, ta), verbose=True)
    print("fitted config:", {k: round(getattr(fitted, k), 4) for k in FREE_PARAMS})
    print(f"loss {hist[0]['loss']:.4f} -> {min(h['loss'] for h in hist):.4f} "
          f"({len(hist)} evals)")
