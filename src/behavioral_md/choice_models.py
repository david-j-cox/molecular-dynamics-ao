"""Competing models of risky choice, and a harness to compare them on skew-controlled gambles.

Roadmap 2.2. The novel claim of the risk arc is that a survival objective is sensitive to the SKEW
of a gamble (not just its mean and variance), and -- uniquely -- that the skew preference REVERSES
with the energy budget (the requirement R). This module sets the survival account against the
standard rivals on gambles whose mean and variance are matched and only the skew varies, so any
difference in predicted choice is a pure skew effect. The models, each scoring a gamble (a list of
``(p, x)`` outcomes) and choosing by a logistic on the value difference:

* ``ev``  -- expected value (risk-neutral baseline).
* ``mv``  -- mean-variance: ``V = mean - b*variance``. Depends ONLY on the first two moments, so it
  is structurally SKEW-BLIND: two gambles matched in mean and variance get the same value (choice
  0.5) whatever their skew. This is the clean "second-moment rival"; Scalar Utility Theory (Kacelnik
  & Bateson 1996), whose risk sensitivity is scalar/Weber noise on magnitude, shares this blindness
  for matched-moment gambles.
* ``eu``  -- expected utility of a cubic utility ``u(x) = x - a*x^2 + c*x^3``: ``u'' = -2a`` is the
  variance aversion and ``u''' = 6c`` the skew preference (prudence; Eeckhoudt & Schlesinger 2006),
  so variance and skew preferences are SEPARATE free parameters. This is the family Genest, Stauffer
  & Schultz (2016) fit to monkeys (a curved utility that predicts variance- and skewness-risk).
* ``pt``  -- prospect theory: ``v(x) = x^alpha`` with rank-dependent probability weighting
  ``w(p) = p^g / (p^g + (1-p)^g)^(1/g)`` (Tversky-Kahneman). Skew-sensitive via the overweighting of
  extreme outcomes, but its skew preference is curvature-fixed -- energy-budget INVARIANT.
* ``survival`` -- the survival-DP advantage ``q_A - q_B`` at a given reserve (see ``survival.py``).
  Skew-sensitive AND energy-budget DEPENDENT: the skew preference reverses across the requirement R
  (the distinguishing prediction; see exp055/moment_dominance and exp058 panel B).

The harness fits a model's parameters to observed choice proportions by maximum likelihood and
compares models by AIC and cross-validated log-likelihood (Wilson & Collins 2019). The headline
result is structural and needs no data (mv cannot express skew); the empirical contact (exp058) fits
the models to the Genest 2016 monkey proportions, where the skew-blind rival fails on the skew
conditions while the skew-sensitive models fit. The decisive energy-budget reversal that separates
``survival`` from ``eu``/``pt`` needs choices at two energy budgets (the matched-moment, skew x
budget design of Caraco & Chasin 1984 -- currently unavailable -- or the exp-2.3 pre-registration).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from behavioral_md.survival import survival_dp


def moments(gamble) -> tuple[float, float, float]:
    """``(mean, variance, standardized skew)`` of a gamble given as ``[(p, x), ...]``."""
    p = np.array([q for q, _ in gamble], float)
    x = np.array([v for _, v in gamble], float)
    m = float((p * x).sum())
    var = float((p * (x - m) ** 2).sum())
    if var <= 0:
        return m, 0.0, 0.0
    skew = float((p * (x - m) ** 3).sum() / var ** 1.5)
    return m, var, skew


# --- model value functions (higher = more preferred) -------------------------------------------

def ev_value(gamble, **_) -> float:
    """Risk-neutral: the expected value."""
    return moments(gamble)[0]


def mv_value(gamble, b: float = 1.0, **_) -> float:
    """Mean-variance: ``mean - b*variance``. Skew-blind by construction."""
    m, var, _ = moments(gamble)
    return m - b * var


def eu_value(gamble, a: float = 0.3, c: float = 0.0, **_) -> float:
    """Expected utility of a cubic utility ``u(x) = x - a*x^2 + c*x^3``.

    ``a`` (u'' = -2a) sets variance aversion; ``c`` (u''' = 6c) sets skew preference (prudence).
    """
    p = np.array([q for q, _ in gamble], float)
    x = np.array([v for _, v in gamble], float)
    return float((p * (x - a * x ** 2 + c * x ** 3)).sum())


def _rank_weight(cum: np.ndarray, gamma: float) -> np.ndarray:
    cum = np.clip(cum, 0.0, 1.0)
    return cum ** gamma / (cum ** gamma + (1.0 - cum) ** gamma) ** (1.0 / gamma)


def pt_value(gamble, alpha: float = 0.88, gamma: float = 0.61, **_) -> float:
    """Prospect theory: ``v(x) = x^alpha`` (x >= 0) with rank-dependent probability weighting."""
    p = np.array([q for q, _ in gamble], float)
    x = np.array([v for _, v in gamble], float)
    order = np.argsort(x)[::-1]                       # best outcome first
    p, x = p[order], x[order]
    cw = _rank_weight(np.cumsum(p), gamma)
    dw = np.diff(np.concatenate([[0.0], cw]))         # decision weights
    return float((dw * np.clip(x, 0.0, None) ** alpha).sum())


STATIC_MODELS = {"ev": ev_value, "mv": mv_value, "eu": eu_value, "pt": pt_value}
# Free parameters searched per static model (besides the choice temperature ``beta``).
MODEL_PARAMS = {"ev": [], "mv": ["b"], "eu": ["a", "c"], "pt": ["alpha", "gamma"]}
_PARAM_INIT = {"b": 1.0, "a": 0.3, "c": 0.0, "alpha": 0.88, "gamma": 0.7, "beta": 8.0}
_PARAM_BOUNDS = {"b": (0.0, 50.0), "a": (-5.0, 5.0), "c": (-20.0, 20.0),
                 "alpha": (0.2, 1.5), "gamma": (0.3, 1.2), "beta": (0.05, 200.0)}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def predict_choice(model, A, B, beta: float = 8.0, **params) -> float:
    """P(choose A over B) = logistic(beta * (value(A) - value(B)))."""
    return float(_sigmoid(beta * (model(A, **params) - model(B, **params))))


def survival_advantage(A, B, reserve: float, day_steps: int = 14, night_steps: int = 16,
                       metabolism: float = 0.03, tstep: int | None = None,
                       n_egrid: int = 401) -> float:
    """Survival-value advantage ``q_A - q_B`` of gamble A over B at a given reserve.

    Computed from the survival DP (A as 'safe' slot, B as 'risky' slot); ``tstep`` selects the
    day-step (default = a smooth midday slice). The sign reverses across the requirement R for a
    skew-pair -- the energy-budget-dependent skew preference no fixed-utility model produces.
    """
    res = survival_dp(A, B, day_steps, night_steps, metabolism, n_egrid=n_egrid)
    t = day_steps // 2 if tstep is None else tstep
    e = res["energy"]
    adv = res["q_safe"][t] - res["q_risky"][t]
    return float(np.interp(np.clip(reserve, e[0], e[-1]), e, adv))


# --- fitting and comparison --------------------------------------------------------------------

def _neg_loglik(model, params_free, free_names, trials) -> float:
    """trials: list of dicts {A, B, k (# chose A), n (total)}."""
    params = {n: v for n, v in zip(free_names, params_free, strict=True)}
    beta = params.pop("beta")
    ll = 0.0
    for tr in trials:
        p = predict_choice(model, tr["A"], tr["B"], beta=beta, **params)
        p = min(max(p, 1e-9), 1 - 1e-9)
        ll += tr["k"] * np.log(p) + (tr["n"] - tr["k"]) * np.log(1 - p)
    return -ll


def fit_model(name: str, trials, restarts: int = 4, seed: int = 0) -> dict:
    """Maximum-likelihood fit of a static model to choice trials; returns params, negloglik, AIC.

    ``trials`` is a list of ``{A, B, k, n}`` (gambles A, B; ``k`` of ``n`` choices were of A). The
    free parameters are the model's own plus the choice temperature ``beta``; AIC counts both.
    """
    model = STATIC_MODELS[name]
    free_names = MODEL_PARAMS[name] + ["beta"]
    bounds = [_PARAM_BOUNDS[p] for p in free_names]
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        x0 = ([_PARAM_INIT[p] for p in free_names] if r == 0
              else [rng.uniform(lo, min(hi, lo + 5)) for lo, hi in bounds])
        res = minimize(lambda x: _neg_loglik(model, x, free_names, trials), x0,
                       method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    k = len(free_names)
    return {"name": name, "params": dict(zip(free_names, best.x, strict=True)),
            "negloglik": float(best.fun), "aic": float(2 * k + 2 * best.fun), "n_params": k}


def compare_models(trials, names=("ev", "mv", "eu", "pt"), **kw) -> list[dict]:
    """Fit each model and return fits sorted by AIC (best first), with delta-AIC from the winner."""
    fits = [fit_model(n, trials, **kw) for n in names]
    fits.sort(key=lambda f: f["aic"])
    best = fits[0]["aic"]
    for f in fits:
        f["delta_aic"] = f["aic"] - best
    return fits


def crossval_loglik(name: str, trials, k_folds: int = 5, seed: int = 0, **kw) -> float:
    """Mean held-out per-trial log-likelihood under k-fold cross-validation (higher = better)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trials))
    folds = np.array_split(idx, k_folds)
    model = STATIC_MODELS[name]
    total_ll, total_n = 0.0, 0
    for f in range(k_folds):
        test = set(folds[f].tolist())
        train = [trials[i] for i in range(len(trials)) if i not in test]
        fit = fit_model(name, train, **kw)
        beta = fit["params"]["beta"]
        ps = {n: v for n, v in fit["params"].items() if n != "beta"}
        for i in folds[f]:
            tr = trials[i]
            p = min(max(predict_choice(model, tr["A"], tr["B"], beta=beta, **ps), 1e-9), 1 - 1e-9)
            total_ll += tr["k"] * np.log(p) + (tr["n"] - tr["k"]) * np.log(1 - p)
            total_n += tr["n"]
    return float(total_ll / total_n)
