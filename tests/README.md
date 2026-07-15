# Tests

Two independent test paths.

## Python suite — `test_rf_macros.py`

Checks every fit, predict, evaluate, out-of-bag and tuning macro against an
equivalent **scikit-learn / numpy** reference on the same fixed-seed data, so a
failure means the macros disagree with a trusted implementation (not merely that
a recorded number drifted). The strongest checks are the single-tree CART
equivalences: with `n_trees:=1, sample_frac:=1.0, replace_sample:=false,
mtry:=d` duckRF grows one deterministic CART that matches sklearn's
`DecisionTree{Regressor,Classifier}` (`max_features=None`) to ~1e-9 on tie-free
data — predictions, `predict_proba`, node impurities (gini / entropy-in-bits /
variance) and MDI `rf_importance` all agree exactly. Where equally-good tied
splits exist, sklearn and duckRF may break the tie differently (both valid), so
those configurations assert the weaker adversarial property that duckRF's
realized objective is *no worse* than sklearn's.

Also covered: soft-voting forest accuracy/R² within tolerance of sklearn's
`RandomForest` on a holdout; every `*_evaluate` metric vs the sklearn metric
function (accuracy / log-loss / brier / AUC / RMSE / MAE / R²); categorical
subset splits vs a brute-force optimum (exact for regression and binary
classification, "at least as good as the best singleton" for the K>2 heuristic),
including numeric-looking levels that stay categorical, ENUM columns, and the
headline categorical-outcome / categorical-features use case; out-of-bag
membership re-derived in numpy and OOB score near a true holdout; determinism
under `PRAGMA threads=1` (bit-identical model, different seed differs) and the
md5 bootstrap-draw replay; `weights_col` and `class_weight:='balanced'`;
`rf_cv` / `rf_cv_depth`; and the full NULL / empty / degenerate / reserved-name /
type / guard error contract.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r tests/requirements.txt   # macOS/Linux
.venv/bin/python -m pytest tests/ -q
```

## SQL smoke test — `smoke.sql`

No Python required — just the DuckDB CLI. Fits on deterministic inline data and
aborts (non-zero exit) on the first failed check. Runs under `PRAGMA threads=1`
so it is bit-for-bit reproducible. Run from the repo root:

```bash
duckdb < tests/smoke.sql
```
