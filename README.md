# duckRF — random forests (classification & regression) in pure DuckDB SQL

Table macros for DuckDB **1.5+**, no extensions and no driver required. duckRF
grows a random forest — **classification or regression** — entirely inside
DuckDB, and gives you **fit**, **predict**, **evaluate**, **out-of-bag**
scoring, **feature importance**, per-tree predictions and **cross-validation**
for tuning, plus **batch fitting** to grow large forests in a bounded memory
envelope. A `splitter := 'random'` switch turns the forest into **Extra Trees**
(extremely randomized trees) — random per-node splits, lower variance, faster
fits — with everything else (predict, importance, OOB, quantiles, batch)
unchanged. Both families take the same arguments and handle **numeric *and*
categorical features and outcomes** natively: `VARCHAR`/`ENUM` columns are true
categoricals (subset splits, not one-hot), and the classification outcome can be
a boolean, an integer, or a string label.

Everything runs as one recursive CTE: the whole forest is grown breadth-first,
all trees at once, with no `random()`, no `hash()`, no UDFs — all randomness is
`md5_number`, a specified algorithm that is stable across DuckDB versions. It is
the sibling project to [duckLM](https://github.com/lab1702/duckLM) and follows
the same conventions: table and column names are passed as **strings**, every
macro is a `CREATE OR REPLACE MACRO`, and internal helpers are prefixed `__rf_`.

## Setup

```sql
.read rf_macros.sql
```

That's it — the whole library is one file of `CREATE OR REPLACE MACRO`
statements. Load it once per session (or `.read` it from your own script), then
call the macros. It works from the DuckDB CLI and from any driver (Python, R,
Node, …) — pass table and column names as **strings**.

## Documentation

- **[CHEATSHEET.md](CHEATSHEET.md)** — one-page signature reference for every
  macro group (fit / predict / evaluate / out-of-bag / importance / summary /
  cross-validation), the model-table schema, and the typical workflow. Start
  here to look something up fast.
- **[GUIDE.md](GUIDE.md)** — the user's guide: task-oriented explanations,
  runnable examples, the categorical-split story, and the full contract /
  edge-case behavior.

Quick taste:

```sql
CREATE TABLE forest AS SELECT * FROM rf_class_fit('penguins', 'species');
SELECT * FROM rf_class_oob('forest', 'penguins', 'species');   -- honest accuracy, no holdout
SELECT * FROM rf_importance('forest');                          -- which features matter (MDI)
SELECT * FROM rf_permutation_importance('forest', 'penguins', 'species');  -- honest, cardinality-unbiased
```

```
-- rf_class_oob('forest', 'penguins', 'species'):
-- ┌─────┬──────────┬──────────┬──────────┬──────┬────────────┐
-- │  n  │ accuracy │ log_loss │  brier   │ auc  │ n_excluded │
-- │ 300 │     0.99 │  0.0373  │  0.0152  │ NULL │          0 │
-- └─────┴──────────┴──────────┴──────────┴──────┴────────────┘
```

Regression is the same shape — `rf_reg_fit` / `rf_reg_predict` /
`rf_reg_evaluate` / `rf_reg_oob` — on a numeric outcome.

## Testing

Two independent paths (details in [tests/README.md](tests/README.md)):

```bash
# Python suite: every fit/predict/evaluate/importance checked against
# scikit-learn on fixed-seed data (single-tree CART equivalence, forest
# metrics, MDI importance). Run under a single thread for bit-reproducibility.
python -m venv .venv && .venv/bin/python -m pip install -r tests/requirements.txt
.venv/bin/python -m pytest tests/ -q

# Pure-SQL smoke test: no Python, just the DuckDB CLI
duckdb < tests/smoke.sql
```

## Files

- [rf_macros.sql](rf_macros.sql) — the entire library: both `*_fit` families
  (with `tree_from`/`tree_to` batch fitting) and `rf_batched_fit_sql`,
  `*_predict` / `*_predict_trees` / `*_evaluate`, `rf_reg_quantile`
  (quantile-regression-forest prediction intervals), the out-of-bag macros,
  `rf_importance`, `rf_permutation_importance`, `rf_summary`, `rf_cv` /
  `rf_cv_depth`, and the shared split-search / scoring core
- [CHEATSHEET.md](CHEATSHEET.md) — one-page signature reference
- [GUIDE.md](GUIDE.md) — the user's guide
- [tests/](tests) — pytest suite (vs scikit-learn) and a pure-SQL smoke test
- [LICENSE](LICENSE) — MIT

## License

MIT — see [LICENSE](LICENSE).
