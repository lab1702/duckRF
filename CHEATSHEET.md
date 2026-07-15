# duckRF cheat sheet

One-page reference. `tbl`/`model`/`outcome`/`*_col` are **string** names; grids
are `INTEGER[]`. Full docs: [GUIDE.md](GUIDE.md). Two families,
`rf_class_*` (categorical outcome) and `rf_reg_*` (numeric outcome), share every
argument.

```sql
.read rf_macros.sql          -- load everything
```

## Typical workflow

```sql
CREATE TABLE m AS SELECT * FROM rf_class_fit('train', 'y');   -- fit (one row per tree node)
SELECT * FROM rf_class_predict('m', 'newdata');               -- score   (+ pred, probs)
SELECT * FROM rf_class_oob('m', 'train', 'y');                -- honest metrics, no holdout
SELECT * FROM rf_class_evaluate('m', 'test', 'y');            -- metrics on a holdout (1 row)
SELECT * FROM rf_importance('m');                             -- MDI importance per feature
SELECT * FROM rf_summary('m');                                -- how it was fit / what grew
```

`rf_reg_*` is identical on a numeric outcome (`rf_reg_predict` adds `prediction`).

## Fit

```
rf_class_fit(tbl, outcome
             , n_trees := 100, mtry := NULL, max_depth := 20
             , min_samples_split := 2, min_samples_leaf := 1
             , min_impurity_decrease := 0.0
             , sample_frac := 1.0, replace_sample := true
             , criterion := 'gini'      -- 'gini' | 'entropy' (bits)
             , seed := 42
             , weights_col := NULL       -- per-row sample weights
             , class_weight := NULL       -- NULL | 'balanced'
             , tree_from := NULL, tree_to := NULL   -- batch fitting (see below)
             , splitter := 'best')        -- 'best' = RF | 'random' = Extra Trees
rf_reg_fit(tbl, outcome, ... , criterion := 'mse', weights_col := NULL,
           tree_from := NULL, tree_to := NULL, splitter := 'best')
   -> the model table (one row per tree node; see schema below)
```

`mtry := NULL` → `floor(sqrt(d))` for classification, `d` for regression
(sklearn's `sqrt` / `1.0`). `max_depth := NULL` grows to purity (hard cap 60);
the default `20` **does** bind above a few thousand rows — `rf_summary` reports
`depth_cap_hit`. `VARCHAR`/`ENUM` features are true categoricals (subset splits).

**Extra Trees.** `splitter := 'random'` draws ONE random split per candidate
feature (uniform numeric threshold in the node's range; categorical coin-flip
subset) instead of searching for the best — lower variance, faster fits, same
model shape (predict/importance/oob/quantile/batch all still work). Textbook ET:
`splitter := 'random', replace_sample := false, sample_frac := 1.0` ≈ sklearn
`ExtraTrees*(bootstrap=False)`. Default `'best'` is the unchanged Random Forest.

## Batch fitting  (bounded memory; UNION reproduces a one-shot fit)

Fitting is one recursive CTE over all trees, so peak memory ≈ `n_trees × max_depth`
(~1–1.5 GB/tree at depth 20 on 50k × 20). Grow the forest in memory-bounded
batches and `UNION` them. `n_trees` stays the **total** (metadata / RNG / OOB);
`tree_from`/`tree_to` grow only the trees with global index in `[tree_from,
tree_to]` (a lone `NULL` → `1` / `n_trees`; require `1 ≤ from ≤ to ≤ n_trees`).

```sql
CREATE TABLE m AS
      SELECT * FROM rf_reg_fit('big','y', n_trees:=300, max_depth:=18, tree_from:=1,   tree_to:=150)
  UNION ALL
      SELECT * FROM rf_reg_fit('big','y', n_trees:=300, max_depth:=18, tree_from:=151, tree_to:=300);
```

`rf_batched_fit_sql` writes that UNION for you, returned **as text** (fetch & run,
like `dummy_encode_sql`; `query()` won't take it inline):

```
rf_batched_fit_sql(tbl, outcome, family        -- 'classification' | 'regression'
                   , n_trees := 100, batch_size := 10, mtry := NULL, max_depth := 20
                   , min_samples_split := 2, min_samples_leaf := 1, min_impurity_decrease := 0.0
                   , sample_frac := 1.0, replace_sample := true, criterion := NULL
                   , seed := 42, weights_col := NULL, class_weight := NULL
                   , splitter := 'best')  -> VARCHAR
```

```python
sql = con.sql("SELECT rf_batched_fit_sql('big','y','regression', n_trees:=300, batch_size:=50)").fetchone()[0]
con.sql(f"CREATE TABLE m AS {sql}")
```

Bit-identical to the one-shot fit for **integer-slot** fits (classification, no
fractional weights) under `threads=1`. Fractional-slot fits (regression, a
fractional `weights_col`, `class_weight:='balanced'` on imbalanced classes) are
**not** bit-identical: shallow-depth structure matches (`DOUBLE` cols to ~1e-9),
but at the default `max_depth:=20` near-tied splits flip and the forests diverge
structurally — predictions differ ~0.05–0.15, a *statistically equivalent* (not
identical) forest. See [GUIDE.md](GUIDE.md).

## Predict  (na_action ∈ 'null' | 'skip_tree'; n_trees := k uses the first k trees)

```
rf_class_predict(model, tbl, na_action := 'null', n_trees := NULL)
   -> input cols + pred VARCHAR + probs MAP(VARCHAR, DOUBLE)   -- soft voting
rf_reg_predict(model, tbl, na_action := 'null', n_trees := NULL)
   -> input cols + prediction DOUBLE
rf_class_predict_trees(model, tbl, ...)  -> __rf_rid__, tree, pred, probs
rf_reg_predict_trees(model, tbl, ...)    -> __rf_rid__, tree, prediction
```

`pred` is the argmax of the soft vote (ties → smallest label). `*_predict_trees`
gives one row per (row, tree) — the ensemble spread of the *mean* estimate.

## Prediction intervals  (Quantile Regression Forest — honest intervals for a new obs)

```
rf_reg_quantile(model, tbl, outcome, quantiles,
                newdata := NULL, na_action := 'null', n_trees := NULL)
   -> newdata cols + quantile_pred MAP(DOUBLE, DOUBLE)   -- {level -> value}
```

`tbl` = reference sample (the training table); its `outcome` responses, weighted by
shared-leaf membership, form the conditional distribution of Y|x. `quantiles` is a
`DOUBLE[]` of levels in (0,1). `newdata := NULL` scores `tbl`. Pull a level with
`quantile_pred[0.05]`. Unlike `*_predict_trees` (spread of the ensemble *mean*,
under-covers), the `[0.05,0.95]` band covers ≈90% of real observations.

## Evaluate  (returns one row; drops rows with NULL outcome or no prediction)

```
rf_class_evaluate(model, tbl, outcome, na_action := 'null', n_trees := NULL)
   -> n, accuracy, log_loss, brier, auc      -- auc: binary only, else NULL
rf_reg_evaluate(model, tbl, outcome, na_action := 'null', n_trees := NULL)
   -> n, rmse, mae, r2
```

`auc` positive class = lexicographically **greater** label; `log_loss` clipped
to `[1e-15, 1-1e-15]`.

## Out-of-bag  (tbl MUST be the exact training table — validated, errors on mismatch)

```
rf_class_oob_predict(model, tbl) -> input cols + pred + probs   -- NULL if in-bag everywhere
rf_reg_oob_predict(model, tbl)   -> input cols + prediction
rf_class_oob(model, tbl, outcome) -> n, accuracy, log_loss, brier, auc, n_excluded
rf_reg_oob(model, tbl, outcome)   -> n, rmse, mae, r2, n_excluded
```

Each row scored only by the trees that did **not** bag it. `n_excluded` = rows
in-bag for every tree.

## Importance & summary

```
rf_importance(model) -> feature, importance    -- MDI, sums to 1, every feature listed
rf_permutation_importance(model, tbl, outcome, n_repeats := 5, seed := 42)
                     -> feature, importance, importance_std   -- score drop when column shuffled
                        (R^2 for regression / accuracy for classification; sklearn-matched,
                         cardinality-UNBIASED; ordered importance DESC, feature; can be negative)
rf_summary(model)    -> family, n_trees, n_nodes, n_leaves, max_depth_reached,
                        mean_leaf_depth, depth_cap_hit, n_features, mtry, max_depth,
                        criterion, splitter, seed, n_train, sample_frac, replace_sample
```

MDI is biased toward high-cardinality features — prefer `rf_permutation_importance`
(honest, cardinality-unbiased; score it on a holdout table) for a ranking that
survives scrutiny.

## Cross-validation  (min cv_error; folds = (row# − 1) % k)

```
rf_cv(tbl, outcome, family, mtry_grid, k := 5, n_trees := 100, max_depth := 20,
      min_samples_leaf := 1, sample_frac := 1.0, seed := 42)   -> mtry, cv_error
rf_cv_depth(tbl, outcome, family, depth_grid, mtry := NULL, k := 5, n_trees := 100,
            min_samples_leaf := 1, sample_frac := 1.0, seed := 42) -> max_depth, cv_error
```

`family` ∈ `'classification'` | `'regression'`; grids are `INTEGER[]`. `cv_error`
= misclassification rate (classification) or MSE (regression). CV is
**best-split (RF) only** — no `splitter` param; tune Extra Trees by fitting
`splitter := 'random'` directly.

## Model table  (one row per tree node; every `*_fit` returns this shape)

```
tree INT, node BIGINT (root=1, children 2v/2v+1), depth INT (root=0), is_leaf BOOL,
split_feature VARCHAR, split_kind VARCHAR ('num'|'cat'|NULL),
threshold DOUBLE (numeric: LEFT iff value <= threshold),
cats_left VARCHAR[], cats_right VARCHAR[], unseen_left BOOL (categorical routing),
n_rows BIGINT, w_node DOUBLE, impurity DOUBLE (gini / entropy-bits / variance),
imp_decrease DOUBLE (÷ w_root for sklearn's "improvement"),
prediction DOUBLE (regression leaf), class_counts MAP(VARCHAR,DOUBLE) (dense class leaf),
+ constant forest metadata on every row: family, n_trees, seed, sample_frac,
replace_sample, n_train, mtry, max_depth, min_samples_split, min_samples_leaf,
min_impurity_decrease, criterion, splitter ('best'|'random'), features VARCHAR[],
feature_kinds VARCHAR[], classes VARCHAR[] (NULL for regression), train_hash HUGEINT.
```

## Reproducibility

Deterministic in `seed`. For a **bit-identical** model across runs and fresh
connections, fit under `PRAGMA threads=1` (parallel float sums are
non-associative; predictions still agree to ~1e-14 regardless). Bootstrap
membership, the mtry lottery, and OOB are bit-stable at any thread count.
