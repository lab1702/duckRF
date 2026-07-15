# duckRF user's guide

Task-oriented documentation for the duckRF random-forest macros. For a one-page
signature reference see [CHEATSHEET.md](CHEATSHEET.md); for install/setup see the
[README](README.md).

- [Fitting](#fitting)
- [Predicting](#predicting)
- [Prediction intervals (`rf_reg_quantile`)](#prediction-intervals-rf_reg_quantile)
- [Evaluating](#evaluating)
- [Out-of-bag](#out-of-bag)
- [Feature importance](#feature-importance)
- [Categorical features](#categorical-features)
- [Tuning hyperparameters (cross-validation)](#tuning-hyperparameters)
- [Contract & fine print](#contract--fine-print)

The examples use a small `penguins` table — a numeric `bill_length_mm`,
`bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, a categorical `island`
(`Biscoe` / `Dream` / `Torgersen`), and a categorical `species` (`Adelie` /
`Chinstrap` / `Gentoo`). Classification predicts `species`; regression predicts
`body_mass_g`. Outputs below were produced under `PRAGMA threads=1` (see
[reproducibility](#reproducibility)); long doubles are rounded for readability.

## Fitting

`rf_class_fit` and `rf_reg_fit` grow a forest on `outcome` from **every other
column** of the input table (except an optional weights column). They return the
**model table** — one row per tree node — which you save with `CREATE TABLE`.

```sql
CREATE TABLE cmodel AS SELECT * FROM rf_class_fit('penguins', 'species');
CREATE TABLE rmodel AS SELECT * FROM rf_reg_fit('penguins', 'body_mass_g');
```

Pick the family by the outcome, not by the outcome's storage type:
`rf_reg_fit`'s outcome must be numeric; `rf_class_fit` casts its outcome to
`VARCHAR`, so a boolean, an integer (`0`/`1`), or a string label all fit happily
as classes. (The prefixes differ by three characters on purpose — a one-letter
typo would silently change the model *family* rather than error.)

Both families take the identical argument list:

| argument | default | meaning |
|---|---|---|
| `tbl` | — | table/view name as a **string** (resolved via `query_table`; schema-qualified names work) |
| `outcome` | — | column to predict, as a string |
| `n_trees` | `100` | number of trees in the forest |
| `mtry` | `NULL` | features sampled per node; `NULL` → `floor(sqrt(d))` for classification, `d` (all features) for regression, matching sklearn's `max_features='sqrt'` / `1.0` |
| `max_depth` | `20` | maximum tree depth (root = 0); `NULL` grows to purity, hard cap 60. See [the note below](#max_depth-binds) — the default **does** bind above a few thousand rows |
| `min_samples_split` | `2` | minimum rows in a node to consider a split |
| `min_samples_leaf` | `1` | minimum rows in each child |
| `min_impurity_decrease` | `0.0` | a split is accepted iff `imp_decrease / w_root + eps >= this` (`>=`, exactly as sklearn: zero-gain splits on an impure node **are** made — XOR data needs that) |
| `sample_frac` | `1.0` | bootstrap sample size as a fraction of *n* (`0 < sample_frac <= 1`, sklearn's `max_samples`) |
| `replace_sample` | `true` | sample with replacement (bagging); `false` draws a random subset without replacement |
| `criterion` | `'gini'` / `'mse'` | `'gini'` or `'entropy'` (in **bits**, log base 2, as sklearn's) for classification; `'mse'` for regression |
| `seed` | `42` | all randomness is `md5_number(seed || …)`; must not be `NULL` |
| `weights_col` | `NULL` | column of non-negative per-row sample weights (multiplies the bootstrap count — sklearn's `sample_weight`) |
| `class_weight` | `NULL` | classification only; `'balanced'` multiplies each row's weight by `n / (K · n_k)` |
| `splitter` | `'best'` | `'best'` = Random Forest (search for the best split); `'random'` = **Extra Trees** (one random split per candidate feature). See [below](#extra-trees) |

**The bootstrap (bagging).** Each tree trains on a resample of the training
rows. With `replace_sample := true` (the default) a tree draws
`ceil(sample_frac · n)` rows *with replacement*, so some rows appear several
times and about a third are left out — those out-of-bag rows are what
[`rf_*_oob`](#out-of-bag) scores. `replace_sample := false` takes a random subset
*without* replacement instead. Every draw is `md5_number(seed || ':' || tree ||
':' || k)`, so the membership is fully determined by `seed` and bit-stable
regardless of threading.

**mtry — the feature lottery.** At each node a random `mtry`-sized subset of the
features is considered for the split (drawn only from features that actually vary
in the node, matching sklearn's effective budget). Smaller `mtry` decorrelates
the trees; `NULL` gives the sklearn defaults above. Tune it with
[`rf_cv`](#tuning-hyperparameters).

**Sample weights.** `weights_col` names a column of non-negative per-row weights;
a weight multiplies that row's bootstrap count, so an integer weight behaves like
replicating the row that many times (sklearn's `sample_weight`). Weights apply to
fitting only — `rf_*_predict` and `rf_*_evaluate` don't take them.

```sql
CREATE TABLE m AS SELECT * FROM rf_class_fit('surveys', 'label', weights_col := 'sampling_weight');
```

**Class weights.** `class_weight := 'balanced'` (classification only) multiplies
each row's weight by `n / (K · n_k)` — sklearn's exact formula, with the class
counts taken *unweighted* — so rare classes pull their weight in an imbalanced
problem. Combine it with `weights_col` and the two multiply.

```sql
CREATE TABLE m AS SELECT * FROM rf_class_fit('transactions', 'is_fraud', class_weight := 'balanced');
```

### <a id="extra-trees"></a>Extra Trees (extremely randomized trees)

`splitter := 'random'` grows an **Extra-Trees** forest instead of a Random
Forest. Everything about the forest is the same — the same bootstrap, the same
`mtry` feature lottery per node, the same impurity/gain algebra, the same
pruning — *except how each node's split is chosen*:

- **Random Forest** (`splitter := 'best'`, the default) searches every candidate
  feature for the single **best** split (the exact CART optimum).
- **Extra Trees** (`splitter := 'random'`) draws **one random split per
  candidate feature** and keeps the feature whose random split has the best gain.
  For a numeric feature it draws a single threshold uniformly in the feature's
  in-node range `[min, max)` (`thr = min + u·(max − min)`, `u ∈ [0,1)` from the
  same `md5_number(seed || …)` stream — no `random()`, so it is fully
  deterministic in `seed`). For a categorical feature it flips an independent
  coin per level to build a random left/right subset. A draw that leaves a child
  empty (or below `min_samples_leaf`) is rejected, and a node with no admissible
  random split becomes a leaf, exactly as on the best path.

Randomizing the split trades a little more bias for **lower variance** and
**faster fits** (no split search), and often helps on **noisy** data. Because
only the split *selection* changes, the model table is byte-for-byte the same
shape and **`rf_*_predict`, `rf_*_evaluate`, `rf_importance` (MDI), `rf_*_oob`,
`rf_reg_quantile` and batch fitting all work on an Extra-Trees model with no
changes.** `rf_summary` reports the `splitter`. (The one exception is
[`rf_cv`](#tuning-hyperparameters) / `rf_cv_depth`, which are **best-split
(Random Forest) only** — they have no `splitter` parameter. To tune an
Extra-Trees forest, sweep hyper-parameters by fitting with `splitter := 'random'`
directly.)

`splitter` is **orthogonal to bootstrapping**. sklearn's `ExtraTrees*` default to
`bootstrap=False`; in duckRF that is the separate `replace_sample`/`sample_frac`
knob. **Textbook Extra Trees** is therefore:

```sql
CREATE TABLE m AS SELECT * FROM rf_reg_fit('housing', 'price',
    splitter := 'random', replace_sample := false, sample_frac := 1.0);
```

which matches `ExtraTreesRegressor(bootstrap=False, max_features=1.0)`
(regression) or `ExtraTreesClassifier(bootstrap=False, max_features='sqrt')`
(classification) — duckRF's `mtry` defaults already match those `max_features`.
Keeping bootstrap **on** (the default) gives a bagged Extra-Trees hybrid and, as
a bonus, leaves out-of-bag rows for [`rf_*_oob`](#out-of-bag) to score.

> The categorical random-split rule (a per-level coin flip into a subset) is
> duckRF's own definition — there is no sklearn reference for categorical Extra
> Trees, just as there is none for RF categorical subset splits.

**What the model table looks like.** One row per node; the root of each tree is
`node = 1`, children are `2·node` (left) and `2·node + 1` (right). Internal nodes
carry the split; leaves carry the prediction. You rarely read it by hand — the
scoring macros do — but it is plain SQL and worth a look:

```sql
SELECT tree, node, depth, is_leaf, split_feature, split_kind,
       round(threshold, 1) AS threshold, n_rows, round(impurity, 4) AS impurity
FROM cmodel WHERE tree = 1 AND node = 1;
-- ┌──────┬──────┬───────┬─────────┬───────────────┬────────────┬───────────┬────────┬──────────┐
-- │ tree │ node │ depth │ is_leaf │ split_feature │ split_kind │ threshold │ n_rows │ impurity │
-- │    1 │    1 │     0 │ false   │ body_mass_g   │ num        │    4402.0 │    191 │   0.6654 │
-- └──────┴──────┴───────┴─────────┴───────────────┴────────────┴───────────┴────────┴──────────┘
```

A numeric split routes a row **left iff `value <= threshold`**. A classification
leaf carries a **dense** `class_counts` map (one key per training class, zeros
included) so probability vectors can be averaged across trees key by key:

```sql
SELECT tree, node, class_counts FROM cmodel WHERE is_leaf AND tree = 1 ORDER BY node LIMIT 2;
-- ┌──────┬──────┬───────────────────────────────────────────┐
-- │ tree │ node │               class_counts                │
-- │    1 │    3 │ {Adelie=0.0, Chinstrap=0.0, Gentoo=101.0} │
-- │    1 │    9 │ {Adelie=56.0, Chinstrap=0.0, Gentoo=0.0}  │
-- └──────┴──────┴───────────────────────────────────────────┘
```

A regression leaf carries `prediction` (the leaf mean, already un-centered — the
forest is fit on the globally centered outcome for numerical precision and adds
the mean back into each leaf). The full schema is in the
[cheat sheet](CHEATSHEET.md#model-table) and the header of `rf_macros.sql`.

`rf_summary` reads that table back into one human-readable row:

```sql
SELECT family, n_trees, n_nodes, n_leaves, max_depth_reached, mtry, depth_cap_hit
FROM rf_summary('cmodel');
-- ┌────────────────┬─────────┬─────────┬──────────┬───────────────────┬──────┬───────────────┐
-- │     family     │ n_trees │ n_nodes │ n_leaves │ max_depth_reached │ mtry │ depth_cap_hit │
-- │ classification │     100 │    1908 │     1004 │                10 │    2 │ false         │
-- └────────────────┴─────────┴─────────┴──────────┴───────────────────┴──────┴───────────────┘
```

## Predicting

`rf_class_predict` / `rf_reg_predict` score a table with a fitted model, matching
model features to columns **by name**. Extra columns (ids, the outcome itself, …)
pass through untouched.

**Regression** adds a `prediction` DOUBLE — the mean of the trees' leaf values:

```sql
SELECT species, body_mass_g, round(prediction, 1) AS predicted_mass
FROM rf_reg_predict('rmodel', 'penguins') LIMIT 3;
-- ┌───────────┬─────────────┬────────────────┐
-- │  species  │ body_mass_g │ predicted_mass │
-- │ Adelie    │        3499 │         3596.6 │
-- │ Chinstrap │        3955 │         3810.9 │
-- │ Gentoo    │        5314 │         5202.3 │
-- └───────────┴─────────────┴────────────────┘
```

**Classification uses soft voting** (as sklearn does, not a hard majority vote):
each tree contributes its leaf's *normalized* class distribution, the forest
probability is the mean of those vectors, and `pred` is the argmax — ties broken
to the **smallest** label. It adds `pred` VARCHAR and `probs` MAP(VARCHAR,DOUBLE),
dense over every training class:

```sql
SELECT species, pred, probs FROM rf_class_predict('cmodel', 'penguins') LIMIT 3;
-- ┌───────────┬───────────┬───────────────────────────────────────────┐
-- │  species  │   pred    │                   probs                   │
-- │ Adelie    │ Adelie    │ {Adelie=1.0, Chinstrap=0.0, Gentoo=0.0}   │
-- │ Chinstrap │ Chinstrap │ {Adelie=0.04, Chinstrap=0.96, Gentoo=0.0} │
-- │ Gentoo    │ Gentoo    │ {Adelie=0.02, Chinstrap=0.0, Gentoo=0.98} │
-- └───────────┴───────────┴───────────────────────────────────────────┘
```

Pull one class probability with `probs['Gentoo']`. The probabilities are the
soft-vote means across the 100 trees, so they need not be a hard `0`/`1`.

**Missing values — `na_action`.** By default (`na_action := 'null'`) a scoring
row with a `NULL` — or an unparseable value, or a missing column — in **any**
model feature is not scored at all and its outputs are `NULL` (duckLM's
contract). `na_action := 'skip_tree'` is more forgiving: the row *is* scored, and
a tree abstains only if its descent actually reaches a node that splits on the
feature that is `NULL` for that row. The prediction is then the average over the
trees that did reach a leaf. Since each tree tests only the handful of features
on the row's path, this recovers most rows that `'null'` throws away:

```sql
-- a penguin with bill_depth_mm missing
SELECT 'null'      AS na_action, pred FROM rf_class_predict('cmodel', 'one', 'null')
UNION ALL
SELECT 'skip_tree', pred FROM rf_class_predict('cmodel', 'one', 'skip_tree');
-- ┌───────────┬─────────┐
-- │ na_action │  pred   │
-- │ null      │ NULL    │   -- dropped: a feature was NULL
-- │ skip_tree │ Adelie  │   -- recovered from the trees that never needed it
-- └───────────┴─────────┘
```

**Capping the forest — `n_trees`.** `n_trees := NULL` scores with every tree;
`n_trees := k` scores with the first `k`. That is how you plot error against
forest size (a learning curve) from a single fitted model, without refitting:

```sql
SELECT 10  AS n_trees, accuracy FROM rf_class_evaluate('cmodel', 'penguins', 'species', n_trees := 10)
UNION ALL
SELECT 100,           accuracy FROM rf_class_evaluate('cmodel', 'penguins', 'species', n_trees := 100);
```

**Per-tree predictions — the ensemble spread.** `rf_reg_predict_trees` /
`rf_class_predict_trees` return one row per `(scoring row, tree)` instead of the
aggregated vote. `__rf_rid__` is the scoring row's 1-based ordinal, matching
`*_predict`'s order. This is duckRF's analogue of a prediction interval: the
scatter of the trees' predictions is the forest's own uncertainty. From it you
can build quantile-regression-forest bands, ensemble-variance diagnostics, or
learning curves:

```sql
-- a 90% band for one row from the tree-to-tree spread
SELECT round(quantile_cont(prediction, 0.05), 0) AS lo,
       round(avg(prediction), 1)                 AS mean,
       round(quantile_cont(prediction, 0.95), 0) AS hi
FROM rf_reg_predict_trees('rmodel', 'penguins') WHERE __rf_rid__ = 1;
-- ┌────────┬────────┬────────┐
-- │   lo   │  mean  │   hi   │
-- │ 3499.0 │ 3596.6 │ 3968.0 │
-- └────────┴────────┴────────┘
```

## Prediction intervals (`rf_reg_quantile`)

`rf_reg_predict_trees` above gives the spread of the *ensemble's estimate of the
mean* — narrow, and it under-covers real observations. For an **honest predictive
interval** for a new observation, use `rf_reg_quantile`, duckRF's Quantile
Regression Forest (Meinshausen 2006) and the analogue of duckLM's `*_predict_ci`.

The idea: the conditional distribution of `Y | X = x` is the **weighted empirical
distribution of the training responses that land in x's leaves** across the
forest. A training row `i` gets weight `w_i(x) = (1/T) Σ_t 1[leaf_t(x_i) =
leaf_t(x)] / n_t`, where `n_t` is the number of training rows in x's leaf of tree
`t`; the weights sum to 1. The level-α prediction is the type-1 inverse CDF of
that weighted pool — a real quantile of `Y`, not of the ensemble mean.

`rf_reg_quantile(model, tbl, outcome, quantiles, newdata := NULL, ...)` takes the
**reference sample** `tbl` (normally the training table) whose `outcome` responses
form the pool, a `DOUBLE[]` of levels in `(0, 1)`, and optional `newdata` to score
(`NULL` scores `tbl` itself). It returns the scored rows plus `quantile_pred`
MAP(DOUBLE, DOUBLE) mapping each level to its predicted value. Here the target's
noise deliberately grows with `x`, and the interval widens to match — uncertainty
the ensemble-mean spread cannot express:

```sql
-- y ~ x with heteroskedastic noise whose spread grows with x (deterministic)
CREATE TABLE demo AS
  SELECT (i / 100.0) AS x,
         (i / 100.0) + (((hash(i) % 2001)::BIGINT - 1000) / 1000.0) * (0.2 + i / 100.0) AS y
  FROM range(1, 601) t(i);
CREATE TABLE demo_m AS
  SELECT * FROM rf_reg_fit('demo', 'y', n_trees := 200, min_samples_leaf := 10);

SELECT round(x, 1) AS x,
       round(quantile_pred[0.05], 2) AS lo,
       round(quantile_pred[0.5],  2) AS median,
       round(quantile_pred[0.95], 2) AS hi,
       round(quantile_pred[0.95] - quantile_pred[0.05], 2) AS width
FROM rf_reg_quantile('demo_m', 'demo', 'y', [0.05, 0.5, 0.95])
WHERE x IN (1.0, 3.0, 5.0) ORDER BY x;
-- ┌─────┬───────┬────────┬──────┬───────┐
-- │  x  │  lo   │ median │  hi  │ width │
-- │ 1.0 │ -0.07 │   1.39 │ 2.09 │  2.16 │   -- narrow where noise is small
-- │ 3.0 │  0.07 │   2.24 │ 5.54 │  5.47 │
-- │ 5.0 │  0.38 │   3.63 │ 8.75 │  8.38 │   -- wide where noise is large
-- └─────┴───────┴────────┴──────┴───────┘
```

Pull a level with `quantile_pred[0.05]`. The `[0.05, 0.95]` band is a genuine 90%
predictive interval: on held-out data it covers ≈90% of actual observations (a
`[0.1, 0.9]` band ≈80%). The median tracks `rf_reg_predict`'s mean closely but need
not equal it. `na_action` and `n_trees` behave exactly as in `rf_reg_predict`; a
row no tree scored gets a `NULL` map. Cost caveat: it walks **both** the reference
sample and `newdata` through every tree, so it is heavier than `rf_reg_predict` —
pass a larger `min_samples_leaf` at fit time (leaves with only a handful of rows
give poorly-calibrated tails, a known QRF property) and cap the reference sample if
it is huge.

## Evaluating

`rf_class_evaluate` / `rf_reg_evaluate` score `tbl` with a fitted model and its
outcome column and return a one-row table of metrics. Rows with a `NULL` outcome,
or that no tree scored, are dropped — `n` counts what was actually evaluated.
Pass a holdout for an out-of-sample estimate, or use [`rf_*_oob`](#out-of-bag)
for an honest estimate with no holdout at all.

**Regression** → `n, rmse, mae, r2`:

```sql
SELECT * FROM rf_reg_evaluate('rmodel', 'penguins', 'body_mass_g');
-- ┌─────┬───────┬───────┬────────┐
-- │  n  │ rmse  │  mae  │   r2   │
-- │ 300 │ 93.40 │ 76.30 │ 0.9807 │
-- └─────┴───────┴───────┴────────┘
```

- `rmse` = `sqrt(mean (y − ŷ)²)`, `mae` = `mean |y − ŷ|`.
- `r2` = `1 − SSE/SST` with `SST = sum (y − mean(y))²` (sklearn's `r2_score`).

**Classification** → `n, accuracy, log_loss, brier, auc`:

```sql
SELECT * FROM rf_class_evaluate('cmodel', 'penguins', 'species');
-- ┌─────┬──────────┬──────────┬────────┬──────┐
-- │  n  │ accuracy │ log_loss │ brier  │ auc  │
-- │ 300 │      1.0 │  0.0119  │ 0.0020 │ NULL │
-- └─────┴──────────┴──────────┴────────┴──────┘
```

- `accuracy` = `mean[pred = y]`.
- `log_loss` = `−mean ln(clip(p_y, 1e-15, 1−1e-15))` — sklearn clips to the
  float64 machine epsilon and does not renormalize.
- `brier` = `mean Σ_k (1[y=k] − p_k)²`, **halved for binary** (sklearn's
  `scale_by_half='auto'` halves when there are fewer than 3 classes).
- `auc` is **binary only** (else `NULL`); the positive class is the
  lexicographically **greater** label, computed as the Mann-Whitney statistic on
  average ranks of its probability. It is `NULL` if a class is absent from `y`.

> **In-sample accuracy of `1.0` is expected, not suspicious.** A bagged forest of
> fully-grown trees memorizes its training rows, so scoring the training table
> flatters it. Never read a forest's fit from in-sample metrics — use
> [out-of-bag](#out-of-bag) or a holdout. On this data the OOB accuracy is `0.99`
> (vs the in-sample `1.0` above) and the OOB regression `r2` is `0.86` (vs the
> in-sample `0.98`).

For a binary outcome you get a real AUC:

```sql
-- species recoded to a boolean "is this a Gentoo?"
SELECT n, accuracy, round(auc, 3) AS auc FROM rf_class_oob('bmodel', 'pb', 'species');
-- ┌─────┬──────────┬─────┐
-- │  n  │ accuracy │ auc │
-- │ 300 │      1.0 │ 1.0 │
-- └─────┴──────────┴─────┘
```

## Out-of-bag

Because each tree leaves about a third of the rows out of its bootstrap sample,
you can score every training row using only the trees that did **not** see it —
an honest generalization estimate with no holdout. `rf_*_oob_predict` add the
prediction columns; `rf_*_oob` return the metric row plus `n_excluded` (rows that
happened to be in-bag for *every* tree and so have no OOB prediction):

```sql
SELECT * FROM rf_class_oob('cmodel', 'penguins', 'species');
-- ┌─────┬──────────┬──────────┬────────┬──────┬────────────┐
-- │  n  │ accuracy │ log_loss │ brier  │ auc  │ n_excluded │
-- │ 300 │     0.99 │  0.0373  │ 0.0152 │ NULL │          0 │
-- └─────┴──────────┴──────────┴────────┴──────┴────────────┘

SELECT * FROM rf_reg_oob('rmodel', 'penguins', 'body_mass_g');
-- ┌─────┬────────┬────────┬────────┬────────────┐
-- │  n  │  rmse  │  mae   │   r2   │ n_excluded │
-- │ 300 │ 249.47 │ 206.51 │ 0.8624 │          0 │
-- └─────┴────────┴────────┴────────┴────────────┘
```

**The exact-training-table requirement.** OOB reconstructs each tree's bootstrap
membership from the model metadata (`seed`, `sample_frac`, `replace_sample`,
`n_train`), and a training row's identity is the **ordinal of the complete rows**
— so `tbl` must be *exactly* the table the model was trained on: unfiltered, same
order. duckRF recomputes `n_train` and an order-dependent row fingerprint
(`train_hash`) and **errors** on any mismatch, rather than return plausible,
silently meaningless numbers:

```sql
SELECT * FROM rf_class_oob('cmodel', 'penguins_filtered', 'species');
-- Invalid Input Error: rf_class_oob: "penguins_filtered" has 200 complete rows
-- but the model was trained on 300; out-of-bag scoring requires the exact
-- training table (row identity is the row ordinal)
```

If you need OOB, keep the training table around unchanged (or re-`CREATE` it the
same way). For scoring *new* data, use `rf_*_predict` / `rf_*_evaluate` instead.

## Feature importance

`rf_importance` returns **mean decrease in impurity (MDI)**, matching sklearn's
`feature_importances_`: per tree, sum each split's `imp_decrease / w_root` by
feature and normalize to sum 1; a stump (a tree with no splits) contributes the
zero vector; average over all trees and renormalize. Every feature is listed —
`0` for ones no tree used — ordered by importance descending then name.

```sql
SELECT feature, round(importance, 4) AS importance FROM rf_importance('cmodel');
-- ┌───────────────────┬────────────┐
-- │      feature      │ importance │
-- │ bill_length_mm    │     0.3588 │
-- │ body_mass_g       │     0.3313 │
-- │ flipper_length_mm │     0.1462 │
-- │ island            │     0.0970 │
-- │ bill_depth_mm     │     0.0666 │
-- └───────────────────┴────────────┘
```

**Honest caveat: MDI is biased toward high-cardinality features** — features with
many distinct values (continuous numerics, and especially many-level
categoricals) get more chances to split and so accrue importance even when they
carry little signal. The bias is actually *worse* in duckRF than in sklearn
precisely because duckRF splits categoricals natively (a many-level categorical
is one powerful feature here, not a pile of one-hot columns). Treat MDI as a
quick, in-sample ranking. For an importance measure that survives scrutiny, reach
for **permutation importance** below.

### Permutation importance (`rf_permutation_importance`)

`rf_permutation_importance(model, tbl, outcome, n_repeats := 5, seed := 42)`
returns `feature, importance, importance_std` and matches
`sklearn.inspection.permutation_importance`. It measures how much the model's
**score degrades when a feature's column is randomly shuffled** on `tbl`: the
score is the estimator's default `.score()` — **R²** for a regression forest,
**accuracy** for a classification forest — computed with the normal prediction
(all trees, soft voting). For feature *j* and repeat *r* it permutes column *j*,
re-scores, and takes `baseline − permuted`; `importance` is the mean over the
`n_repeats` repeats and `importance_std` their population standard deviation
(`np.std`, ddof 0). Rows are the same ones `rf_*_evaluate` scores (every feature
present, outcome non-NULL), and the shuffle reuses those rows' values, so
completeness is preserved. Output is ordered by `importance DESC, feature`.

```sql
SELECT feature, round(importance, 4) AS importance, round(importance_std, 4) AS std
FROM rf_permutation_importance('cmodel', 'penguins', 'species', n_repeats := 20);
-- ┌───────────────────┬────────────┬────────┐
-- │      feature      │ importance │  std   │
-- │ flipper_length_mm │     0.5123 │ 0.0184 │
-- │ bill_length_mm    │     0.3011 │ 0.0142 │
-- │ island            │     0.0208 │ 0.0039 │
-- │ …                 │        …   │    …   │
-- └───────────────────┴────────────┴────────┘
```

**Why prefer it to MDI.** Permutation importance is **not biased by cardinality**:
a many-level categorical or a high-resolution numeric feature gets no free credit,
because a feature only scores if shuffling it actually *breaks predictions*. A
noise feature therefore sits at ~0 — and its importance **can go slightly
negative** (shuffling happened to help by chance); that sign is real and is kept.
Score it on a **holdout** table for a leakage-free ranking (it works on any table,
not just the training set), or on the training table for an in-sample view. Raise
`n_repeats` to shrink `importance_std`.

Caveats: it costs a full scoring pass per (feature × repeat), so it is heavier
than MDI; and like sklearn's, it can **split credit between two correlated
features** (shuffling either alone leaves the other to compensate, so both look
less important than the pair truly is). The permutation is md5-seeded (this
library's only randomness), not numpy's, so importances match sklearn
*statistically* (rank / top-k), not bit-for-bit — but they are **deterministic in
`seed`** under `PRAGMA threads=1`.

## Categorical features

`VARCHAR` and `ENUM` columns are **true categoricals**: the split is a **subset**
of the levels routed left, not a one-hot threshold. duckRF finds it with a single
cumulative-sum scan over the levels sorted by an impurity-relevant key. In the
model table a categorical split records both the levels going left (`cats_left`)
and right (`cats_right`), plus `unseen_left`:

```sql
SELECT split_feature, cats_left, cats_right, unseen_left
FROM cmodel WHERE split_kind = 'cat' AND tree = 1 LIMIT 1;
-- ┌───────────────┬─────────────────┬─────────────┬─────────────┐
-- │ split_feature │    cats_left    │ cats_right  │ unseen_left │
-- │ island        │ [Biscoe, Dream] │ [Torgersen] │ true        │
-- └───────────────┴─────────────────┴─────────────┴─────────────┘
```

**Why both lists?** A level present in *neither* list was never seen at that node
during training (common deep in a tree, or a genuinely new level at predict
time). `cats_left` alone cannot tell "seen, went right" from "never seen", so
predict routes an unseen level by `unseen_left` — it follows the **heavier**
child. That is why `cats_right` is stored explicitly.

**Why this beats one-hot.** A subset split can separate `{Biscoe, Dream}` from
`{Torgersen}` in a *single* node. One-hot encoding would need several
`island = X` indicators and several levels of the tree to express the same
partition, fragmenting the data, growing deeper trees, and diluting the feature's
importance across dummy columns. The native split keeps the categorical as one
feature with its full expressive power.

**How optimal it is — honestly.** For **regression and binary classification**
the scan is the **exact optimum** over all `2^(L−1)−1` non-trivial subsets: sort
the levels by the in-level mean of `y` (regression) or `P(y = positive | level)`
(binary), and the best prefix is provably the best subset (the Fisher/Breiman
result). For **K > 2 classes** it evaluates the `K` orderings by `P(y = k |
level)` and keeps the best — the standard Breiman/Ripley heuristic (LightGBM does
the same). That heuristic is **not guaranteed optimal** for K > 2, and it does
not enumerate every one-vs-rest singleton split. It is fast, standard, and works
well in practice, but it is a heuristic and we say so.

Rows with a `NULL` in a categorical feature (or any feature) are dropped from
training, as elsewhere.

## Tuning hyperparameters

`rf_cv` and `rf_cv_depth` run **k-fold cross-validation** over a grid, returning
one row per grid value with the held-out error. **Smaller `cv_error` is better**
— it is the misclassification rate for classification, MSE for regression. Pass
`family` as a word (`'classification'` | `'regression'`, matching duckLM's
`cv_l2` vocabulary) and the grid as `INTEGER[]`. The whole sweep is fit in one
recursive pass over the `(grid value × fold × tree)` space.

Sweep **mtry**:

```sql
SELECT * FROM rf_cv('penguins', 'species', 'classification', [1, 2, 3, 4, 5], k := 5, n_trees := 50)
ORDER BY cv_error;
-- ┌──────┬─────────────┐
-- │ mtry │  cv_error   │
-- │    1 │    0.023333 │
-- │    2 │    0.023333 │
-- │    4 │    0.026667 │
-- │    3 │    0.030000 │
-- │    5 │    0.033333 │
-- └──────┴─────────────┘
```

Sweep **max_depth**:

```sql
SELECT * FROM rf_cv_depth('penguins', 'body_mass_g', 'regression', [2, 4, 8, 20], k := 5, n_trees := 50)
ORDER BY cv_error LIMIT 1;
-- ┌───────────┬───────────┐
-- │ max_depth │ cv_error  │
-- │         2 │ 694857.02 │   -- the shallow forest already captures this signal
-- └───────────┴───────────┘
```

**Folds are assigned deterministically** as `(row# − 1) % k` over the complete
rows (duckLM's convention) — the rows are **not** shuffled. If your table is
ordered by the outcome, shuffle it first (e.g. build it with
`ORDER BY md5_number(rowid)`), or the folds will be unrepresentative. Cost scales
with `k · |grid| · trees · features · rows · depth`, so keep the grid modest.
`rf_cv` bags with the same bootstrap as `*_fit` (`sample_frac` of each fold's
training rows, with replacement) and uses `'gini'` / `'mse'`.

## Scaling to large forests (batch fitting)

The whole forest is grown in **one recursive CTE over all trees at once**, so peak
memory grows with `n_trees × max_depth`: each recursion level holds a frontier of
`≈ n_trees × n` `(tree, row)` assignments plus the model rows accumulated so far.
As a rough figure, a depth-20 forest on 50k × 20 data runs on the order of
~1–1.5 GB **per tree** — a few hundred trees can exhaust memory.

**Batch the fit.** `rf_class_fit` / `rf_reg_fit` take `tree_from` / `tree_to`
(both default `NULL` = every tree). They grow only the trees whose **global**
index is in `[tree_from, tree_to]`; `n_trees` stays the **total** forest size and
still drives the metadata, the RNG and OOB replay. Because every tree is grown
independently and all per-tree randomness keys on the global tree index, the
`UNION ALL` of the batches reproduces the one-shot fit:

```sql
CREATE TABLE m AS
      SELECT * FROM rf_reg_fit('big', 'y', n_trees := 300, max_depth := 18, tree_from := 1,   tree_to := 100)
  UNION ALL
      SELECT * FROM rf_reg_fit('big', 'y', n_trees := 300, max_depth := 18, tree_from := 101, tree_to := 200)
  UNION ALL
      SELECT * FROM rf_reg_fit('big', 'y', n_trees := 300, max_depth := 18, tree_from := 201, tree_to := 300);
```

Each arm's recursion is bounded to 100 trees, so peak memory is roughly a third of
the one-shot fit while the resulting `m` is the same 300-tree model. For an even
smaller envelope, fit the arms **sequentially** into one table (`CREATE TABLE m AS
<arm1>; INSERT INTO m <arm2>; …`) so only one batch is in flight at a time; the
arms are fully independent, so a driver can also fit them on **parallel
connections** and `UNION` the tables. Require `1 ≤ tree_from ≤ tree_to ≤ n_trees`
(a lone `NULL` takes its default of `1` / `n_trees`).

### `rf_batched_fit_sql` — write the batched UNION for you

`rf_batched_fit_sql(tbl, outcome, family, n_trees := 100, batch_size := 10, …)`
returns the `SELECT … UNION ALL SELECT …` **as text**, one arm per `batch_size`
trees. `family` (`'classification'` | `'regression'`) picks the fit macro,
`criterion := NULL` emits the family default, and `class_weight` is emitted only
for classification. Like duckLM's `dummy_encode_sql`, a macro can't grow a
data-dependent number of arms, so you **fetch the text and run it as a second
statement** (`query()` takes only a constant literal, so there is no inline form):

```python
# Python / R / any driver — two lines:
sql = con.sql("SELECT rf_batched_fit_sql('big','y','regression', "
              "n_trees:=300, batch_size:=50, max_depth:=18)").fetchone()[0]
con.sql(f"CREATE TABLE m AS {sql}")
```

Use **smaller batches for larger or deeper data**. `batch_size` need not divide
`n_trees` — batch *i* covers `[(i-1)·batch_size + 1, min(i·batch_size, n_trees)]`.

### Is the batched model *identical* to a one-shot fit?

In exact arithmetic, yes — bit for bit. In practice it depends on whether a node's
weight sums are integers, because DuckDB's hash aggregate sums them in an order
that depends on how many trees share the arm (the same float-associativity
described under [Reproducibility](#reproducibility) for threads):

* **Integer-slot fits** — classification with `'gini'`/`'entropy'`, no
  `class_weight` (or one that resolves to `1.0`), and integer/absent `weights_col`
  — sum exactly. The batched model is **bit-identical on every column**, `impurity`
  included, under `PRAGMA threads=1`.
* **Fractional-slot fits** — regression (the outcome is centered by a fractional
  mean), a fractional `weights_col`, or `class_weight := 'balanced'` on imbalanced
  classes — carry ~1e-11 noise in the slot sums, because DuckDB's `GROUP BY
  (tree, node)` sums a node's values in an order that depends on how many trees
  share the arm. At **shallow depth** the gains are well separated, so the **tree
  structure** (split feature, direction, routing, `n_rows`) is still reproduced
  exactly and only the `DOUBLE` columns (`threshold`, `prediction`, `impurity`,
  `imp_decrease`) move — they agree to ~1e-9 rather than bit-for-bit. But this is
  **not** a whole-forest guarantee: once you grow deep, near-tied gains appear and
  a flipped split cascades into different sub-trees. At the library's **default
  `max_depth := 20`** a batched regression forest diverges *structurally* from the
  one-shot fit — hundreds of nodes differ and forest predictions can differ by
  **~0.05–0.15** (measured: 400 trees × depth 20, n = 400). The batch grows a
  *different but statistically equivalent* forest — metadata is unchanged, OOB
  accuracy tracks closely — exactly the way two **multi-threaded** one-shot fits
  of the same data diverge. **Do not** rely on cell-for-cell equality of a batched
  vs one-shot regression / balanced-imbalanced model; rely on statistical
  equivalence.

Fit under `PRAGMA threads=1` for the strongest guarantee. Bit-identity of the
batched UNION holds **only** for integer-slot fits; for fractional-slot fits it is
statistical equivalence, not bit-for-bit reproduction.

## Contract & fine print

### Reproducibility

All randomness is `md5_number(seed || …)` — no `random()`, no `hash()` — a
specified algorithm stable across DuckDB versions, so a fit is **deterministic in
`seed`**. The bootstrap membership, the mtry feature lottery, and OOB
reconstruction are **bit-stable for a given seed regardless of threading**.

There is one caveat for **bit-identical model reproducibility**: the split-search
arithmetic (the impurity / gain slot sums) is evaluated by DuckDB's *parallel*
hash aggregation, and floating-point addition is non-associative, so partial sums
are combined in a thread-scheduling-dependent order. Two fits of the same table
with the same seed on a multi-threaded connection can therefore differ by ~1e-12
per node in `impurity` / `imp_decrease` / `prediction`, and when two candidate
splits are closer in gain than that noise, even pick a *different* split — so the
tree **structure** (and `rf_importance`) may differ run to run. The predictive
impact is negligible (forest predictions agree to ~1e-14). For a bit-identical
model across repeated runs and fresh connections, fit under **`PRAGMA
threads=1`**; that is the only configuration in which "same seed ⇒ identical
model" holds exactly.

### NULL handling

Training **drops any row with a `NULL` in the outcome or in *any* feature** — even
a feature no tree would go on to use (same contract as duckLM). A feature column
that is *entirely* `NULL` is rejected with a clear error rather than silently
excluded, as is an empty table or one with no feature column. At predict time,
`na_action := 'null'` yields `NULL` outputs for a row with a `NULL`/unparseable/
missing feature; `na_action := 'skip_tree'` lets the row be scored by the trees
whose path never needs the missing feature (see [Predicting](#predicting)).

### NaN / Inf

A `NaN` or `±Inf` in a numeric feature, or in a regression outcome, is **an
error** — it is not `NULL`, it would survive into the slot sums, and a `NaN` gain
both passes a `>` filter and sorts first, silently hijacking the split search.
Clean or drop those rows first.

```sql
SELECT * FROM rf_reg_fit('penguins_with_nan', 'body_mass_g');
-- Invalid Input Error: rf_reg_fit: feature column(s) contain NaN or Inf:
-- "bill_length_mm"; clean or drop these rows
```

### Reserved names

Column **and table** names beginning with `__rf_` (any case) are reserved
everywhere. A feature named `pred` or `probs` (classification) or `prediction`
(regression) is rejected **at fit** — it would collide with the predict output
columns and produce a model that could never be scored:

```sql
SELECT * FROM rf_reg_fit('t', 'y');   -- t has a feature column named "prediction"
-- Invalid Input Error: rf_reg_fit: a feature column named "prediction" collides
-- with rf_reg_predict's output column and the model could never be scored;
-- please rename it
```

Feature types must be numeric (any int/uint width, `FLOAT`, `DOUBLE`, `DECIMAL`),
`BOOLEAN`, `VARCHAR`, or `ENUM`. Anything else (`DATE`, `TIMESTAMP`, `BLOB`,
`LIST`, `STRUCT`, …) errors by name and type — cast it first. A single-class
classification outcome errors.

### <a id="max_depth-binds"></a>max_depth binds

The default `max_depth := 20` **deviates from sklearn** (whose default is
unlimited) and **does bind**: node ids are heap-numbered (so depth is capped at
60 to stay inside `BIGINT`), and depth is the recursion's iteration count, so both
time and peak memory grow linearly in it. A fully grown CART is ~`2·log2(n)` deep,
so on a regression forest even a few hundred rows can hit the cap. `rf_summary`
reports `depth_cap_hit` — `true` iff some tree was actually truncated (an impure
leaf that only stopped at the cap). Pass `max_depth := NULL` to grow to purity
(hard cap 60):

```sql
SELECT max_depth, depth_cap_hit, max_depth_reached FROM rf_summary('rmodel')          -- default 20
UNION ALL
SELECT max_depth, depth_cap_hit, max_depth_reached FROM rf_summary('rmodel_unlimited');-- max_depth := NULL
-- ┌───────────┬───────────────┬───────────────────┐
-- │ max_depth │ depth_cap_hit │ max_depth_reached │
-- │        20 │ true          │                20 │   -- the default DID stunt this forest
-- │        60 │ false         │                25 │
-- └───────────┴───────────────┴───────────────────┘
```

### Out-of-bag needs the exact training table

Row identity is the ordinal of the complete rows, so `rf_*_oob*` require *exactly*
the training table — unfiltered, same order. `n_train` and `train_hash` are
recomputed and a mismatch is a clear error (see [Out-of-bag](#out-of-bag)).

### Single-tree CART equivalence

A forest of one tree, trained on the whole data without bagging and considering
every feature at each node — `n_trees := 1, sample_frac := 1.0, replace_sample :=
false, mtry := d` — is exactly one CART tree:

```sql
CREATE TABLE cart AS
SELECT * FROM rf_reg_fit('penguins', 'body_mass_g',
         n_trees := 1, sample_frac := 1.0, replace_sample := false, mtry := 5, max_depth := NULL);
SELECT n_trees, n_nodes, n_leaves, max_depth_reached FROM rf_summary('cart');
-- ┌─────────┬─────────┬──────────┬───────────────────┐
-- │ n_trees │ n_nodes │ n_leaves │ max_depth_reached │
-- │       1 │     599 │      300 │                19 │
-- └─────────┴─────────┴──────────┴───────────────────┘
```

In that configuration duckRF matches sklearn's `DecisionTreeRegressor` /
`DecisionTreeClassifier` to ~1e-9 on data with no tied splits (the test suite
sees regression agreement to ~2e-15 and classification `predict_proba` agreement
of `0.0` at shallow depth). Where splits are genuinely tied, tie-breaking
conventions can send the two implementations down different but equally-good
paths — as they can between any two CART implementations.
