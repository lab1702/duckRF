-- ============================================================================
-- Random forests (classification & regression) as pure DuckDB (>= 1.5) SQL
-- macros. No extensions, no driver, no UDFs: the whole forest is grown inside
-- ONE recursive CTE, breadth-first, all trees at once.
--
-- Sibling project to duckLM (https://github.com/lab1702/duckLM); same
-- conventions: table names are passed as *strings* and resolved with
-- query_table(), every macro is a CREATE OR REPLACE MACRO, internal helpers
-- are prefixed __rf_.
--
-- Public macros (this file -- fitting):
--   rf_class_fit(tbl, outcome, ...)  -> forest model table (see below)
--   rf_reg_fit(tbl, outcome, ...)    -> forest model table (see below)
--
--   NOTE on naming: the two families are rf_class_* / rf_reg_*, not rfc_* /
--   rfr_*. Both families accept the same argument list and the same outcome
--   types (rf_class_ casts the outcome to VARCHAR, so a 0/1 column fits
--   happily under either), so a one-character difference between the prefixes
--   would turn a typo into a silently wrong MODEL FAMILY rather than an error.
--   The family tokens differ by three characters on purpose.
--
-- Scoring / diagnostics (rf_class_predict, rf_reg_predict, *_evaluate,
-- *_oob*, rf_importance, rf_permutation_importance, rf_summary, rf_cv) live
-- further down this file.
--
-- Internal helper (do not call directly, subject to change):
--   __rf_fit(tbl, outcome, family, caller, n_trees, mtry, max_depth,
--            min_samples_split, min_samples_leaf, min_impurity_decrease,
--            sample_frac, replace_sample, criterion, seed, weights_col,
--            class_weight)
--
-- Fit parameters (identical for both families):
--   n_trees               := 100    number of trees
--   mtry                  := NULL   features sampled per node; NULL means
--                                   greatest(1, floor(sqrt(d))) for
--                                   classification and d for regression,
--                                   matching sklearn's max_features='sqrt' /
--                                   1.0 defaults
--   max_depth             := 20     NULL grows to purity. NOTE the deviation
--                                   from sklearn (whose default is unlimited):
--                                   node ids are heap numbered (root = 1,
--                                   children 2v / 2v+1), so depth is capped at
--                                   60 to stay inside BIGINT, and depth is the
--                                   iteration count of the recursion -- both
--                                   time and peak memory grow linearly in it.
--                                   A fully grown CART is ~2*log2(n) deep, so
--                                   max_depth := 20 DOES bind above a few
--                                   thousand rows; rf_summary(model) reports
--                                   depth_cap_hit.
--   min_samples_split     := 2      minimum rows in a node to consider a split
--   min_samples_leaf      := 1      minimum rows in each child
--   min_impurity_decrease := 0.0    a split is accepted iff
--                                   imp_decrease / w_root + eps >= this
--                                   (>=, exactly as sklearn: zero-gain splits
--                                   on an impure node ARE made)
--   sample_frac           := 1.0    bootstrap sample size as a fraction of n
--                                   (0 < sample_frac <= 1, like sklearn's
--                                   max_samples)
--   replace_sample        := true   sample with replacement (bagging). false
--                                   takes a random subset without replacement
--   criterion             := 'gini' 'gini' | 'entropy' for classification
--                                   (entropy is in BITS, i.e. log base 2, as
--                                   sklearn's is), 'mse' for regression
--   seed                  := 42     all randomness is md5_number(seed || ...)
--   weights_col           := NULL   column of non-negative per-row sample
--                                   weights; multiplies the bootstrap count
--                                   (sklearn's sample_weight)
--   class_weight          := NULL   classification only; 'balanced' multiplies
--                                   each row's weight by n / (K * n_k)
--
-- Requirements / behavior:
--   * Feature columns must be numeric (any int/uint width, FLOAT, DOUBLE,
--     DECIMAL), BOOLEAN, VARCHAR or ENUM. Anything else (DATE, TIMESTAMP,
--     BLOB, LIST, STRUCT, ...) errors by name and type -- cast it first.
--   * VARCHAR / ENUM features are true categoricals: the split is a SUBSET of
--     the levels (cats_left), found by the Fisher/Breiman prefix scan, not a
--     one-hot threshold. For regression and binary classification the prefix
--     scan is the exact optimum over all 2^(L-1)-1 subsets; for K > 2 classes
--     it evaluates the K orderings by P(y = k | level) and keeps the best --
--     the standard Breiman/Ripley heuristic (LightGBM does the same). It is
--     NOT guaranteed optimal for K > 2, and it does not enumerate every
--     one-vs-rest singleton split.
--   * Rows with a NULL in the outcome or in ANY feature are dropped from
--     training (same contract as duckLM). A feature column that is entirely
--     NULL errors, as does an empty table or a table with no feature column.
--   * NaN / +-Inf in a numeric feature or in a regression outcome errors: they
--     are not NULL, they would survive into the slot sums, and a NaN gain both
--     passes a ">" filter and sorts first, silently hijacking the split search.
--   * rf_reg_fit's outcome must be numeric; rf_class_fit casts the outcome to
--     VARCHAR (BOOLEAN / INTEGER / VARCHAR labels all work). A single-class
--     classification outcome errors.
--   * Column AND table names beginning with "__rf_" (any case) are reserved.
--   * Randomness is deterministic given seed -- no random(), no hash(), only
--     md5_number (a specified algorithm, stable across DuckDB versions). The
--     bootstrap membership, the mtry feature lottery and OOB reconstruction are
--     therefore bit-stable for a given seed regardless of threading.
--
--     CAVEAT on bit-exact model REPRODUCIBILITY: the split-search arithmetic
--     (the impurity / gain slot sums) is evaluated by DuckDB's PARALLEL hash
--     aggregation, and floating-point addition is non-associative, so partial
--     sums are combined in a thread-scheduling-dependent order. Two fits of the
--     same table with the same seed on a multi-threaded connection can thus
--     differ by ~1e-12 per node in impurity / imp_decrease / prediction, and --
--     when two candidate splits are closer in gain than that noise -- can even
--     pick a different split, so the tree STRUCTURE (and rf_importance) may
--     differ run to run. The predictive impact is negligible (forest
--     predictions agree to ~1e-14). For a BIT-IDENTICAL model across repeated
--     runs and fresh connections, fit under `PRAGMA threads=1`; that is the only
--     configuration in which "same seed => identical model" holds exactly.
--
-- Model table (one row per tree node; every *_fit returns this shape):
--   tree                  INTEGER   1..n_trees
--   node                  BIGINT    heap numbering: root = 1, children 2v, 2v+1
--   depth                 INTEGER   root = 0
--   is_leaf               BOOLEAN
--   split_feature         VARCHAR   NULL at leaves
--   split_kind            VARCHAR   'num' | 'cat' | NULL
--   threshold             DOUBLE    numeric split: LEFT iff value <= threshold
--   cats_left             VARCHAR[] categorical: levels routed LEFT
--   cats_right            VARCHAR[] categorical: levels routed RIGHT. Both
--                                   lists are needed: a level absent from BOTH
--                                   was not seen at this node in training and
--                                   goes to unseen_left. cats_left alone cannot
--                                   distinguish "seen, went right" from
--                                   "never seen".
--   unseen_left           BOOLEAN   categorical: where an unseen level goes
--                                   (the heavier child)
--   n_rows                BIGINT    distinct training rows in the node
--   w_node                DOUBLE    total bootstrap weight in the node
--   impurity              DOUBLE    gini / entropy (bits) / variance
--   imp_decrease          DOUBLE    w_par*imp_par - w_L*imp_L - w_R*imp_R
--                                   (NULL at leaves). Divide by w_root for
--                                   sklearn's "improvement".
--   prediction            DOUBLE    regression leaf value (NULL otherwise)
--   class_counts          MAP(VARCHAR, DOUBLE)  classification leaf weighted
--                                   counts. DENSE: one key per training class,
--                                   zeros included, so probability vectors can
--                                   be averaged across trees key-by-key.
--   -- forest metadata, constant on every row (dictionary-compressed to
--   -- nothing). It makes the model self-describing: *_oob_* can rebuild the
--   -- bootstrap membership, rf_importance can report 0 for never-used
--   -- features, predict can emit a dense probability vector, and rf_summary
--   -- can tell the user how the thing was actually fit.
--   family                VARCHAR   'classification' | 'regression'
--   n_trees               INTEGER
--   seed                  BIGINT
--   sample_frac           DOUBLE
--   replace_sample        BOOLEAN
--   n_train               BIGINT    complete training rows
--   mtry                  INTEGER   effective (after the NULL default)
--   max_depth             INTEGER   effective (after the NULL default)
--   min_samples_split     INTEGER
--   min_samples_leaf      INTEGER
--   min_impurity_decrease DOUBLE
--   criterion             VARCHAR
--   features              VARCHAR[] every feature column, in name order
--   feature_kinds         VARCHAR[] 'num' | 'cat', aligned with features
--   classes               VARCHAR[] sorted class labels (NULL for regression)
--   train_hash            HUGEINT   order-dependent fingerprint of the complete
--                                   training rows, so *_oob_* can refuse a
--                                   table that is not the one we trained on
--
-- Example:
--   CREATE TABLE m AS SELECT * FROM rf_class_fit('iris', 'species');
--   SELECT * FROM rf_class_predict('m', 'iris_new');
-- ============================================================================

-- NOTE on naming: every internal CTE is prefixed __rf_ because query_table()
-- resolves bare table names against CTEs already defined in the enclosing
-- WITH; an internal CTE named e.g. "rows" would shadow a user table of the
-- same name. The __rf_ prefix is reserved (and enforced below).

-- NOTE on FROM lists: never write "FROM a, b LEFT JOIN c ON ...". The comma
-- binds looser than JOIN, so that parses as a CROSS JOIN (b LEFT JOIN c) and
-- makes a's columns correlated on the null-producing side; inside a recursive
-- term DuckDB cannot decorrelate that and the macro fails to BIND with
-- "Non-inner join on correlated columns not supported" -- an error that names
-- none of the twenty CTEs involved. Always spell the CROSS JOIN out.


-- ---------------------------------------------------------------------------
-- Slot-vector algebra.
--
-- Both families are driven by one split-search core by carrying node statistics
-- as a vector of "slots":
--   classification: one slot per class k, s_k = sum of w over rows with y = k
--   regression:     three slots, (sum w, sum w*y, sum w*y*y)
-- For a node with slot vector s and weight W:
--   W(s)  = sum_k s_k          (classification)          | s_1 (regression)
--   Q(s)  = sum_k s_k^2 / W    (gini)
--         = sum_k s_k*log2(s_k/W)                        (entropy, in BITS --
--           sklearn's Entropy criterion is log base 2, and min_impurity_decrease
--           is compared against a number in those units, so ln would silently
--           rescale the user's threshold by 1/ln2)
--         = s_2^2 / s_1                                  (mse)
--   imp(s) = 1 - sum (s_k/W)^2 | -sum (s_k/W) log2(s_k/W) | s_3/s_1 - (s_2/s_1)^2
-- and, for every criterion, imp_decrease = Q(left) + Q(right) - Q(parent):
-- the parent term and (for mse) the sum-of-squares term are constant across
-- candidate splits, so ONE cumulative-sum machine maximizing Q_L + Q_R finds
-- the best split for all three.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO __rf_wt(vec, crit) AS (
    CASE WHEN crit = 'mse' THEN vec[1] ELSE list_sum(vec) END
);

CREATE OR REPLACE MACRO __rf_q(vec, crit) AS (
    CASE crit
      WHEN 'gini'    THEN list_sum(list_transform(vec, lambda x: x * x)) / list_sum(vec)
      WHEN 'entropy' THEN list_sum(list_transform(vec, lambda x:
                              CASE WHEN x > 0 THEN x * log2(x / list_sum(vec)) ELSE 0.0 END))
      ELSE vec[2] * vec[2] / vec[1]
    END
);

CREATE OR REPLACE MACRO __rf_imp(vec, crit) AS (
    CASE crit
      WHEN 'gini'    THEN 1.0 - list_sum(list_transform(vec, lambda x:
                              (x / list_sum(vec)) * (x / list_sum(vec))))
      WHEN 'entropy' THEN -list_sum(list_transform(vec, lambda x:
                              CASE WHEN x > 0 THEN (x / list_sum(vec)) * log2(x / list_sum(vec))
                                   ELSE 0.0 END))
      -- variance, floored at 0: computed on the GLOBALLY CENTERED outcome (see
      -- __rf_ybar), so the E[y^2] - E[y]^2 cancellation that ruins split choice
      -- on large-mean targets (prices, counts, years) cannot bite. Centering is
      -- exactly gain-invariant.
      ELSE greatest(vec[3] / vec[1] - (vec[2] / vec[1]) * (vec[2] / vec[1]), 0.0)
    END
);


CREATE OR REPLACE MACRO __rf_fit(tbl, outcome, family, caller, n_trees, mtry, max_depth,
                                 min_samples_split, min_samples_leaf, min_impurity_decrease,
                                 sample_frac, replace_sample, criterion, seed,
                                 weights_col, class_weight) AS TABLE
WITH RECURSIVE
-- Column types. DESCRIBE does not bind query_table() inside a macro; this
-- LEFT JOIN onto a constant row does, keeps column names AND order, and on an
-- empty table returns one all-NULL typename row per column -- which is exactly
-- how the empty-table check below detects emptiness.
__rf_types AS MATERIALIZED (
    SELECT colname, typename,
           CASE WHEN typename IN ('BOOLEAN','TINYINT','SMALLINT','INTEGER','BIGINT','HUGEINT',
                                  'UTINYINT','USMALLINT','UINTEGER','UBIGINT','UHUGEINT',
                                  'FLOAT','DOUBLE')
                     OR starts_with(typename, 'DECIMAL') THEN 'num'
                WHEN typename = 'VARCHAR' OR starts_with(typename, 'ENUM') THEN 'cat'
                WHEN typename IS NULL THEN NULL
                ELSE 'bad'
           END AS kind
    FROM (SELECT *
          FROM (SELECT 1 AS __rf_one)
          LEFT JOIN (SELECT typeof(COLUMNS('^(.*)$')) AS '\1' FROM query_table(tbl) LIMIT 1) ON true)
         UNPIVOT INCLUDE NULLS (typename FOR colname IN (COLUMNS(* EXCLUDE (__rf_one))))
),
-- ONE VARCHAR long-form for everything. UNPIVOT needs a single common type and
-- which columns are numeric is data-dependent, so cast every column to VARCHAR
-- once and recover numbers from the discovered types. CAST(DOUBLE AS VARCHAR)
-- and back is bit-exact, so nothing is lost. It also sidesteps
-- TRY_CAST(DATE AS DOUBLE) binder failures and the BOOLEAN 'true'/'false'
-- round-trip. UNPIVOT drops NULL cells -- that is how incomplete rows are
-- detected below.
__rf_slong AS MATERIALIZED (
    SELECT __rf_rid__ AS rid, name AS col, value AS sval
    FROM (UNPIVOT (SELECT row_number() OVER () AS __rf_rid__, CAST(COLUMNS(*) AS VARCHAR)
                   FROM query_table(tbl))
          ON COLUMNS(* EXCLUDE (__rf_rid__)) INTO NAME name VALUE value)
),
__rf_present AS (SELECT DISTINCT col FROM __rf_slong),
-- Feature columns = everything except the outcome and the optional weights
-- column. j is the position in name order; the model's features[] array is
-- built in this order.
__rf_featcols AS MATERIALIZED (
    SELECT colname AS col, kind, typename,
           row_number() OVER (ORDER BY colname) AS j
    FROM __rf_types
    WHERE colname != outcome AND colname != coalesce(weights_col, '')
),
__rf_d AS (SELECT count(*)::BIGINT AS d FROM __rf_featcols),
-- A row is COMPLETE iff it has one non-NULL cell per column of the table.
__rf_complete AS MATERIALIZED (
    SELECT rid FROM __rf_slong
    GROUP BY rid
    HAVING count(*) = (SELECT count(*) FROM __rf_types)
),
__rf_ysval AS MATERIALIZED (
    SELECT s.rid, s.sval,
           CASE WHEN t.typename = 'BOOLEAN' THEN CASE WHEN s.sval = 'true' THEN 1.0 ELSE 0.0 END
                WHEN t.kind = 'num' THEN TRY_CAST(s.sval AS DOUBLE) END AS yv
    FROM __rf_slong s
    SEMI JOIN __rf_complete c ON c.rid = s.rid
    JOIN __rf_types t ON t.colname = s.col
    WHERE s.col = outcome
),
__rf_classes AS MATERIALIZED (
    SELECT sval AS cls, row_number() OVER (ORDER BY sval) AS k
    FROM (SELECT DISTINCT sval FROM __rf_ysval)
    WHERE family = 'classification'
),
__rf_classlist AS (SELECT list(cls ORDER BY k) AS classes FROM __rf_classes),
-- Numeric feature cells over complete rows, used only for the NaN/Inf guard
-- (the real feature table __rf_feat is renumbered and comes later).
__rf_numcells AS (
    SELECT s.col, TRY_CAST(s.sval AS DOUBLE) AS v
    FROM __rf_slong s
    SEMI JOIN __rf_complete c ON c.rid = s.rid
    JOIN __rf_featcols f ON f.col = s.col
    WHERE f.kind = 'num' AND f.typename != 'BOOLEAN'
),
-- Effective hyperparameters, so that the model records what was actually used.
__rf_cfg AS (
    SELECT coalesce(mtry, CASE WHEN family = 'classification'
                               THEN greatest(1, floor(sqrt((SELECT d FROM __rf_d))))
                               ELSE (SELECT d FROM __rf_d) END)::BIGINT AS mtry_eff,
           coalesce(max_depth, 60)::BIGINT                              AS depth_eff
),
-- Every guard in one CASE, evaluated in order. A guard CTE only fires if the
-- consuming query REFERENCES its boolean -- a cross-joined "ok" column that is
-- projected away gets optimized out and the error never raises. So __rf_rows
-- below does "CROSS JOIN __rf_chk ck ... WHERE ck.ok", and so does the final
-- SELECT. (The aggregate-free scalar-subquery form here means this CTE always
-- produces exactly one row, even on an empty input table, so the checks still
-- run when there is no data at all.)
__rf_chk AS (
    SELECT CASE
             WHEN starts_with(lower(tbl), '__rf_')
               THEN error(caller || ': table names beginning with "__rf_" are reserved for internal use; please rename')
             WHEN (SELECT count(*) FROM __rf_types WHERE starts_with(lower(colname), '__rf_')) > 0
               THEN error(caller || ': column names beginning with "__rf_" are reserved for internal use; please rename')
             WHEN (SELECT coalesce(bool_and(typename IS NULL), true) FROM __rf_types)
               THEN error(caller || ': input table "' || tbl || '" is empty')
             WHEN (SELECT count(*) FROM __rf_types WHERE colname = outcome) = 0
               THEN error(caller || ': outcome column "' || outcome || '" not found in "' || tbl || '"')
             WHEN weights_col IS NOT NULL
                  AND (SELECT count(*) FROM __rf_types WHERE colname = weights_col) = 0
               THEN error(caller || ': weights column "' || weights_col || '" not found in "' || tbl || '"')
             WHEN (SELECT d FROM __rf_d) = 0
               THEN error(caller || ': no feature columns besides the outcome')
             WHEN (SELECT count(*) FROM __rf_featcols WHERE kind = 'bad') > 0
               THEN error(caller || ': unsupported feature column type(s): '
                          || (SELECT string_agg('"' || col || '" is ' || typename, ', ' ORDER BY col)
                              FROM __rf_featcols WHERE kind = 'bad')
                          || '; features must be numeric, BOOLEAN, VARCHAR or ENUM -- cast or drop them')
             -- Reserved output-name collisions, rejected at FIT time (mirroring
             -- duckLM's '(Intercept)' guard). A feature named 'pred'/'probs'
             -- (classification) or 'prediction' (regression) would build a model
             -- that rf_*_predict / rf_*_oob_predict can NEVER score: the scoring
             -- table must carry that same-named feature, which then trips the
             -- predict-time output-collision guard. Rejecting here (case-
             -- insensitively) turns a later, inconsistent failure into a clear
             -- one at fit.
             WHEN family = 'classification'
                  AND (SELECT count(*) FROM __rf_featcols WHERE lower(col) IN ('pred', 'probs')) > 0
               THEN error(caller || ': a feature column named "pred" or "probs" collides with rf_class_predict''s output columns and the model could never be scored; please rename it')
             WHEN family = 'regression'
                  AND (SELECT count(*) FROM __rf_featcols WHERE lower(col) = 'prediction') > 0
               THEN error(caller || ': a feature column named "prediction" collides with rf_reg_predict''s output column and the model could never be scored; please rename it')
             WHEN family = 'regression'
                  AND (SELECT kind FROM __rf_types WHERE colname = outcome) != 'num'
               THEN error(caller || ': outcome column "' || outcome || '" must be numeric for regression, but is '
                          || (SELECT typename FROM __rf_types WHERE colname = outcome)
                          || ' -- use rf_class_fit for a categorical outcome')
             -- A feature column with no non-NULL value anywhere would silently
             -- vanish; reject it rather than fit a model the user did not ask for.
             WHEN (SELECT count(*) FROM __rf_featcols f
                   WHERE f.col NOT IN (SELECT col FROM __rf_present)) > 0
               THEN error(caller || ': feature column(s) entirely NULL: '
                          || (SELECT string_agg('"' || f.col || '"', ', ' ORDER BY f.col)
                              FROM __rf_featcols f WHERE f.col NOT IN (SELECT col FROM __rf_present))
                          || '; drop them (e.g. SELECT * EXCLUDE (...)) or fill them')
             WHEN (SELECT count(*) FROM __rf_present WHERE col = outcome) = 0
               THEN error(caller || ': outcome column "' || outcome || '" is entirely NULL')
             WHEN (SELECT count(*) FROM __rf_complete) = 0
               THEN error(caller || ': no complete (non-NULL) rows to train on')
             -- NaN and +-Inf are NOT NULL, so they survive the completeness test,
             -- poison a node's slot sums, and a NaN gain sorts FIRST under
             -- "ORDER BY gain DESC" -- one bad cell would silently choose the split.
             WHEN (SELECT count(*) FROM __rf_numcells WHERE NOT isfinite(v)) > 0
               THEN error(caller || ': feature column(s) contain NaN or Inf: '
                          || (SELECT string_agg(DISTINCT '"' || col || '"', ', ')
                              FROM __rf_numcells WHERE NOT isfinite(v))
                          || '; clean or drop these rows')
             WHEN family = 'regression'
                  AND (SELECT count(*) FROM __rf_ysval WHERE NOT isfinite(yv)) > 0
               THEN error(caller || ': outcome column "' || outcome || '" contains NaN or Inf; clean or drop these rows')
             WHEN family = 'classification' AND (SELECT count(*) FROM __rf_classes) < 2
               THEN error(caller || ': outcome column "' || outcome || '" has a single class ("'
                          || (SELECT any_value(cls) FROM __rf_classes)
                          || '"); classification needs at least two')
             WHEN n_trees < 1
               THEN error(caller || ': n_trees must be >= 1, got ' || n_trees)
             WHEN mtry IS NOT NULL AND (mtry < 1 OR mtry > (SELECT d FROM __rf_d))
               THEN error(caller || ': mtry must be between 1 and the number of features ('
                          || (SELECT d FROM __rf_d) || '), got ' || mtry)
             WHEN max_depth IS NOT NULL AND (max_depth < 1 OR max_depth > 60)
               THEN error(caller || ': max_depth must be between 1 and 60, or NULL to grow to purity; got ' || max_depth)
             WHEN min_samples_split < 2
               THEN error(caller || ': min_samples_split must be >= 2, got ' || min_samples_split)
             WHEN min_samples_leaf < 1
               THEN error(caller || ': min_samples_leaf must be >= 1, got ' || min_samples_leaf)
             WHEN min_impurity_decrease < 0
               THEN error(caller || ': min_impurity_decrease must be >= 0, got ' || min_impurity_decrease)
             WHEN sample_frac <= 0 OR sample_frac > 1
               THEN error(caller || ': sample_frac must be in (0, 1], got ' || sample_frac)
             WHEN family = 'classification' AND criterion NOT IN ('gini', 'entropy')
               THEN error(caller || ': criterion must be ''gini'' or ''entropy'', got ''' || criterion || '''')
             WHEN family = 'regression' AND criterion != 'mse'
               THEN error(caller || ': criterion must be ''mse'', got ''' || criterion || '''')
             WHEN class_weight IS NOT NULL AND class_weight != 'balanced'
               THEN error(caller || ': class_weight must be NULL or ''balanced'', got ''' || class_weight || '''')
             WHEN seed IS NULL
               THEN error(caller || ': seed must not be NULL (the forest is deterministic in it)')
             ELSE true
           END AS ok
),
-- Complete rows renumbered 1..n. Everything downstream keys off i, never off
-- the raw row_number(): the model must not depend on how many NULL rows were
-- dropped. This is also where the guards are forced to fire.
__rf_rows AS MATERIALIZED (
    SELECT c.rid, row_number() OVER (ORDER BY c.rid) AS i
    FROM __rf_complete c CROSS JOIN __rf_chk ck
    WHERE ck.ok
),
__rf_n AS (SELECT count(*)::BIGINT AS n FROM __rf_rows),
-- Global weighted-free mean of the regression outcome. The outcome is fit on
-- y - ybar throughout and ybar is added back into the leaf prediction: gain and
-- impurity are provably invariant to shifting y, but the naive
-- sum(w*y^2) - sum(w*y)^2/W form loses catastrophic precision when
-- |mean(y)| >> sd(y) (prices, revenue, counts, years) -- enough to pick a
-- genuinely WORSE split than the true argmax. One extra pass buys exactness.
__rf_ybar AS (
    SELECT CASE WHEN family = 'regression' THEN coalesce(avg(y.yv), 0.0) ELSE 0.0 END AS ybar
    FROM __rf_ysval y SEMI JOIN __rf_rows r ON r.rid = y.rid
),
__rf_y AS MATERIALIZED (
    SELECT r.i AS rid, y.sval AS cls, y.yv - (SELECT ybar FROM __rf_ybar) AS yv
    FROM __rf_ysval y JOIN __rf_rows r ON r.rid = y.rid
),
-- Per-row sample weight: the weights column (1.0 when absent) times the
-- 'balanced' class weight n / (K * n_k) (sklearn's exact formula; the class
-- counts are unweighted, as sklearn's compute_class_weight uses bincount(y)).
__rf_wcol AS (
    SELECT r.i AS rid, coalesce(TRY_CAST(s.sval AS DOUBLE), 1.0) AS w
    FROM __rf_rows r
    LEFT JOIN __rf_slong s ON s.rid = r.rid AND s.col = coalesce(weights_col, '')
),
__rf_cw AS (
    SELECT y.cls,
           (SELECT n FROM __rf_n)::DOUBLE
             / ((SELECT count(*) FROM __rf_classes) * count(*)) AS cw
    FROM __rf_y y
    WHERE class_weight = 'balanced'
    GROUP BY y.cls
),
__rf_rw AS MATERIALIZED (
    SELECT w.rid, w.w * coalesce(cw.cw, 1.0) AS rw
    FROM __rf_wcol w
    JOIN __rf_y y ON y.rid = w.rid
    LEFT JOIN __rf_cw cw ON cw.cls = y.cls
),
__rf_wchk AS (
    SELECT CASE
             WHEN weights_col IS NOT NULL AND min(w) < 0
               THEN error(caller || ': weights must be non-negative')
             WHEN weights_col IS NOT NULL AND coalesce(sum(w), 0) <= 0
               THEN error(caller || ': sample weights sum to zero')
             ELSE true
           END AS ok
    FROM __rf_wcol
),
-- Slots: one per class (classification) or the three moments (regression).
__rf_slots AS MATERIALIZED (
    SELECT k AS slot FROM __rf_classes
    UNION ALL
    SELECT unnest([1, 2, 3]) AS slot WHERE family = 'regression'
),
__rf_S AS (SELECT count(*)::BIGINT AS ns FROM __rf_slots),
-- Per-row slot contribution u: a row contributes w*u to slot s.
__rf_u AS MATERIALIZED (
    SELECT y.rid, c.k AS slot, 1.0::DOUBLE AS u
    FROM __rf_y y JOIN __rf_classes c ON c.cls = y.cls
    UNION ALL
    SELECT y.rid, s.slot,
           CASE s.slot WHEN 1 THEN 1.0::DOUBLE WHEN 2 THEN y.yv ELSE y.yv * y.yv END
    FROM __rf_y y CROSS JOIN (SELECT unnest([1, 2, 3]) AS slot) s
    WHERE family = 'regression'
),
-- Feature cells, renumbered, split into a numeric value v and a level lv.
__rf_feat AS MATERIALIZED (
    SELECT r.i AS rid, s.col, f.kind,
           CASE WHEN f.typename = 'BOOLEAN' THEN CASE WHEN s.sval = 'true' THEN 1.0 ELSE 0.0 END
                WHEN f.kind = 'num' THEN TRY_CAST(s.sval AS DOUBLE) END AS v,
           CASE WHEN f.kind = 'cat' THEN s.sval END AS lv
    FROM __rf_slong s
    JOIN __rf_rows r ON r.rid = s.rid
    JOIN __rf_featcols f ON f.col = s.col
),
-- Bootstrap. Deterministic from md5_number only.
--   replace_sample: m = ceil(sample_frac*n) draws per tree, draw k picks row
--     (md5_number(seed:tree:k) mod n) + 1. The modulus MUST be UHUGEINT:
--     md5_number returns UHUGEINT, n is BIGINT, and DuckDB has no
--     UHUGEINT % BIGINT overload -- it silently casts BOTH to DOUBLE, throwing
--     away ~75 bits of the hash, and the "bootstrap" then draws from a few
--     dozen distinct rows for any n. Every tree would see the same rows and
--     bagging would do nothing.
--   otherwise: rank rows by md5_number(seed:tree:i) and take the first m.
-- The result is a WEIGHTED row set (tree, rid, cnt), never a materialized
-- resample. cnt is multiplied by the row's sample weight.
__rf_trees AS (SELECT unnest(range(1, n_trees + 1))::INTEGER AS tree),
__rf_m AS (SELECT greatest(1, ceil(sample_frac * (SELECT n FROM __rf_n)))::BIGINT AS m),
__rf_boot AS MATERIALIZED (
    SELECT t.tree, d.rid, count(*)::DOUBLE * any_value(rw.rw) AS w
    FROM __rf_trees t
    CROSS JOIN LATERAL (
        SELECT (md5_number(seed || ':' || t.tree || ':' || g.kk)
                % (SELECT n FROM __rf_n)::UHUGEINT)::BIGINT + 1 AS rid
        FROM range(1, (SELECT m FROM __rf_m) + 1) g(kk)
    ) d
    JOIN __rf_rw rw ON rw.rid = d.rid
    CROSS JOIN __rf_wchk wc
    WHERE replace_sample AND wc.ok
    GROUP BY t.tree, d.rid
    UNION ALL
    SELECT tree, rid, w FROM (
        SELECT t.tree, r.i AS rid, rw.rw AS w
        FROM __rf_trees t
        CROSS JOIN __rf_rows r
        JOIN __rf_rw rw ON rw.rid = r.i
        CROSS JOIN __rf_wchk wc
        WHERE NOT replace_sample AND wc.ok
        QUALIFY row_number() OVER (PARTITION BY t.tree
                                   ORDER BY md5_number(seed || ':' || t.tree || ':' || r.i))
                <= (SELECT m FROM __rf_m)
    )
),
-- Root weight per tree: sklearn's "improvement" (what min_impurity_decrease is
-- compared against) is imp_decrease / w_root, so it must be a per-tree constant.
__rf_wroot AS MATERIALIZED (
    SELECT tree, sum(w) AS w_root FROM __rf_boot GROUP BY tree
),
-- Order-dependent fingerprint of the training rows. *_oob_* must be handed the
-- exact table the model was trained on (row identity is the ordinal); with this
-- it can say so instead of returning plausible garbage.
__rf_hash AS (
    SELECT coalesce(sum((md5_number(rowkey) % 4611686018427387847::UHUGEINT)::BIGINT), 0)::HUGEINT AS h
    FROM (SELECT r.i || '|' || string_agg(s.sval, '|' ORDER BY s.col) AS rowkey
          FROM __rf_slong s JOIN __rf_rows r ON r.rid = s.rid
          GROUP BY r.i)
),

-- =========================================================================
-- The forest. ONE recursive CTE: every tree grows in the same recursion, one
-- LEVEL per iteration (breadth-first), so the iteration count is the depth and
-- not the number of nodes.
--
-- Tagged union: 'assign' rows are the frontier (which training row currently
-- sits in which node); 'split' and 'leaf' rows are the model. The recursive
-- term reads only WHERE tag = 'assign', so model rows accumulate in the result
-- and are never reprocessed, and the recursion halts by itself when a level
-- produces no new assignments. The model payload rides in ONE nullable STRUCT
-- so the frontier rows -- by far the most numerous -- stay narrow.
-- =========================================================================
__rf_tr AS (
    SELECT 'assign' AS tag, b.tree, 1::BIGINT AS node, 0::INTEGER AS depth,
           b.rid, b.w,
           NULL::STRUCT(split_feature VARCHAR, split_kind VARCHAR, threshold DOUBLE,
                        cats_left VARCHAR[], cats_right VARCHAR[], unseen_left BOOLEAN,
                        n_rows BIGINT, w_node DOUBLE, impurity DOUBLE, imp_decrease DOUBLE,
                        prediction DOUBLE, class_counts MAP(VARCHAR, DOUBLE)) AS m
    FROM __rf_boot b
  UNION ALL
    (
     WITH cur AS (SELECT tree, node, depth, rid, w FROM __rf_tr WHERE tag = 'assign'),
     -- Sparse slot sums, then densified against the full slot list: a class
     -- absent from a node must still occupy its position in the vector.
     nsl AS (
        SELECT c.tree, c.node, u.slot, sum(c.w * u.u) AS s
        FROM cur c JOIN __rf_u u ON u.rid = c.rid
        GROUP BY c.tree, c.node, u.slot
     ),
     ns AS (
        SELECT tree, node, any_value(depth) AS depth, count(*) AS nrows, sum(w) AS wn
        FROM cur GROUP BY tree, node
     ),
     nvec AS (
        SELECT g.tree, g.node, list(coalesce(x.s, 0.0) ORDER BY g.slot) AS pvec
        FROM (SELECT n.tree, n.node, k.slot FROM ns n CROSS JOIN __rf_slots k) g
        LEFT JOIN nsl x ON x.tree = g.tree AND x.node = g.node AND x.slot = g.slot
        GROUP BY g.tree, g.node
     ),
     nstat AS (
        SELECT ns.tree, ns.node, ns.depth, ns.nrows, ns.wn, v.pvec,
               __rf_imp(v.pvec, criterion) AS imp,
               __rf_q(v.pvec, criterion)   AS qpar
        FROM ns JOIN nvec v ON v.tree = ns.tree AND v.node = ns.node
     ),
     -- Splittable nodes. The purity test is on the NORMALIZED impurity against
     -- DBL_EPSILON, exactly as sklearn's (impurity <= EPSILON); comparing a
     -- weight-scaled quantity to an absolute constant would make tree depth
     -- depend on the units of y.
     sn AS (
        SELECT * FROM nstat
        WHERE depth < (SELECT depth_eff FROM __rf_cfg)
          AND nrows >= min_samples_split
          AND imp > 2.220446049250313e-16
     ),
     -- Features that actually vary inside the node. sklearn's max_features is a
     -- budget on NON-CONSTANT features (a constant feature does not consume a
     -- draw), so restricting the mtry lottery to varying features reproduces its
     -- effective budget and never wastes a draw on a feature that cannot split.
     nonconst AS (
        SELECT c.tree, c.node, f.col, f.kind
        FROM cur c
        JOIN sn s ON s.tree = c.tree AND s.node = c.node
        JOIN __rf_feat f ON f.rid = c.rid
        GROUP BY c.tree, c.node, f.col, f.kind
        HAVING coalesce(min(f.v) < max(f.v), false) OR coalesce(min(f.lv) < max(f.lv), false)
     ),
     mt AS (
        SELECT tree, node, col, kind
        FROM nonconst
        QUALIFY row_number() OVER (PARTITION BY tree, node
                                   ORDER BY md5_number(seed || ':' || tree || ':' || node || ':' || col))
                <= (SELECT mtry_eff FROM __rf_cfg)
     ),
     -- The node's rows restricted to the sampled features.
     cf AS (
        SELECT c.tree, c.node, c.rid, c.w, m.col, m.kind, f.v, f.lv
        FROM cur c
        JOIN mt m ON m.tree = c.tree AND m.node = c.node
        JOIN __rf_feat f ON f.rid = c.rid AND f.col = m.col
     ),
     -- Buckets: one per distinct numeric value, one per categorical level. Row
     -- counts and slot sums are aggregated separately because a classification
     -- row contributes to exactly one slot.
     bcnt AS (
        SELECT tree, node, col, kind,
               CASE WHEN kind = 'num' THEN CAST(v AS VARCHAR) ELSE lv END AS bucket,
               any_value(v) AS bnum, count(*) AS bn
        FROM cf
        GROUP BY tree, node, col, kind, bucket
     ),
     bslot AS (
        SELECT cf.tree, cf.node, cf.col,
               CASE WHEN cf.kind = 'num' THEN CAST(cf.v AS VARCHAR) ELSE cf.lv END AS bucket,
               u.slot, sum(cf.w * u.u) AS s
        FROM cf JOIN __rf_u u ON u.rid = cf.rid
        GROUP BY cf.tree, cf.node, cf.col, bucket, u.slot
     ),
     bvec AS (
        SELECT b.tree, b.node, b.col, b.kind, b.bucket, b.bnum, b.bn,
               list(coalesce(x.s, 0.0) ORDER BY g.slot) AS bv
        FROM (SELECT b2.tree, b2.node, b2.col, b2.kind, b2.bucket, b2.bnum, b2.bn, k.slot
              FROM bcnt b2 CROSS JOIN __rf_slots k) g
        JOIN bcnt b ON b.tree = g.tree AND b.node = g.node AND b.col = g.col AND b.bucket = g.bucket
        LEFT JOIN bslot x ON x.tree = g.tree AND x.node = g.node AND x.col = g.col
             AND x.bucket = g.bucket AND x.slot = g.slot
        GROUP BY b.tree, b.node, b.col, b.kind, b.bucket, b.bnum, b.bn
     ),
     -- One cumulative-sum machine for both feature kinds: give every bucket a
     -- sort key, sum the slot vectors in key order, and every prefix boundary is
     -- a candidate split (left = prefix, right = parent - prefix).
     --   numeric     -> ord 0, key = the value itself (exact CART)
     --   categorical -> regression: key = mean(y) in the level; the prefix split
     --                  is then provably the optimum over all subsets
     --                  (Fisher/Breiman).
     --                  classification: one ordering per class k, key =
     --                  P(y = k | level); exact for K = 2, a heuristic above.
     -- Sort-key ties break on the bucket name so the scan is deterministic.
     cands AS (
        SELECT tree, node, col, kind, bucket, bn, bv, 0::BIGINT AS ord, bnum AS skey
        FROM bvec WHERE kind = 'num'
        UNION ALL
        SELECT b.tree, b.node, b.col, b.kind, b.bucket, b.bn, b.bv, o.slot AS ord,
               CASE WHEN criterion = 'mse' THEN b.bv[2] / b.bv[1]
                    ELSE b.bv[o.slot] / list_sum(b.bv) END AS skey
        FROM bvec b
        CROSS JOIN __rf_slots o
        WHERE b.kind = 'cat'
          AND (criterion != 'mse' OR o.slot = 1)   -- regression has ONE ordering
     ),
     -- Slot dimension back to rows so the cumulative sums can be windowed.
     dense AS (
        SELECT c.tree, c.node, c.col, c.ord, c.bucket, c.skey, c.bn,
               k.slot, c.bv[k.slot] AS s
        FROM cands c CROSS JOIN __rf_slots k
     ),
     cum AS (
        SELECT tree, node, col, ord, bucket, skey, slot,
               sum(s)     OVER pw AS cs,
               sum(bn)    OVER pw AS cn,
               lead(skey) OVER pw AS nextkey
        FROM dense
        WINDOW pw AS (PARTITION BY tree, node, col, ord, slot ORDER BY skey, bucket
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
     ),
     pref AS (
        SELECT tree, node, col, ord, bucket, skey,
               any_value(cn) AS nl, any_value(nextkey) AS nextkey,
               list(cs ORDER BY slot) AS lvec
        FROM cum
        GROUP BY tree, node, col, ord, bucket, skey
     ),
     -- Left/right slot vectors of every candidate. They are materialized as
     -- COLUMNS before Q() touches them: __rf_q expands its argument inside a
     -- lambda body, and DuckDB cannot put a scalar subquery -- or another lambda
     -- -- there ("subqueries in lambda expressions are not supported").
     cvec AS (
        SELECT p.tree, p.node, p.col, p.ord, p.bucket, p.skey, p.nextkey, p.nl, p.lvec,
               list_transform(p.lvec, lambda x, j: s.pvec[j] - x) AS rvec,
               s.depth, s.nrows, s.wn, s.pvec, s.imp, s.qpar, f.kind, wr.w_root
        FROM pref p
        JOIN sn s ON s.tree = p.tree AND s.node = p.node
        JOIN __rf_featcols f ON f.col = p.col
        JOIN __rf_wroot wr ON wr.tree = p.tree
        WHERE p.nextkey IS NOT NULL              -- a prefix boundary needs a right side
          AND p.nl >= min_samples_leaf
          AND s.nrows - p.nl >= min_samples_leaf
     ),
     -- imp_decrease = Q(L) + Q(R) - Q(parent). sklearn accepts a split iff
     -- improvement + EPSILON >= min_impurity_decrease, i.e. it happily makes
     -- ZERO-GAIN splits on an impure node (XOR data needs exactly that). A
     -- "gain > 0" filter would truncate the tree and break CART equivalence.
     scored AS (
        SELECT c.*,
               __rf_q(c.lvec, criterion) + __rf_q(c.rvec, criterion) - c.qpar AS gain,
               __rf_wt(c.lvec, criterion) AS wl
        FROM cvec c
     ),
     best AS (
        SELECT * FROM scored
        WHERE isfinite(gain)
          AND gain / w_root + 2.220446049250313e-16 >= min_impurity_decrease
        QUALIFY row_number() OVER (PARTITION BY tree, node
                 ORDER BY gain DESC, col, ord, skey, bucket) = 1
     ),
     -- The threshold is computed ONCE here and reused by both the model row and
     -- the child assignment, so the scored partition and the realized partition
     -- can never drift apart. The midpoint of two adjacent doubles rounds UP to
     -- the right-hand value about half the time (and overflows to +inf for huge
     -- magnitudes); since descent is "left iff v <= threshold", that would send
     -- the entire right bucket LEFT, leaving the right child empty, the node
     -- unsplit in fact, and the model structurally invalid. sklearn clamps the
     -- same way.
     bestthr AS (
        SELECT b.*,
               CASE WHEN b.kind = 'num' THEN
                 CASE WHEN NOT isfinite((b.skey + b.nextkey) / 2.0)
                        OR (b.skey + b.nextkey) / 2.0 >= b.nextkey
                      THEN b.skey ELSE (b.skey + b.nextkey) / 2.0 END
               END AS thr
        FROM best b
     ),
     -- Both level lists: cats_left is the winning prefix, cats_right the rest of
     -- the levels PRESENT AT THIS NODE. A level in neither was never seen here
     -- (common deep in a tree, even with no genuinely new levels) and predict
     -- routes it by unseen_left; without cats_right the descent could not tell
     -- "seen, went right" from "never seen".
     bestdef AS (
        SELECT b.tree, b.node, b.depth, b.col, b.kind, b.thr, b.gain, b.nrows, b.wn,
               b.imp, b.pvec, b.wl, b.wn - b.wl AS wr,
               list(p.bucket ORDER BY p.bucket)
                 FILTER (b.kind = 'cat' AND (p.skey, p.bucket) <= (b.skey, b.bucket)) AS cats_left,
               list(p.bucket ORDER BY p.bucket)
                 FILTER (b.kind = 'cat' AND (p.skey, p.bucket) >  (b.skey, b.bucket)) AS cats_right
        FROM bestthr b
        LEFT JOIN pref p ON p.tree = b.tree AND p.node = b.node
                        AND p.col = b.col AND p.ord = b.ord
        GROUP BY b.tree, b.node, b.depth, b.col, b.kind, b.thr, b.gain, b.nrows, b.wn,
                 b.imp, b.pvec, b.wl, b.skey, b.bucket
     )
     -- internal (split) nodes
     SELECT 'split', d.tree, d.node, d.depth, NULL::BIGINT, NULL::DOUBLE,
            struct_pack(
              split_feature := d.col,
              split_kind    := d.kind,
              threshold     := d.thr,
              cats_left     := d.cats_left,
              cats_right    := d.cats_right,
              unseen_left   := CASE WHEN d.kind = 'cat' THEN d.wl >= d.wr END,
              n_rows        := d.nrows::BIGINT,
              w_node        := d.wn,
              impurity      := d.imp,
              imp_decrease  := d.gain,
              prediction    := NULL::DOUBLE,
              class_counts  := NULL::MAP(VARCHAR, DOUBLE))
     FROM bestdef d
     UNION ALL
     -- leaves: every current node with no admissible split. class_counts is
     -- DENSE over the training classes (zeros included) so that predict can
     -- average probability vectors across trees key by key; prediction adds the
     -- global mean back on (the forest was fit on the centered outcome).
     SELECT 'leaf', s.tree, s.node, s.depth, NULL::BIGINT, NULL::DOUBLE,
            struct_pack(
              split_feature := NULL::VARCHAR,
              split_kind    := NULL::VARCHAR,
              threshold     := NULL::DOUBLE,
              cats_left     := NULL::VARCHAR[],
              cats_right    := NULL::VARCHAR[],
              unseen_left   := NULL::BOOLEAN,
              n_rows        := s.nrows::BIGINT,
              w_node        := s.wn,
              impurity      := s.imp,
              imp_decrease  := NULL::DOUBLE,
              prediction    := CASE WHEN family = 'regression'
                                    THEN s.pvec[2] / s.pvec[1] + (SELECT ybar FROM __rf_ybar) END,
              class_counts  := CASE WHEN family = 'classification'
                                    THEN map_from_entries(list_transform(
                                           (SELECT classes FROM __rf_classlist),
                                           lambda c, j: struct_pack(key := c, value := s.pvec[j]))) END)
     FROM nstat s
     WHERE NOT EXISTS (SELECT 1 FROM bestdef d WHERE d.tree = s.tree AND d.node = s.node)
     UNION ALL
     -- the next frontier
     SELECT 'assign', c.tree,
            d.node * 2 + CASE WHEN d.kind = 'num'
                              THEN CASE WHEN f.v <= d.thr THEN 0 ELSE 1 END
                              ELSE CASE WHEN list_contains(d.cats_left, f.lv) THEN 0 ELSE 1 END
                         END,
            c.depth + 1, c.rid, c.w, NULL
     FROM cur c
     JOIN bestdef d ON d.tree = c.tree AND d.node = c.node
     JOIN __rf_feat f ON f.rid = c.rid AND f.col = d.col
    )
)
SELECT t.tree,
       t.node,
       t.depth,
       t.tag = 'leaf'          AS is_leaf,
       t.m.split_feature       AS split_feature,
       t.m.split_kind          AS split_kind,
       t.m.threshold           AS threshold,
       t.m.cats_left           AS cats_left,
       t.m.cats_right          AS cats_right,
       t.m.unseen_left         AS unseen_left,
       t.m.n_rows              AS n_rows,
       t.m.w_node              AS w_node,
       t.m.impurity            AS impurity,
       t.m.imp_decrease        AS imp_decrease,
       t.m.prediction          AS prediction,
       t.m.class_counts        AS class_counts,
       family                                       AS family,
       n_trees::INTEGER                             AS n_trees,
       seed::BIGINT                                 AS seed,
       sample_frac::DOUBLE                          AS sample_frac,
       replace_sample                               AS replace_sample,
       (SELECT n FROM __rf_n)                       AS n_train,
       (SELECT mtry_eff FROM __rf_cfg)::INTEGER     AS mtry,
       (SELECT depth_eff FROM __rf_cfg)::INTEGER    AS max_depth,
       min_samples_split::INTEGER                   AS min_samples_split,
       min_samples_leaf::INTEGER                    AS min_samples_leaf,
       min_impurity_decrease::DOUBLE                AS min_impurity_decrease,
       criterion                                    AS criterion,
       (SELECT list(col ORDER BY j) FROM __rf_featcols)   AS features,
       (SELECT list(kind ORDER BY j) FROM __rf_featcols)  AS feature_kinds,
       (SELECT classes FROM __rf_classlist)               AS classes,
       (SELECT h FROM __rf_hash)                          AS train_hash
FROM __rf_tr t
CROSS JOIN __rf_chk ck
WHERE t.tag IN ('split', 'leaf') AND ck.ok
ORDER BY t.tree, t.node;


-- ---------------------------------------------------------------------------
-- Public fit wrappers
-- ---------------------------------------------------------------------------

CREATE OR REPLACE MACRO rf_class_fit(tbl, outcome, n_trees := 100, mtry := NULL, max_depth := 20,
                                     min_samples_split := 2, min_samples_leaf := 1,
                                     min_impurity_decrease := 0.0, sample_frac := 1.0,
                                     replace_sample := true, criterion := 'gini', seed := 42,
                                     weights_col := NULL, class_weight := NULL) AS TABLE
SELECT * FROM __rf_fit(tbl, outcome, 'classification', 'rf_class_fit', n_trees, mtry, max_depth,
                       min_samples_split, min_samples_leaf, min_impurity_decrease,
                       sample_frac, replace_sample, criterion, seed, weights_col, class_weight);

CREATE OR REPLACE MACRO rf_reg_fit(tbl, outcome, n_trees := 100, mtry := NULL, max_depth := 20,
                                   min_samples_split := 2, min_samples_leaf := 1,
                                   min_impurity_decrease := 0.0, sample_frac := 1.0,
                                   replace_sample := true, criterion := 'mse', seed := 42,
                                   weights_col := NULL) AS TABLE
SELECT * FROM __rf_fit(tbl, outcome, 'regression', 'rf_reg_fit', n_trees, mtry, max_depth,
                       min_samples_split, min_samples_leaf, min_impurity_decrease,
                       sample_frac, replace_sample, criterion, seed, weights_col, NULL);


-- ###########################################################################
-- SCORING, EVALUATION, IMPORTANCE, OUT-OF-BAG, TUNING
--
-- Everything below reads a fitted model *table* (whatever *_fit returned, saved
-- with CREATE TABLE m AS SELECT * FROM rf_class_fit(...)) plus a data table, and
-- is passed BOTH by name, as strings, exactly like duckLM.
--
-- Public macros:
--   rf_class_predict(model, tbl, na_action := 'null', n_trees := NULL)
--   rf_reg_predict(model, tbl, na_action := 'null', n_trees := NULL)
--   rf_class_predict_trees(model, tbl, na_action := 'null', n_trees := NULL)
--   rf_reg_predict_trees(model, tbl, na_action := 'null', n_trees := NULL)
--   rf_class_evaluate(model, tbl, outcome, na_action := 'null', n_trees := NULL)
--   rf_reg_evaluate(model, tbl, outcome, na_action := 'null', n_trees := NULL)
--   rf_class_oob_predict(model, tbl) / rf_reg_oob_predict(model, tbl)
--   rf_class_oob(model, tbl, outcome) / rf_reg_oob(model, tbl, outcome)
--   rf_importance(model)
--   rf_permutation_importance(model, tbl, outcome, n_repeats := 5, seed := 42)
--   rf_summary(model)
--   rf_cv(tbl, outcome, family, mtry_grid, ...) / rf_cv_depth(...)
--
-- Internal helpers: __rf_walk, __rf_class_eval, __rf_reg_eval, __rf_cv.
-- ###########################################################################


-- ---------------------------------------------------------------------------
-- __rf_walk -- the shared scorer.
--
-- ONE recursive CTE walks EVERY tree of the model for EVERY scoring row at
-- once: seed (row, tree) at node 1, then at each iteration join the frontier to
-- the model's internal nodes and to the row's value of that node's split
-- feature and descend. The iteration count is the depth of the deepest tree,
-- not the number of nodes. A (row, tree) pair that lands on a leaf simply
-- produces no child and stays in the recursive table, so the final SELECT just
-- joins the frontier to the leaves.
--
-- Returns one row per (scoring row, tree) that reached a leaf:
--     rid          BIGINT   the scoring row's 1-based ordinal (row_number())
--     tree         INTEGER
--     node         BIGINT   the leaf it landed on
--     prediction   DOUBLE   leaf value        (regression models)
--     class_counts MAP(VARCHAR, DOUBLE)  dense leaf counts (classification)
--
-- Descent:
--   numeric      LEFT iff v <= threshold.
--   categorical  THREE-way: list_contains(cats_left, lv) -> left;
--                list_contains(cats_right, lv) -> right; otherwise the level was
--                not seen at this node in training (a genuinely new level, or a
--                level filtered out by an ancestor split / not drawn into this
--                tree's bootstrap -- the common case deep in a tree) and it
--                follows unseen_left, i.e. it goes to the heavier child.
--
-- NULL contract (na_action):
--   'null'       (default, and duckLM's contract) a scoring row with a NULL --
--                or an unparseable value, or a missing column -- in ANY model
--                feature is not seeded at all and every output for it is NULL.
--   'skip_tree'  the row is seeded; a tree ABSTAINS only if the descent actually
--                reaches a node that splits on a feature that is NULL for that
--                row (the JOIN to the cell finds nothing and the path dies). The
--                row's prediction is the average over the trees that did reach a
--                leaf, and is NULL only if every tree abstained. Trees test only
--                the handful of features on the row's path, so this recovers
--                most rows that 'null' throws away.
--
-- n_trees := NULL scores with every tree; n_trees := k scores with the first k,
-- which is how you plot error against forest size.
--
-- oob := true reconstructs each tree's bootstrap sample from the model metadata
-- (seed, n_trees, sample_frac, replace_sample, n_train -- the same
-- md5_number(seed || ':' || tree || ':' || k) % n draw the fit used) and seeds
-- ONLY the (row, tree) pairs where the row was OUT of that tree's bag. It
-- therefore requires the exact training table: row identity is the ordinal of
-- the complete rows. n_train and train_hash are recomputed from tbl and a
-- mismatch is an error -- a wrong table would otherwise return plausible,
-- silently meaningless numbers, which is the worst failure mode there is.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO __rf_walk(model, tbl, caller, family, na_action, n_trees, oob) AS TABLE
WITH RECURSIVE
-- Forest metadata. It is constant on every model row, so any_value() is the
-- idiom; this is the whole point of carrying it there.
__rf_meta AS MATERIALIZED (
    SELECT any_value(family)         AS family,
           any_value(features)       AS features,
           any_value(feature_kinds)  AS feature_kinds,
           any_value(classes)        AS classes,
           any_value(seed)           AS seed,
           any_value(sample_frac)    AS sample_frac,
           any_value(replace_sample) AS replace_sample,
           any_value(n_train)        AS n_train,
           any_value(train_hash)     AS train_hash,
           count(*)                  AS n_model_rows
    FROM query_table(model)
),
__rf_mfeat AS MATERIALIZED (
    SELECT unnest(features) AS col, unnest(feature_kinds) AS kind FROM __rf_meta
),
-- Column types of the SCORING table (same discovery trick as __rf_fit: DESCRIBE
-- does not bind query_table() inside a macro). Only used to recognise BOOLEAN,
-- whose VARCHAR form is 'true'/'false' and would TRY_CAST to NULL.
__rf_types AS MATERIALIZED (
    SELECT colname, typename
    FROM (SELECT *
          FROM (SELECT 1 AS __rf_one)
          LEFT JOIN (SELECT typeof(COLUMNS('^(.*)$')) AS '\1' FROM query_table(tbl) LIMIT 1) ON true)
         UNPIVOT INCLUDE NULLS (typename FOR colname IN (COLUMNS(* EXCLUDE (__rf_one))))
),
__rf_slong AS MATERIALIZED (
    SELECT __rf_rid__ AS rid, name AS col, value AS sval
    FROM (UNPIVOT (SELECT row_number() OVER () AS __rf_rid__, CAST(COLUMNS(*) AS VARCHAR)
                   FROM query_table(tbl))
          ON COLUMNS(* EXCLUDE (__rf_rid__)) INTO NAME name VALUE value)
),
-- Guards. As in __rf_fit, a guard only fires if its boolean is REFERENCED, so
-- every consumer below CROSS JOINs __rf_chk and puts ck.ok in its WHERE.
__rf_chk AS (
    SELECT CASE
             WHEN starts_with(lower(tbl), '__rf_') OR starts_with(lower(model), '__rf_')
               THEN error(caller || ': table names beginning with "__rf_" are reserved for internal use; please rename')
             WHEN (SELECT count(*) FROM __rf_types WHERE starts_with(lower(colname), '__rf_')) > 0
               THEN error(caller || ': column names beginning with "__rf_" are reserved for internal use; please rename')
             WHEN (SELECT n_model_rows FROM __rf_meta) = 0
               THEN error(caller || ': model table "' || model || '" is empty')
             WHEN (SELECT family FROM __rf_meta) != family
               THEN error(caller || ': model "' || model || '" is a ' || (SELECT family FROM __rf_meta)
                          || ' forest; use the rf_' || CASE WHEN (SELECT family FROM __rf_meta) = 'regression'
                                                            THEN 'reg' ELSE 'class' END || '_* macros for it')
             -- NOTE: no empty-table guard here. Unlike __rf_fit (where an empty
             -- table is a user error), an empty SCORING table must yield zero
             -- output rows, not an error -- predicting a filtered set that
             -- happens to be empty is routine, and this mirrors duckLM's predict
             -- contract. An empty table naturally produces zero frontier seeds
             -- and zero output rows; the OOB path additionally catches it via the
             -- n_train / train_hash mismatch in __rf_oobchk.
             -- Output-name collisions, case-insensitively (DuckDB identifiers
             -- are case-folded, so a "PREDICTION" column would still capture a
             -- downstream "SELECT prediction").
             WHEN caller IN ('rf_class_predict', 'rf_class_oob_predict')
                  AND (SELECT count(*) FROM __rf_types WHERE lower(colname) IN ('pred', 'probs')) > 0
               THEN error(caller || ': the input table already has a "pred" or "probs" column, which collides with the output columns; rename or drop it first (e.g. SELECT * EXCLUDE (pred, probs))')
             WHEN caller IN ('rf_reg_predict', 'rf_reg_oob_predict')
                  AND (SELECT count(*) FROM __rf_types WHERE lower(colname) = 'prediction') > 0
               THEN error(caller || ': the input table already has a "prediction" column, which collides with the output column; rename or drop it first (e.g. SELECT * EXCLUDE (prediction))')
             WHEN na_action NOT IN ('null', 'skip_tree')
               THEN error(caller || ': na_action must be ''null'' or ''skip_tree'', got ''' || na_action || '''')
             WHEN n_trees IS NOT NULL AND n_trees < 1
               THEN error(caller || ': n_trees must be >= 1 (or NULL for every tree), got ' || n_trees)
             ELSE true
           END AS ok
),
-- Scoring cells. The model's kind decides how a cell is read (numeric vs level),
-- the scoring table's type only decides the BOOLEAN special case. A column the
-- model wants but the scoring table does not have simply produces no cell, so
-- the row is incomplete -- which is the documented "missing feature column
-- yields NULL outputs" contract.
__rf_cells AS MATERIALIZED (
    SELECT s.rid, f.col, f.kind,
           CASE WHEN t.typename = 'BOOLEAN' THEN CASE WHEN s.sval = 'true' THEN 1.0 ELSE 0.0 END
                WHEN f.kind = 'num' THEN TRY_CAST(s.sval AS DOUBLE) END AS v,
           CASE WHEN f.kind = 'cat' THEN s.sval END AS lv
    FROM __rf_slong s
    JOIN __rf_mfeat f ON f.col = s.col
    JOIN __rf_types t ON t.colname = s.col
),
__rf_usable AS MATERIALIZED (
    SELECT * FROM __rf_cells
    WHERE (kind = 'num' AND v IS NOT NULL) OR (kind = 'cat' AND lv IS NOT NULL)
),
__rf_rowids AS (SELECT DISTINCT rid FROM __rf_slong),
__rf_full AS (
    SELECT rid FROM __rf_usable
    GROUP BY rid
    HAVING count(*) = (SELECT len(features) FROM __rf_meta)
),
-- The rows we are willing to seed: under 'null' only the fully-observed ones,
-- under 'skip_tree' every row (the descent itself decides where to stop).
__rf_start AS (
    SELECT r.rid
    FROM __rf_rowids r
    WHERE na_action = 'skip_tree' OR r.rid IN (SELECT rid FROM __rf_full)
),
__rf_nodes AS MATERIALIZED (
    SELECT * FROM query_table(model)
    WHERE tree <= coalesce(n_trees, 2147483647)
),
__rf_treelist AS (SELECT DISTINCT tree FROM __rf_nodes),
__rf_int AS MATERIALIZED (
    SELECT tree, node, split_feature, split_kind, threshold, cats_left, cats_right, unseen_left
    FROM __rf_nodes WHERE NOT is_leaf
),
__rf_leaf AS MATERIALIZED (
    SELECT tree, node, prediction, class_counts FROM __rf_nodes WHERE is_leaf
),

-- ---- out-of-bag membership, replayed from the model metadata ---------------
-- The training row identity is the ordinal of the COMPLETE rows (a row with a
-- NULL anywhere -- in any column, not just the features -- was dropped at fit),
-- so it has to be reconstructed the same way here.
__rf_allcomplete AS MATERIALIZED (
    SELECT rid FROM __rf_slong
    GROUP BY rid
    HAVING count(*) = (SELECT count(*) FROM __rf_types)
),
__rf_inum AS MATERIALIZED (
    SELECT rid, row_number() OVER (ORDER BY rid) AS i FROM __rf_allcomplete
),
__rf_hash AS (
    SELECT coalesce(sum((md5_number(rowkey) % 4611686018427387847::UHUGEINT)::BIGINT), 0)::HUGEINT AS h
    FROM (SELECT r.i || '|' || string_agg(s.sval, '|' ORDER BY s.col) AS rowkey
          FROM __rf_slong s JOIN __rf_inum r ON r.rid = s.rid
          GROUP BY r.i)
),
__rf_oobchk AS (
    SELECT CASE
             WHEN (SELECT count(*) FROM __rf_inum) != (SELECT n_train FROM __rf_meta)
               THEN error(caller || ': "' || tbl || '" has ' || (SELECT count(*) FROM __rf_inum)
                          || ' complete rows but the model was trained on ' || (SELECT n_train FROM __rf_meta)
                          || '; out-of-bag scoring requires the exact training table (row identity is the row ordinal)')
             WHEN (SELECT h FROM __rf_hash) != (SELECT train_hash FROM __rf_meta)
               THEN error(caller || ': "' || tbl || '" is not the table this model was trained on (row fingerprint mismatch); out-of-bag scoring requires the exact training table, unfiltered and in the same order')
             ELSE true
           END AS ok
),
-- The fit's bootstrap draw, replayed bit for bit. The modulus MUST be UHUGEINT
-- (md5_number returns UHUGEINT; a BIGINT modulus silently casts both sides to
-- DOUBLE and throws away most of the hash).
__rf_m AS (
    SELECT greatest(1, ceil((SELECT sample_frac FROM __rf_meta) * (SELECT n_train FROM __rf_meta)))::BIGINT AS m
),
__rf_boot AS (
    SELECT t.tree, d.i
    FROM __rf_treelist t
    CROSS JOIN LATERAL (
        SELECT (md5_number((SELECT seed FROM __rf_meta) || ':' || t.tree || ':' || g.kk)
                % (SELECT n_train FROM __rf_meta)::UHUGEINT)::BIGINT + 1 AS i
        FROM range(1, (SELECT m FROM __rf_m) + 1) g(kk)
    ) d
    WHERE oob AND (SELECT replace_sample FROM __rf_meta)
    UNION ALL
    SELECT tree, i FROM (
        SELECT t.tree, r.i
        FROM __rf_treelist t
        CROSS JOIN __rf_inum r
        WHERE oob AND NOT (SELECT replace_sample FROM __rf_meta)
        QUALIFY row_number() OVER (PARTITION BY t.tree
                                   ORDER BY md5_number((SELECT seed FROM __rf_meta) || ':' || t.tree || ':' || r.i))
                <= (SELECT m FROM __rf_m)
    )
),
__rf_oobpairs AS (
    SELECT t.tree, r.rid
    FROM __rf_treelist t
    CROSS JOIN __rf_inum r
    ANTI JOIN __rf_boot b ON b.tree = t.tree AND b.i = r.i
),

-- ---- the descent ----------------------------------------------------------
__rf_w AS (
    SELECT s.rid, t.tree, 1::BIGINT AS node
    FROM __rf_start s
    CROSS JOIN __rf_treelist t
    CROSS JOIN __rf_chk ck
    WHERE ck.ok AND NOT oob
  UNION ALL
    SELECT p.rid, p.tree, 1::BIGINT AS node
    FROM __rf_oobpairs p
    CROSS JOIN __rf_chk ck
    CROSS JOIN __rf_oobchk oc
    WHERE oob AND ck.ok AND oc.ok
  UNION ALL
    SELECT w.rid, w.tree,
           w.node * 2 + CASE WHEN i.split_kind = 'num'
                             THEN CASE WHEN c.v <= i.threshold THEN 0 ELSE 1 END
                             ELSE CASE WHEN list_contains(i.cats_left,  c.lv) THEN 0
                                       WHEN list_contains(i.cats_right, c.lv) THEN 1
                                       WHEN i.unseen_left                     THEN 0
                                       ELSE 1 END
                        END AS node
    FROM __rf_w w
    JOIN __rf_int i ON i.tree = w.tree AND i.node = w.node
    -- 'skip_tree' lives here: a row with no usable cell for THIS node's split
    -- feature finds no join partner, its path dies, and that tree abstains for
    -- that row. Under 'null' such rows were never seeded in the first place.
    JOIN __rf_usable c ON c.rid = w.rid AND c.col = i.split_feature
)
SELECT w.rid, w.tree, w.node, l.prediction, l.class_counts
FROM __rf_w w
JOIN __rf_leaf l ON l.tree = w.tree AND l.node = w.node;


-- ---------------------------------------------------------------------------
-- Predict (soft voting for classification).
--
-- rf_reg_predict:   input rows + prediction DOUBLE = mean of the trees' leaf
--                   predictions.
-- rf_class_predict: input rows + pred VARCHAR + probs MAP(VARCHAR, DOUBLE).
--   Each tree votes a NORMALIZED class distribution (its leaf counts / leaf
--   weight); the forest probability is the mean of those vectors across the
--   trees that scored the row (soft voting, as sklearn does -- not majority
--   vote). probs is DENSE over every training class. pred is the argmax, ties
--   broken to the SMALLEST label (matching duckLM's alphabetical reference
--   class and sklearn's stable argmax over sorted classes).
--
-- A row no tree scored (all NULL features under 'null', or every tree abstained
-- under 'skip_tree') gets NULL outputs.
-- ---------------------------------------------------------------------------
-- Family guard as a scalar macro so it fires even when the walk produces no rows
-- (scoring a regression model with rf_class_* leaves class_counts all-NULL, so the
-- walk output is empty and the walk's own guard would be optimized away). It
-- aggregates over the always-present model table, so it is always evaluated.
CREATE OR REPLACE MACRO __rf_famchk(model, caller, fam) AS (
    (SELECT CASE
              WHEN count(*) = 0 THEN error(caller || ': model table "' || model || '" is empty')
              WHEN any_value(family) != fam
                THEN error(caller || ': model "' || model || '" is a ' || any_value(family)
                           || ' forest; use the rf_' || CASE WHEN any_value(family) = 'regression'
                                                             THEN 'reg' ELSE 'class' END || '_* macros for it')
              ELSE true END
     FROM query_table(model))
);

CREATE OR REPLACE MACRO rf_reg_predict(model, tbl, na_action := 'null', n_trees := NULL) AS TABLE
SELECT n.* EXCLUDE (__rf_rid__), p.prediction
FROM (SELECT row_number() OVER () AS __rf_rid__, * FROM query_table(tbl)) n
CROSS JOIN (SELECT __rf_famchk(model, 'rf_reg_predict', 'regression') AS ok) g
LEFT JOIN (
    SELECT rid, avg(prediction) AS prediction
    FROM __rf_walk(model, tbl, 'rf_reg_predict', 'regression', na_action, n_trees, false)
    GROUP BY rid
) p ON p.rid = n.__rf_rid__
WHERE g.ok
ORDER BY n.__rf_rid__;

CREATE OR REPLACE MACRO rf_class_predict(model, tbl, na_action := 'null', n_trees := NULL) AS TABLE
WITH __rf_cls AS (SELECT any_value(classes) AS classes FROM query_table(model)),
-- Per (row, class): mean over the scoring trees of that tree's normalized count.
__rf_prob AS (
    SELECT rid, cls, avg(cnt / wsum) AS p
    FROM (
        SELECT rid,
               unnest(map_keys(class_counts))   AS cls,
               unnest(map_values(class_counts)) AS cnt,
               list_sum(map_values(class_counts)) AS wsum
        FROM __rf_walk(model, tbl, 'rf_class_predict', 'classification', na_action, n_trees, false)
    )
    GROUP BY rid, cls
),
__rf_agg AS (
    SELECT rid,
           map_from_entries(list(struct_pack(key := cls, value := p) ORDER BY cls)) AS probs,
           (list(cls ORDER BY p DESC, cls))[1]                                       AS pred
    FROM __rf_prob
    GROUP BY rid
)
SELECT n.* EXCLUDE (__rf_rid__), a.pred, a.probs
FROM (SELECT row_number() OVER () AS __rf_rid__, * FROM query_table(tbl)) n
CROSS JOIN (SELECT __rf_famchk(model, 'rf_class_predict', 'classification') AS ok) g
LEFT JOIN __rf_agg a ON a.rid = n.__rf_rid__
WHERE g.ok
ORDER BY n.__rf_rid__;


-- ---------------------------------------------------------------------------
-- Per-tree predictions: the ensemble aggregation removed. One row per
-- (scoring row, tree). Buys prediction intervals (spread across trees),
-- quantile-regression-forest style summaries, ensemble-variance diagnostics
-- and learning curves for free.
--   rf_reg_predict_trees   -> __rf_rid__, tree, prediction
--   rf_class_predict_trees -> __rf_rid__, tree, pred, probs (that tree's leaf)
-- __rf_rid__ is the scoring row's 1-based ordinal, matching *_predict's order.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO rf_reg_predict_trees(model, tbl, na_action := 'null', n_trees := NULL) AS TABLE
SELECT rid AS __rf_rid__, tree, prediction
FROM __rf_walk(model, tbl, 'rf_reg_predict_trees', 'regression', na_action, n_trees, false)
CROSS JOIN (SELECT __rf_famchk(model, 'rf_reg_predict_trees', 'regression') AS ok) g
WHERE g.ok
ORDER BY rid, tree;

CREATE OR REPLACE MACRO rf_class_predict_trees(model, tbl, na_action := 'null', n_trees := NULL) AS TABLE
SELECT rid AS __rf_rid__, tree,
       (list(cls ORDER BY p DESC, cls))[1] AS pred,
       map_from_entries(list(struct_pack(key := cls, value := p) ORDER BY cls)) AS probs
FROM (
    SELECT rid, tree,
           unnest(map_keys(class_counts))   AS cls,
           unnest(map_values(class_counts)) / list_sum(map_values(class_counts)) AS p
    FROM __rf_walk(model, tbl, 'rf_class_predict_trees', 'classification', na_action, n_trees, false)
    CROSS JOIN (SELECT __rf_famchk(model, 'rf_class_predict_trees', 'classification') AS ok) g
    WHERE g.ok
)
GROUP BY rid, tree
ORDER BY rid, tree;


-- ---------------------------------------------------------------------------
-- Evaluate.
--
-- Rows whose outcome is NULL, or that no tree scored, are dropped (n counts
-- what was actually evaluated), exactly as duckLM's *_evaluate.
--
-- Regression (__rf_reg_eval) -> n, rmse, mae, r2
--   rmse = sqrt(mean (y - yhat)^2),  mae = mean |y - yhat|,
--   r2   = 1 - SSE / SST,  SST = sum (y - mean(y))^2  (sklearn r2_score).
--
-- Classification (__rf_class_eval) -> n, accuracy, log_loss, brier, auc
--   probs is the DENSE forest distribution; pred is its argmax (ties -> smallest
--   label). accuracy = mean[pred = y].
--   log_loss = -mean ln(clip(p_y, 1e-15, 1-1e-15))  -- sklearn clips to the
--     float64 machine epsilon and does NOT renormalize (matched to 1e-9).
--   brier = mean sum_k (1[y=k] - p_k)^2, HALVED for binary (sklearn's
--     scale_by_half='auto' halves when < 3 classes, i.e. binary).
--   auc: binary only, else NULL. Positive class = the lexicographically GREATER
--     label; computed as the Mann-Whitney statistic on average ranks of the
--     positive-class probability. NULL if a class is absent from y.
-- The oob flag routes __rf_walk through the out-of-bag membership; the public
-- rf_*_oob wrappers set it and also report n_excluded (training rows in-bag for
-- every tree, hence unscored).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO __rf_reg_eval(model, tbl, outcome, caller, na_action, n_trees, oob) AS TABLE
WITH
__rf_pred AS (
    SELECT rid, avg(prediction) AS yhat
    FROM __rf_walk(model, tbl, caller, 'regression', na_action, n_trees, oob)
    GROUP BY rid
),
-- The outcome column, pulled out via the same VARCHAR long-form as __rf_walk.
-- This BINDS even when `outcome` is not a column of tbl (a COLUMNS(lambda c: c =
-- outcome) expansion would instead binder-fail at plan time with "empty set of
-- columns", short-circuiting the clean "no rows ..." guard below). A missing
-- column just yields no rows here, so __rf_rows is empty and __rf_ck fires.
__rf_truth AS (
    SELECT rid, TRY_CAST(sval AS DOUBLE) AS y
    FROM (SELECT __rf_rid__ AS rid, name AS col, value AS sval
          FROM (UNPIVOT (SELECT row_number() OVER () AS __rf_rid__, CAST(COLUMNS(*) AS VARCHAR)
                         FROM query_table(tbl))
                ON COLUMNS(* EXCLUDE (__rf_rid__)) INTO NAME name VALUE value))
    WHERE col = outcome
),
__rf_rows AS (
    SELECT t.y, p.yhat
    FROM __rf_truth t JOIN __rf_pred p ON p.rid = t.rid
    WHERE t.y IS NOT NULL AND p.yhat IS NOT NULL
),
__rf_ck AS (
    SELECT CASE WHEN (SELECT __rf_famchk(model, caller, 'regression')) IS NULL THEN false
                WHEN (SELECT count(*) FROM __rf_rows) = 0
                THEN error(caller || ': no rows with a non-NULL prediction and outcome to evaluate')
                ELSE true END AS ok
),
__rf_agg AS (
    SELECT count(*)::DOUBLE AS n, avg(y) AS ybar,
           sum((y - yhat) * (y - yhat)) AS sse, sum(abs(y - yhat)) AS sae
    FROM __rf_rows
),
__rf_sst AS (SELECT sum((y - a.ybar) * (y - a.ybar)) AS sst FROM __rf_rows r, __rf_agg a)
SELECT a.n::BIGINT AS n, sqrt(a.sse / a.n) AS rmse, a.sae / a.n AS mae,
       1.0 - a.sse / s.sst AS r2
FROM __rf_agg a, __rf_sst s, __rf_ck ck
WHERE ck.ok;

CREATE OR REPLACE MACRO __rf_class_eval(model, tbl, outcome, caller, na_action, n_trees, oob) AS TABLE
WITH
__rf_cls AS (SELECT any_value(classes) AS classes FROM query_table(model)),
__rf_prob AS (
    SELECT rid, cls, avg(cnt / wsum) AS p
    FROM (SELECT rid,
                 unnest(map_keys(class_counts))   AS cls,
                 unnest(map_values(class_counts)) AS cnt,
                 list_sum(map_values(class_counts)) AS wsum
          FROM __rf_walk(model, tbl, caller, 'classification', na_action, n_trees, oob))
    GROUP BY rid, cls
),
-- The outcome column via the VARCHAR long-form (see __rf_reg_eval): binds even
-- when `outcome` is absent, so the clean "no rows ..." guard fires instead of a
-- cryptic COLUMNS "empty set of columns" binder error.
__rf_truth AS (
    SELECT rid, sval AS y
    FROM (SELECT __rf_rid__ AS rid, name AS col, value AS sval
          FROM (UNPIVOT (SELECT row_number() OVER () AS __rf_rid__, CAST(COLUMNS(*) AS VARCHAR)
                         FROM query_table(tbl))
                ON COLUMNS(* EXCLUDE (__rf_rid__)) INTO NAME name VALUE value))
    WHERE col = outcome
),
-- Per scored row: argmax pred, the full brier term, and the true-class prob.
__rf_perrow AS (
    SELECT pr.rid, t.y AS y,
           (list(pr.cls ORDER BY pr.p DESC, pr.cls))[1] AS pred,
           sum(((pr.cls = t.y)::INT - pr.p) * ((pr.cls = t.y)::INT - pr.p)) AS brier_row,
           sum(pr.p) FILTER (pr.cls = t.y) AS p_true,
           sum(pr.p) FILTER (pr.cls = (SELECT list_max(classes) FROM __rf_cls)) AS p_pos
    FROM __rf_prob pr JOIN __rf_truth t ON t.rid = pr.rid
    WHERE t.y IS NOT NULL
    GROUP BY pr.rid, t.y
),
__rf_ck AS (
    SELECT CASE WHEN (SELECT __rf_famchk(model, caller, 'classification')) IS NULL THEN false
                WHEN (SELECT count(*) FROM __rf_perrow) = 0
                THEN error(caller || ': no rows with a non-NULL prediction and outcome to evaluate')
                ELSE true END AS ok
),
-- AUC (binary only): Mann-Whitney U on average ranks of the positive-class prob.
__rf_pos AS (SELECT (SELECT list_max(classes) FROM __rf_cls) AS poscls,
                    (SELECT len(classes) FROM __rf_cls) AS nclasses),
__rf_ranked AS (
    SELECT (y = (SELECT poscls FROM __rf_pos))::INT AS ypos, avg(rn) OVER (PARTITION BY p_pos) AS rk
    FROM (SELECT y, p_pos, row_number() OVER (ORDER BY p_pos) AS rn FROM __rf_perrow)
),
__rf_auc AS (
    SELECT CASE WHEN (SELECT nclasses FROM __rf_pos) != 2 OR sum(ypos) = 0 OR sum(ypos) = count(*)
                THEN NULL
                ELSE (sum(CASE WHEN ypos = 1 THEN rk END) - sum(ypos) * (sum(ypos) + 1) / 2.0)
                     / (sum(ypos) * (count(*) - sum(ypos))) END AS auc
    FROM __rf_ranked
),
-- Metrics as a single-row aggregate (one row even over an empty __rf_perrow):
-- this makes the FROM below always non-empty, so __rf_ck.ok is evaluated and the
-- clean "no rows ..." guard fires instead of the query silently returning n = 0.
__rf_metrics AS (
    SELECT count(*)::BIGINT AS n,
           avg((pred = y)::INT) AS accuracy,
           -avg(ln(least(greatest(p_true, 1e-15), 1.0 - 1e-15))) AS log_loss,
           avg(brier_row) * CASE WHEN (SELECT nclasses FROM __rf_pos) < 3 THEN 0.5 ELSE 1.0 END AS brier
    FROM __rf_perrow
)
SELECT mx.n, mx.accuracy, mx.log_loss, mx.brier, (SELECT auc FROM __rf_auc) AS auc
FROM __rf_metrics mx, __rf_ck ck
WHERE ck.ok;


CREATE OR REPLACE MACRO rf_reg_evaluate(model, tbl, outcome, na_action := 'null', n_trees := NULL) AS TABLE
SELECT * FROM __rf_reg_eval(model, tbl, outcome, 'rf_reg_evaluate', na_action, n_trees, false);

CREATE OR REPLACE MACRO rf_class_evaluate(model, tbl, outcome, na_action := 'null', n_trees := NULL) AS TABLE
SELECT * FROM __rf_class_eval(model, tbl, outcome, 'rf_class_evaluate', na_action, n_trees, false);


-- ---------------------------------------------------------------------------
-- MDI feature importance, matching sklearn's feature_importances_ exactly.
--
-- Per tree: raw_j = sum over the tree's split nodes on feature j of
-- imp_decrease / w_root (w_root is the tree's ROOT weight -- imp_decrease is a
-- weight-scaled quantity, and dividing by w_root turns it into sklearn's
-- "improvement"). Normalize each tree's raw vector to sum 1; a tree with NO
-- splits (a stump) contributes the ZERO vector, not NULL and not a skip.
-- Average the normalized vectors over ALL n_trees, then RENORMALIZE that
-- average to sum 1 -- this last step is exactly what sklearn does (it averages
-- only over non-stump trees, which is algebraically the same as averaging the
-- zero-padded vectors over all trees and renormalizing). If every tree is a
-- stump, return all zeros rather than dividing by zero. Every feature appears,
-- 0 for never-used ones; ordered by importance descending then name.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO rf_importance(model) AS TABLE
WITH
__rf_meta AS (SELECT any_value(features) AS features, any_value(n_trees) AS n_trees FROM query_table(model)),
__rf_feats AS (SELECT unnest(features) AS feature FROM __rf_meta),
-- w_root per tree = the root node's weight (node = 1).
__rf_wroot AS (SELECT tree, w_node AS w_root FROM query_table(model) WHERE node = 1),
-- Raw MDI per (tree, feature).
__rf_raw AS (
    SELECT m.tree, m.split_feature AS feature, sum(m.imp_decrease / r.w_root) AS raw
    FROM query_table(model) m JOIN __rf_wroot r ON r.tree = m.tree
    WHERE NOT m.is_leaf
    GROUP BY m.tree, m.split_feature
),
__rf_treesum AS (SELECT tree, sum(raw) AS tot FROM __rf_raw GROUP BY tree),
-- Normalized per tree; only non-stump trees have any rows here. Summing these
-- over the grid of (tree, feature) and dividing by n_trees is the mean of the
-- zero-padded vectors.
__rf_norm AS (
    SELECT r.feature, sum(r.raw / s.tot) AS contrib
    FROM __rf_raw r JOIN __rf_treesum s ON s.tree = r.tree AND s.tot > 0
    GROUP BY r.feature
),
__rf_mean AS (
    SELECT f.feature, coalesce(n.contrib, 0.0) / (SELECT n_trees FROM __rf_meta) AS mean_imp
    FROM __rf_feats f LEFT JOIN __rf_norm n ON n.feature = f.feature
),
__rf_tot AS (SELECT sum(mean_imp) AS g FROM __rf_mean)
SELECT m.feature,
       CASE WHEN t.g > 0 THEN m.mean_imp / t.g ELSE 0.0 END AS importance
FROM __rf_mean m, __rf_tot t
ORDER BY importance DESC, feature;


-- ---------------------------------------------------------------------------
-- Out-of-bag prediction. Score each TRAINING row using only the trees that did
-- NOT draw it into their bootstrap sample. `tbl` MUST be the exact training
-- table -- unfiltered, same order (row identity is the ordinal of the complete
-- rows); __rf_walk recomputes n_train and the row fingerprint and errors on a
-- mismatch. A row in-bag for every tree gets NULL outputs.
--   rf_reg_oob_predict   -> input rows + prediction DOUBLE
--   rf_class_oob_predict -> input rows + pred VARCHAR + probs MAP
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO rf_reg_oob_predict(model, tbl) AS TABLE
SELECT n.* EXCLUDE (__rf_rid__), p.prediction
FROM (SELECT row_number() OVER () AS __rf_rid__, * FROM query_table(tbl)) n
CROSS JOIN (SELECT __rf_famchk(model, 'rf_reg_oob_predict', 'regression') AS ok) g
LEFT JOIN (
    SELECT rid, avg(prediction) AS prediction
    FROM __rf_walk(model, tbl, 'rf_reg_oob_predict', 'regression', 'null', NULL, true)
    GROUP BY rid
) p ON p.rid = n.__rf_rid__
WHERE g.ok
ORDER BY n.__rf_rid__;

CREATE OR REPLACE MACRO rf_class_oob_predict(model, tbl) AS TABLE
WITH __rf_prob AS (
    SELECT rid, cls, avg(cnt / wsum) AS p
    FROM (SELECT rid,
                 unnest(map_keys(class_counts))   AS cls,
                 unnest(map_values(class_counts)) AS cnt,
                 list_sum(map_values(class_counts)) AS wsum
          FROM __rf_walk(model, tbl, 'rf_class_oob_predict', 'classification', 'null', NULL, true))
    GROUP BY rid, cls
),
__rf_agg AS (
    SELECT rid,
           map_from_entries(list(struct_pack(key := cls, value := p) ORDER BY cls)) AS probs,
           (list(cls ORDER BY p DESC, cls))[1] AS pred
    FROM __rf_prob GROUP BY rid
)
SELECT n.* EXCLUDE (__rf_rid__), a.pred, a.probs
FROM (SELECT row_number() OVER () AS __rf_rid__, * FROM query_table(tbl)) n
CROSS JOIN (SELECT __rf_famchk(model, 'rf_class_oob_predict', 'classification') AS ok) g
LEFT JOIN __rf_agg a ON a.rid = n.__rf_rid__
WHERE g.ok
ORDER BY n.__rf_rid__;


-- ---------------------------------------------------------------------------
-- Out-of-bag evaluation: the *_evaluate metric row, computed on out-of-bag
-- predictions, plus n_excluded = training rows that were in-bag for EVERY tree
-- and so have no OOB prediction. `tbl` must be the training table.
--   rf_reg_oob   -> n, rmse, mae, r2, n_excluded
--   rf_class_oob -> n, accuracy, log_loss, brier, auc, n_excluded
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO rf_reg_oob(model, tbl, outcome) AS TABLE
SELECT e.*,
       ((SELECT any_value(n_train) FROM query_table(model)) - e.n)::BIGINT AS n_excluded
FROM __rf_reg_eval(model, tbl, outcome, 'rf_reg_oob', 'null', NULL, true) e;

CREATE OR REPLACE MACRO rf_class_oob(model, tbl, outcome) AS TABLE
SELECT e.*,
       ((SELECT any_value(n_train) FROM query_table(model)) - e.n)::BIGINT AS n_excluded
FROM __rf_class_eval(model, tbl, outcome, 'rf_class_oob', 'null', NULL, true) e;


-- ---------------------------------------------------------------------------
-- rf_summary(model): one row describing how the forest was fit and what grew.
-- The hyperparameters are read from the model metadata; the structure is a pure
-- aggregate over the node rows. depth_cap_hit is TRUE iff some tree was actually
-- truncated by max_depth -- i.e. there is a leaf at depth = max_depth whose
-- impurity is still > 0 (an impure leaf that only stopped because it hit the
-- cap). Without it a user has no way to know max_depth := 20 stunted the forest.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO rf_summary(model) AS TABLE
SELECT any_value(family)                                   AS family,
       any_value(n_trees)                                  AS n_trees,
       count(*)                                            AS n_nodes,
       count(*) FILTER (is_leaf)                           AS n_leaves,
       max(depth)                                          AS max_depth_reached,
       avg(depth) FILTER (is_leaf)                         AS mean_leaf_depth,
       bool_or(is_leaf AND depth = (SELECT any_value(max_depth) FROM query_table(model))
               AND impurity > 2.220446049250313e-16) AS depth_cap_hit,
       any_value(len(features))                            AS n_features,
       any_value(mtry)                                     AS mtry,
       any_value(max_depth)                                AS max_depth,
       any_value(criterion)                                AS criterion,
       any_value(seed)                                     AS seed,
       any_value(n_train)                                  AS n_train,
       any_value(sample_frac)                              AS sample_frac,
       any_value(replace_sample)                           AS replace_sample
FROM query_table(model);


-- ---------------------------------------------------------------------------
-- Cross-validation for tuning.
--
-- rf_cv(tbl, outcome, family, mtry_grid, k := 5, n_trees := 100, max_depth := 20,
--       min_samples_leaf := 1, sample_frac := 1.0, seed := 42)
--   family in 'classification' | 'regression' (word-style, matching duckLM's
--   cv_l2 vocabulary). mtry_grid is INTEGER[]. Returns (mtry, cv_error), one row
--   per grid value; SMALLER cv_error is better. cv_error is the misclassification
--   rate (classification) or MSE (regression) over held-out rows, k-fold.
--
-- Folds are assigned deterministically as (row# - 1) % k over the complete rows
-- (duckLM's convention). The rows are NOT shuffled -- if the table is ordered by
-- the outcome, shuffle it before calling (e.g. fit on a table selected
-- ORDER BY md5_number(...)).
--
-- rf_cv_depth(tbl, outcome, family, depth_grid, mtry := NULL, k := 5, ...)
--   the same, sweeping max_depth instead of mtry; returns (max_depth, cv_error).
--
-- Because a table macro cannot be re-invoked per (grid value, fold), the whole
-- sweep is ONE recursive forest over the (grid value x fold x tree) space: group
-- g = (grid_index - 1) * k + held_fold trains its trees on the rows whose fold
-- is NOT held_fold, with that group's mtry (or max_depth); a second recursion
-- then descends the held-out rows through their group's trees. Bagging is the
-- same bootstrap as *_fit (with replacement, sample_frac of the group's training
-- rows). Criterion is 'gini' (classification) / 'mse' (regression).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO __rf_cv(tbl, outcome, family, grid, sweep, k, n_trees, mtry_fixed,
                                max_depth_fixed, min_samples_leaf, sample_frac, seed) AS TABLE
WITH RECURSIVE
__cv_types AS MATERIALIZED (
    SELECT colname, typename,
           CASE WHEN typename IN ('BOOLEAN','TINYINT','SMALLINT','INTEGER','BIGINT','HUGEINT',
                                  'UTINYINT','USMALLINT','UINTEGER','UBIGINT','UHUGEINT','FLOAT','DOUBLE')
                     OR starts_with(typename, 'DECIMAL') THEN 'num'
                WHEN typename = 'VARCHAR' OR starts_with(typename, 'ENUM') THEN 'cat' END AS kind
    FROM (SELECT * FROM (SELECT 1 AS __rf_one)
          LEFT JOIN (SELECT typeof(COLUMNS('^(.*)$')) AS '\1' FROM query_table(tbl) LIMIT 1) ON true)
         UNPIVOT INCLUDE NULLS (typename FOR colname IN (COLUMNS(* EXCLUDE (__rf_one))))
),
__cv_slong AS MATERIALIZED (
    SELECT __rf_rid__ AS rid, name AS col, value AS sval
    FROM (UNPIVOT (SELECT row_number() OVER () AS __rf_rid__, CAST(COLUMNS(*) AS VARCHAR) FROM query_table(tbl))
          ON COLUMNS(* EXCLUDE (__rf_rid__)) INTO NAME name VALUE value)
),
__cv_featcols AS MATERIALIZED (
    SELECT colname AS col, kind FROM __cv_types WHERE colname != outcome
),
__cv_d AS (SELECT count(*)::BIGINT AS d FROM __cv_featcols),
__cv_complete AS MATERIALIZED (
    SELECT rid FROM __cv_slong GROUP BY rid HAVING count(*) = (SELECT count(*) FROM __cv_types)
),
__cv_chk AS (
    SELECT CASE
             WHEN family NOT IN ('classification', 'regression')
               THEN error('rf_cv: family must be ''classification'' or ''regression'', got ''' || family || '''')
             WHEN k < 2 THEN error('rf_cv: k must be >= 2, got ' || k)
             WHEN len(grid) < 1 THEN error('rf_cv: grid must be non-empty')
             WHEN sweep = 'mtry' AND list_aggregate(grid, 'min') < 1
               THEN error('rf_cv: every mtry must be >= 1')
             WHEN sweep = 'mtry' AND list_aggregate(grid, 'max') > (SELECT d FROM __cv_d)
               THEN error('rf_cv: mtry must not exceed the number of features (' || (SELECT d FROM __cv_d) || ')')
             WHEN sweep = 'depth' AND list_aggregate(grid, 'min') < 1
               THEN error('rf_cv: every max_depth must be >= 1')
             WHEN n_trees < 1 THEN error('rf_cv: n_trees must be >= 1')
             WHEN (SELECT count(*) FROM __cv_complete) = 0 THEN error('rf_cv: no complete rows')
             ELSE true END AS ok
),
__cv_rows AS MATERIALIZED (
    SELECT c.rid, row_number() OVER (ORDER BY c.rid) AS i,
           (row_number() OVER (ORDER BY c.rid) - 1) % k AS fold
    FROM __cv_complete c CROSS JOIN __cv_chk ck WHERE ck.ok
),
__cv_ysval AS MATERIALIZED (
    SELECT r.i AS rid, s.sval AS cls,
           TRY_CAST(s.sval AS DOUBLE) AS yv
    FROM __cv_slong s JOIN __cv_rows r ON r.rid = s.rid WHERE s.col = outcome
),
__cv_ybar AS (SELECT CASE WHEN family = 'regression' THEN avg(yv) ELSE 0.0 END AS ybar FROM __cv_ysval),
__cv_y AS MATERIALIZED (
    SELECT rid, cls, yv - (SELECT ybar FROM __cv_ybar) AS yv FROM __cv_ysval
),
__cv_classes AS MATERIALIZED (
    SELECT cls, row_number() OVER (ORDER BY cls) AS kk
    FROM (SELECT DISTINCT cls FROM __cv_ysval) WHERE family = 'classification'
),
__cv_classlist AS (SELECT list(cls ORDER BY kk) AS classes FROM __cv_classes),
__cv_slots AS MATERIALIZED (
    SELECT kk AS slot FROM __cv_classes
    UNION ALL SELECT unnest([1,2,3]) WHERE family = 'regression'
),
__cv_crit AS (SELECT CASE WHEN family = 'classification' THEN 'gini' ELSE 'mse' END AS crit),
__cv_u AS MATERIALIZED (
    SELECT y.rid, c.kk AS slot, 1.0::DOUBLE AS u
    FROM __cv_y y JOIN __cv_classes c ON c.cls = y.cls
    UNION ALL
    SELECT y.rid, s.slot, CASE s.slot WHEN 1 THEN 1.0 WHEN 2 THEN y.yv ELSE y.yv*y.yv END
    FROM __cv_y y CROSS JOIN (SELECT unnest([1,2,3]) AS slot) s WHERE family = 'regression'
),
__cv_feat AS MATERIALIZED (
    SELECT r.i AS rid, s.col, f.kind,
           CASE WHEN f.kind = 'num' AND (SELECT typename FROM __cv_types t WHERE t.colname = s.col) = 'BOOLEAN'
                     THEN CASE WHEN s.sval = 'true' THEN 1.0 ELSE 0.0 END
                WHEN f.kind = 'num' THEN TRY_CAST(s.sval AS DOUBLE) END AS v,
           CASE WHEN f.kind = 'cat' THEN s.sval END AS lv
    FROM __cv_slong s JOIN __cv_rows r ON r.rid = s.rid JOIN __cv_featcols f ON f.col = s.col
),
-- Groups: g = (grid_index-1)*k + held_fold. mtry / max_depth per group.
__cv_groups AS MATERIALIZED (
    SELECT (gi.g - 1) * k + hf.hf AS g, gi.g AS gidx, hf.hf AS held,
           CASE WHEN sweep = 'mtry' THEN grid[gi.g]
                ELSE coalesce(mtry_fixed, CASE WHEN family = 'classification'
                                               THEN greatest(1, floor(sqrt((SELECT d FROM __cv_d))))
                                               ELSE (SELECT d FROM __cv_d) END) END::BIGINT AS mtry_g,
           CASE WHEN sweep = 'depth' THEN grid[gi.g] ELSE coalesce(max_depth_fixed, 60) END::BIGINT AS depth_g
    FROM range(1, len(grid)+1) gi(g) CROSS JOIN range(0, k) hf(hf)
),
-- Training rows within each group, renumbered 1..m_g for the bootstrap draw.
__cv_train AS MATERIALIZED (
    SELECT g.g, r.rid AS i, row_number() OVER (PARTITION BY g.g ORDER BY r.i) AS j
    FROM __cv_groups g JOIN __cv_rows r ON r.fold != g.held
),
__cv_mg AS MATERIALIZED (SELECT g, count(*)::BIGINT AS mg FROM __cv_train GROUP BY g),
__cv_trees AS (SELECT unnest(range(1, n_trees+1))::INTEGER AS tree),
__cv_boot AS MATERIALIZED (
    SELECT b.g, b.tree, tr.i AS rid, count(*)::DOUBLE AS w
    FROM (
        SELECT g.g, t.tree, d.j
        FROM __cv_groups g CROSS JOIN __cv_trees t
        CROSS JOIN LATERAL (
            SELECT (md5_number(seed || ':cv:' || g.g || ':' || t.tree || ':' || kk.kk)
                    % (SELECT mg FROM __cv_mg m WHERE m.g = g.g)::UHUGEINT)::BIGINT + 1 AS j
            FROM range(1, greatest(1, ceil(sample_frac * (SELECT mg FROM __cv_mg m WHERE m.g = g.g)))::BIGINT + 1) kk(kk)
        ) d
    ) b
    JOIN __cv_train tr ON tr.g = b.g AND tr.j = b.j
    GROUP BY b.g, b.tree, tr.i
),
__cv_wroot AS MATERIALIZED (SELECT g, tree, sum(w) AS w_root FROM __cv_boot GROUP BY g, tree),

-- ===== build all trees for all groups, breadth-first =====
__cv_tr AS (
    SELECT 'assign' AS tag, b.g, b.tree, 1::BIGINT AS node, 0::INTEGER AS depth, b.rid, b.w,
           NULL::STRUCT(col VARCHAR, kind VARCHAR, thr DOUBLE, cats_left VARCHAR[],
                        unseen_left BOOLEAN, pred DOUBLE, cc MAP(VARCHAR, DOUBLE)) AS m
    FROM __cv_boot b
  UNION ALL
    (
     WITH cur AS (SELECT g, tree, node, depth, rid, w FROM __cv_tr WHERE tag = 'assign'),
     nsl AS (SELECT c.g, c.tree, c.node, u.slot, sum(c.w*u.u) AS s
             FROM cur c JOIN __cv_u u ON u.rid = c.rid GROUP BY c.g, c.tree, c.node, u.slot),
     ns AS (SELECT g, tree, node, any_value(depth) AS depth, count(*) AS nrows, sum(w) AS wn
            FROM cur GROUP BY g, tree, node),
     nvec AS (SELECT gg.g, gg.tree, gg.node, list(coalesce(x.s,0.0) ORDER BY gg.slot) AS pvec
              FROM (SELECT n.g, n.tree, n.node, s.slot FROM ns n CROSS JOIN __cv_slots s) gg
              LEFT JOIN nsl x ON x.g=gg.g AND x.tree=gg.tree AND x.node=gg.node AND x.slot=gg.slot
              GROUP BY gg.g, gg.tree, gg.node),
     nstat AS (SELECT ns.*, v.pvec, __rf_imp(v.pvec,(SELECT crit FROM __cv_crit)) AS imp,
                      __rf_q(v.pvec,(SELECT crit FROM __cv_crit)) AS qpar
               FROM ns JOIN nvec v ON v.g=ns.g AND v.tree=ns.tree AND v.node=ns.node),
     sn AS (SELECT s.*, gr.mtry_g, gr.depth_g FROM nstat s JOIN __cv_groups gr ON gr.g = s.g
            WHERE s.depth < gr.depth_g AND s.nrows >= 2 AND s.imp > 2.220446049250313e-16),
     nonconst AS (SELECT c.g, c.tree, c.node, f.col, f.kind
                  FROM cur c JOIN sn s ON s.g=c.g AND s.tree=c.tree AND s.node=c.node
                  JOIN __cv_feat f ON f.rid = c.rid
                  GROUP BY c.g, c.tree, c.node, f.col, f.kind
                  HAVING coalesce(min(f.v)<max(f.v),false) OR coalesce(min(f.lv)<max(f.lv),false)),
     mt AS (SELECT nc.g, nc.tree, nc.node, nc.col, nc.kind FROM nonconst nc
            JOIN sn s ON s.g=nc.g AND s.tree=nc.tree AND s.node=nc.node
            QUALIFY row_number() OVER (PARTITION BY nc.g, nc.tree, nc.node
                     ORDER BY md5_number(seed||':cv:'||nc.g||':'||nc.tree||':'||nc.node||':'||nc.col)) <= s.mtry_g),
     cf AS (SELECT c.g, c.tree, c.node, c.rid, c.w, m.col, m.kind, f.v, f.lv
            FROM cur c JOIN mt m ON m.g=c.g AND m.tree=c.tree AND m.node=c.node
            JOIN __cv_feat f ON f.rid=c.rid AND f.col=m.col),
     bcnt AS (SELECT g, tree, node, col, kind,
                     CASE WHEN kind='num' THEN CAST(v AS VARCHAR) ELSE lv END AS bucket,
                     any_value(v) AS bnum, count(*) AS bn
              FROM cf GROUP BY g, tree, node, col, kind, bucket),
     bslot AS (SELECT cf.g, cf.tree, cf.node, cf.col,
                      CASE WHEN cf.kind='num' THEN CAST(cf.v AS VARCHAR) ELSE cf.lv END AS bucket,
                      u.slot, sum(cf.w*u.u) AS s
               FROM cf JOIN __cv_u u ON u.rid=cf.rid
               GROUP BY cf.g, cf.tree, cf.node, cf.col, bucket, u.slot),
     bvec AS (SELECT b.g, b.tree, b.node, b.col, b.kind, b.bucket, b.bnum, b.bn,
                     list(coalesce(x.s,0.0) ORDER BY gg.slot) AS bv
              FROM (SELECT b2.g,b2.tree,b2.node,b2.col,b2.kind,b2.bucket,b2.bnum,b2.bn,s.slot
                    FROM bcnt b2 CROSS JOIN __cv_slots s) gg
              JOIN bcnt b ON b.g=gg.g AND b.tree=gg.tree AND b.node=gg.node AND b.col=gg.col AND b.bucket=gg.bucket
              LEFT JOIN bslot x ON x.g=gg.g AND x.tree=gg.tree AND x.node=gg.node AND x.col=gg.col
                   AND x.bucket=gg.bucket AND x.slot=gg.slot
              GROUP BY b.g,b.tree,b.node,b.col,b.kind,b.bucket,b.bnum,b.bn),
     cands AS (SELECT g,tree,node,col,kind,bucket,bn,bv,0::BIGINT AS ord, bnum AS skey
               FROM bvec WHERE kind='num'
               UNION ALL
               SELECT b.g,b.tree,b.node,b.col,b.kind,b.bucket,b.bn,b.bv,o.slot AS ord,
                      CASE WHEN (SELECT crit FROM __cv_crit)='mse' THEN b.bv[2]/b.bv[1]
                           ELSE b.bv[o.slot]/list_sum(b.bv) END AS skey
               FROM bvec b CROSS JOIN __cv_slots o
               WHERE b.kind='cat' AND ((SELECT crit FROM __cv_crit)!='mse' OR o.slot=1)),
     dense AS (SELECT c.g,c.tree,c.node,c.col,c.kind,c.ord,c.bucket,c.skey,c.bn,s.slot,c.bv[s.slot] AS s
               FROM cands c CROSS JOIN __cv_slots s),
     cum AS (SELECT g,tree,node,col,kind,ord,bucket,skey,slot,
                    sum(s) OVER pw AS cs, sum(bn) OVER pw AS cn, lead(skey) OVER pw AS nextkey
             FROM dense
             WINDOW pw AS (PARTITION BY g,tree,node,col,ord,slot ORDER BY skey,bucket
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)),
     pref AS (SELECT g,tree,node,col,ord,bucket,skey, any_value(kind) AS kind, any_value(cn) AS nl,
                     any_value(nextkey) AS nextkey, list(cs ORDER BY slot) AS lvec
              FROM cum GROUP BY g,tree,node,col,ord,bucket,skey),
     cvec AS (SELECT p.*, list_transform(p.lvec, lambda x,jj: s.pvec[jj]-x) AS rvec,
                     __rf_wt(p.lvec,(SELECT crit FROM __cv_crit)) AS wl,
                     s.depth AS depth, s.pvec, s.qpar, s.wn AS wn, wr.w_root
              FROM pref p JOIN sn s ON s.g=p.g AND s.tree=p.tree AND s.node=p.node
              JOIN __cv_wroot wr ON wr.g=p.g AND wr.tree=p.tree
              WHERE p.nextkey IS NOT NULL AND p.nl >= min_samples_leaf
                AND s.nrows - p.nl >= min_samples_leaf),
     scored AS (SELECT c.*, __rf_q(c.lvec,(SELECT crit FROM __cv_crit))
                            + __rf_q(c.rvec,(SELECT crit FROM __cv_crit)) - c.qpar AS gain
                FROM cvec c),
     best AS (SELECT * FROM scored WHERE isfinite(gain) AND gain/w_root + 2.220446049250313e-16 >= 0.0
              QUALIFY row_number() OVER (PARTITION BY g,tree,node ORDER BY gain DESC, col, ord, skey, bucket)=1),
     bthr AS (SELECT b.*, CASE WHEN b.kind='num' THEN
                            CASE WHEN NOT isfinite((b.skey+b.nextkey)/2.0) OR (b.skey+b.nextkey)/2.0 >= b.nextkey
                                 THEN b.skey ELSE (b.skey+b.nextkey)/2.0 END END AS thr FROM best b),
     bdef AS (SELECT b.g,b.tree,b.node,b.depth,b.col,b.kind,b.thr, b.wn, b.wl,
                     list(p.bucket) FILTER (b.kind='cat' AND (p.skey,p.bucket)<=(b.skey,b.bucket)) AS cats_left
              FROM bthr b LEFT JOIN pref p ON p.g=b.g AND p.tree=b.tree AND p.node=b.node AND p.col=b.col AND p.ord=b.ord
              GROUP BY b.g,b.tree,b.node,b.depth,b.col,b.kind,b.thr,b.wn,b.wl,b.skey,b.bucket)
     SELECT 'split', d.g, d.tree, d.node, d.depth, NULL::BIGINT, NULL::DOUBLE,
            struct_pack(col:=d.col, kind:=d.kind, thr:=d.thr, cats_left:=d.cats_left,
                        unseen_left:=CASE WHEN d.kind='cat' THEN d.wl >= d.wn - d.wl END,
                        pred:=NULL::DOUBLE, cc:=NULL::MAP(VARCHAR,DOUBLE))
     FROM bdef d
     UNION ALL
     SELECT 'leaf', s.g, s.tree, s.node, s.depth, NULL::BIGINT, NULL::DOUBLE,
            struct_pack(col:=NULL::VARCHAR, kind:=NULL::VARCHAR, thr:=NULL::DOUBLE,
                        cats_left:=NULL::VARCHAR[], unseen_left:=NULL::BOOLEAN,
                        pred:=CASE WHEN family='regression' THEN s.pvec[2]/s.pvec[1]+(SELECT ybar FROM __cv_ybar) END,
                        cc:=CASE WHEN family='classification' THEN map_from_entries(list_transform(
                              (SELECT classes FROM __cv_classlist), lambda c,jj: struct_pack(key:=c, value:=s.pvec[jj]))) END)
     FROM nstat s
     WHERE NOT EXISTS (SELECT 1 FROM best b WHERE b.g=s.g AND b.tree=s.tree AND b.node=s.node)
     UNION ALL
     SELECT 'assign', c.g, c.tree,
            d.node*2 + CASE WHEN d.kind='num' THEN CASE WHEN f.v<=d.thr THEN 0 ELSE 1 END
                            ELSE CASE WHEN list_contains(d.cats_left,f.lv) THEN 0 ELSE 1 END END,
            c.depth+1, c.rid, c.w, NULL
     FROM cur c JOIN bdef d ON d.g=c.g AND d.tree=c.tree AND d.node=c.node
     JOIN __cv_feat f ON f.rid=c.rid AND f.col=d.col
    )
),
__cv_model AS MATERIALIZED (
    SELECT g, tree, node, tag='leaf' AS is_leaf, m.col AS split_feature, m.kind AS split_kind,
           m.thr AS threshold, m.cats_left, m.unseen_left, m.pred AS prediction, m.cc AS class_counts
    FROM __cv_tr WHERE tag IN ('split','leaf')
),
__cv_int AS MATERIALIZED (SELECT * FROM __cv_model WHERE NOT is_leaf),
__cv_leaf AS MATERIALIZED (SELECT g, tree, node, prediction, class_counts FROM __cv_model WHERE is_leaf),
-- ===== score held-out rows through their group's trees =====
__cv_sc AS (
    SELECT gr.gidx, gr.g, t.tree, r.rid, 1::BIGINT AS node
    FROM __cv_rows r JOIN __cv_groups gr ON gr.held = r.fold CROSS JOIN __cv_trees t
  UNION ALL
    SELECT s.gidx, s.g, s.tree, s.rid,
           s.node*2 + CASE WHEN i.split_kind='num' THEN CASE WHEN f.v<=i.threshold THEN 0 ELSE 1 END
                           ELSE CASE WHEN list_contains(i.cats_left,f.lv) THEN 0
                                     WHEN i.unseen_left THEN 0 ELSE 1 END END
    FROM __cv_sc s JOIN __cv_int i ON i.g=s.g AND i.tree=s.tree AND i.node=s.node
    JOIN __cv_feat f ON f.rid=s.rid AND f.col=i.split_feature
),
__cv_landed AS (
    SELECT s.gidx, s.rid, s.tree, l.prediction, l.class_counts
    FROM __cv_sc s JOIN __cv_leaf l ON l.g=s.g AND l.tree=s.tree AND l.node=s.node
),
-- forest prediction per (grid value, row)
__cv_regpred AS (
    SELECT gidx, rid, avg(prediction) AS yhat FROM __cv_landed GROUP BY gidx, rid
),
__cv_clsprob AS (
    SELECT gidx, rid, cls, avg(cnt / wsum) AS p
    FROM (SELECT gidx, rid,
                 unnest(map_keys(class_counts))   AS cls,
                 unnest(map_values(class_counts)) AS cnt,
                 list_sum(map_values(class_counts)) AS wsum
          FROM __cv_landed)
    GROUP BY gidx, rid, cls
),
__cv_clspred AS (
    SELECT gidx, rid, (list(cls ORDER BY p DESC, cls))[1] AS pred FROM __cv_clsprob GROUP BY gidx, rid
),
__cv_err AS (
    SELECT p.gidx, avg((p.pred != y.cls)::INT) AS err
    FROM __cv_clspred p JOIN __cv_ysval y ON y.rid = p.rid WHERE family='classification'
    GROUP BY p.gidx
    UNION ALL
    SELECT p.gidx, avg((y.yv - p.yhat)*(y.yv - p.yhat)) AS err
    FROM __cv_regpred p JOIN __cv_ysval y ON y.rid = p.rid WHERE family='regression'
    GROUP BY p.gidx
),
-- Family / outcome guards that DO NOT depend on any family-gated CTE. __cv_chk
-- (k / grid / n_trees / complete-rows) is forced only via __cv_rows, which feeds
-- the recursion -> __cv_err; but __cv_err's two branches are each gated on
-- family = one of the two valid literals, so an ILLEGAL family (or a nonexistent
-- outcome, which makes __cv_ysval and hence __cv_err empty) constant-folds the
-- whole result to empty, the optimizer prunes __cv_rows, and __cv_chk's error()
-- never runs -- the query silently returns zero rows. These two checks live in a
-- single-row guard CTE that DRIVES the final SELECT (LEFT JOIN __cv_err), so it
-- is always evaluated regardless of whether __cv_err is empty.
__cv_guard AS (
    SELECT CASE
             WHEN family NOT IN ('classification', 'regression')
               THEN error('rf_cv: family must be ''classification'' or ''regression'', got ''' || family || '''')
             WHEN (SELECT count(*) FROM __cv_types WHERE colname = outcome) = 0
               THEN error('rf_cv: outcome column "' || outcome || '" not found in "' || tbl || '"')
             ELSE true
           END AS ok
)
SELECT grid[e.gidx] AS param, e.err AS cv_error
FROM __cv_guard g
LEFT JOIN __cv_err e ON true
WHERE g.ok AND e.gidx IS NOT NULL
ORDER BY e.gidx;

CREATE OR REPLACE MACRO rf_cv(tbl, outcome, family, mtry_grid, k := 5, n_trees := 100,
                              max_depth := 20, min_samples_leaf := 1, sample_frac := 1.0, seed := 42) AS TABLE
SELECT param AS mtry, cv_error
FROM __rf_cv(tbl, outcome, family, mtry_grid, 'mtry', k, n_trees, NULL, max_depth,
             min_samples_leaf, sample_frac, seed);

CREATE OR REPLACE MACRO rf_cv_depth(tbl, outcome, family, depth_grid, mtry := NULL, k := 5,
                                    n_trees := 100, min_samples_leaf := 1, sample_frac := 1.0, seed := 42) AS TABLE
SELECT param AS max_depth, cv_error
FROM __rf_cv(tbl, outcome, family, depth_grid, 'depth', k, n_trees, mtry, NULL,
             min_samples_leaf, sample_frac, seed);


-- ---------------------------------------------------------------------------
-- Permutation feature importance, matching sklearn.inspection.permutation_importance.
--
-- The honest, cardinality-UNBIASED complement to MDI (rf_importance): instead of
-- crediting a feature for the impurity its splits removed at FIT time (which
-- inflates high-cardinality features -- and worse here, native many-level
-- categoricals -- because they simply get more chances to split), it measures how
-- much the model's SCORE on `tbl` degrades when that feature's column is randomly
-- shuffled. A feature the forest genuinely relies on loses score when broken; a
-- noise feature does not, so its importance sits at ~0 (and CAN go slightly
-- negative -- that sign is real and is kept).
--
--   rf_permutation_importance(model, tbl, outcome, n_repeats := 5, seed := 42)
--       -> feature VARCHAR, importance DOUBLE, importance_std DOUBLE
--   ordered by importance DESC, feature (like rf_importance).
--
-- ONE macro serves both families: `family` is read from the model metadata and
-- the "score" is the estimator's default .score() -- R^2 for a regression forest,
-- accuracy for a classification forest -- computed with the model's NORMAL
-- prediction (all trees, soft voting for classification), on exactly the rows
-- rf_*_evaluate would score: rows with every model feature present AND a non-NULL
-- (finite, for regression) outcome. Permutation reshuffles values WITHIN that
-- scored set, so completeness is preserved.
--
-- Definition (sklearn-exact): for feature j and repeat r, permute column j across
-- the scored rows, re-score -> s_{j,r}; per-repeat importance = baseline - s_{j,r}.
-- Reported importance = mean_r (baseline - s_{j,r}); importance_std = population
-- std (np.std, ddof=0 -> stddev_pop) of those per-repeat values. Since baseline is
-- constant, stddev_pop(baseline - s) = stddev_pop(s).
--
-- RNG NOTE: the permutation is md5-seeded (this library's only randomness source),
-- NOT numpy's, so importances match sklearn STATISTICALLY (rank/top-k), not to
-- 1e-9 -- exactly like every other forest-level check here. Deterministic given
-- `seed` under PRAGMA threads=1.
--
-- Implementation: __rf_walk reads features from query_table(tbl) BY NAME and cannot
-- be handed a permuted table, so this is a self-contained recursive descent that
-- carries a JOB dimension. A job is either the baseline (pf = NULL) or one
-- (feature j, repeat r) with pf = j, rep = r. Every frontier row carries (pf, rep);
-- at a node splitting on feature c the effective cell is the base cell EXCEPT when
-- c = pf, where the permuted cell is used (a LEFT JOIN whose match requires
-- w.pf = split_feature; baseline's pf = NULL never matches -> base values). The
-- descent uses the SAME branch expression as __rf_walk.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO rf_permutation_importance(model, tbl, outcome,
                                                  n_repeats := 5, seed := 42) AS TABLE
WITH RECURSIVE
-- Forest metadata, constant on every model row (any_value idiom, as __rf_walk).
__rf_meta AS MATERIALIZED (
    SELECT any_value(family)        AS family,
           any_value(features)      AS features,
           any_value(feature_kinds) AS feature_kinds,
           any_value(classes)       AS classes,
           count(*)                 AS n_model_rows
    FROM query_table(model)
),
__rf_mfeat AS MATERIALIZED (
    SELECT unnest(features) AS col, unnest(feature_kinds) AS kind FROM __rf_meta
),
-- Column types of the SCORING table (same discovery trick as __rf_walk: DESCRIBE
-- does not bind query_table() inside a macro). Only used to recognise BOOLEAN,
-- whose VARCHAR form is 'true'/'false' and would TRY_CAST to NULL.
__rf_types AS MATERIALIZED (
    SELECT colname, typename
    FROM (SELECT *
          FROM (SELECT 1 AS __rf_one)
          LEFT JOIN (SELECT typeof(COLUMNS('^(.*)$')) AS '\1' FROM query_table(tbl) LIMIT 1) ON true)
         UNPIVOT INCLUDE NULLS (typename FOR colname IN (COLUMNS(* EXCLUDE (__rf_one))))
),
-- ONE VARCHAR long-form of the whole scoring table (features AND the outcome).
__rf_slong AS MATERIALIZED (
    SELECT __rf_rid__ AS rid, name AS col, value AS sval
    FROM (UNPIVOT (SELECT row_number() OVER () AS __rf_rid__, CAST(COLUMNS(*) AS VARCHAR)
                   FROM query_table(tbl))
          ON COLUMNS(* EXCLUDE (__rf_rid__)) INTO NAME name VALUE value)
),
-- Guards. As elsewhere, a guard only fires if its boolean is REFERENCED, so the
-- scored-set CTE forces it (upstream) and the final SELECT drives off it (so it
-- still fires when the scored set is empty, e.g. a missing outcome column). The
-- outcome is pulled from the tolerant VARCHAR long-form, never COLUMNS(lambda),
-- so a missing column hits this guard rather than a binder error.
__rf_chk AS (
    SELECT CASE
             WHEN starts_with(lower(tbl), '__rf_') OR starts_with(lower(model), '__rf_')
               THEN error('rf_permutation_importance: table names beginning with "__rf_" are reserved for internal use; please rename')
             WHEN (SELECT count(*) FROM __rf_types WHERE starts_with(lower(colname), '__rf_')) > 0
               THEN error('rf_permutation_importance: column names beginning with "__rf_" are reserved for internal use; please rename')
             WHEN (SELECT n_model_rows FROM __rf_meta) = 0
               THEN error('rf_permutation_importance: model table "' || model || '" is empty')
             WHEN n_repeats < 1
               THEN error('rf_permutation_importance: n_repeats must be >= 1, got ' || n_repeats)
             WHEN (SELECT count(*) FROM __rf_types WHERE colname = outcome) = 0
               THEN error('rf_permutation_importance: outcome column "' || outcome || '" not found in "' || tbl || '"')
             ELSE true
           END AS ok
),
-- Scoring cells (as __rf_walk): the model's kind decides num vs level; the scoring
-- table's type only decides the BOOLEAN special case.
__rf_cells AS MATERIALIZED (
    SELECT s.rid, f.col, f.kind,
           CASE WHEN t.typename = 'BOOLEAN' THEN CASE WHEN s.sval = 'true' THEN 1.0 ELSE 0.0 END
                WHEN f.kind = 'num' THEN TRY_CAST(s.sval AS DOUBLE) END AS v,
           CASE WHEN f.kind = 'cat' THEN s.sval END AS lv
    FROM __rf_slong s
    JOIN __rf_mfeat f ON f.col = s.col
    JOIN __rf_types t ON t.colname = s.col
),
__rf_usable AS MATERIALIZED (
    SELECT * FROM __rf_cells
    WHERE (kind = 'num' AND v IS NOT NULL) OR (kind = 'cat' AND lv IS NOT NULL)
),
-- Rows with every model feature usable (the 'null' na_action scored set).
__rf_full AS (
    SELECT rid FROM __rf_usable
    GROUP BY rid
    HAVING count(*) = (SELECT len(features) FROM __rf_meta)
),
-- The outcome column via the tolerant long-form: ys = the raw label (VARCHAR,
-- comparable to the model's classes, which rf_class_fit stored as VARCHAR), yv =
-- the numeric value for regression. A missing outcome yields no rows here.
__rf_truth AS (
    SELECT rid, sval AS ys, TRY_CAST(sval AS DOUBLE) AS yv
    FROM __rf_slong WHERE col = outcome
),
-- The scored set: complete feature rows with a valid outcome, numbered i = 1..n
-- by row_number() OVER (ORDER BY rid). Everything downstream keys off i. This is
-- the upstream point that FORCES __rf_chk (CROSS JOIN ... WHERE ck.ok).
__rf_scored AS MATERIALIZED (
    SELECT f.rid, row_number() OVER (ORDER BY f.rid) AS i
    FROM __rf_full f
    JOIN __rf_truth tr ON tr.rid = f.rid
    CROSS JOIN __rf_chk ck
    WHERE ck.ok
      AND ((SELECT family FROM __rf_meta) = 'classification' OR isfinite(tr.yv))
),
__rf_srows AS (SELECT i FROM __rf_scored),
-- Base feature cells over the scored set, keyed by the scored ordinal i.
__rf_base AS MATERIALIZED (
    SELECT sc.i, u.col, u.v, u.lv
    FROM __rf_usable u JOIN __rf_scored sc ON sc.rid = u.rid
),
-- Outcome per scored ordinal (uncentered y for R^2, label for accuracy).
__rf_yset AS MATERIALIZED (
    SELECT sc.i AS rid, tr.ys, tr.yv
    FROM __rf_scored sc JOIN __rf_truth tr ON tr.rid = sc.rid
),
-- SST for R^2 (constant across jobs): ybar = mean y over the scored set.
__rf_ybar AS (SELECT avg(yv) AS ybar FROM __rf_yset),
__rf_sst AS (SELECT sum((yv - (SELECT ybar FROM __rf_ybar)) * (yv - (SELECT ybar FROM __rf_ybar))) AS sst
             FROM __rf_yset),
-- Jobs: one per (feature, repeat). Baseline is added to the job set below.
__rf_jobs AS (
    SELECT f.col AS pf, r.rep::INTEGER AS rep
    FROM __rf_mfeat f
    CROSS JOIN (SELECT unnest(range(1, n_repeats + 1)) AS rep) r
),
__rf_jobset AS (
    SELECT NULL::VARCHAR AS pf, NULL::INTEGER AS rep
    UNION ALL
    SELECT pf, rep FROM __rf_jobs
),
-- The permutation, "shuffle by sorting on a random key" (fixed points are fine,
-- as numpy's permutation has them too). For each (feature, repeat): identity
-- position ipos = i; shuffle position spos ranks i by md5_number(seed:P:...). The
-- row at identity position p receives feature pf's value from the row at shuffle
-- position p. The md5 key ties break on i so the order is fully deterministic.
__rf_perm AS (
    SELECT j.pf, j.rep, s.i,
           row_number() OVER (PARTITION BY j.pf, j.rep ORDER BY s.i) AS ipos,
           row_number() OVER (PARTITION BY j.pf, j.rep
                              ORDER BY md5_number(seed || ':P:' || j.pf || ':' || j.rep || ':' || s.i), s.i) AS spos
    FROM __rf_jobs j CROSS JOIN __rf_srows s
),
__rf_permmap AS (
    SELECT a.pf, a.rep, a.i AS dest_i, b.i AS src_i
    FROM __rf_perm a JOIN __rf_perm b ON a.pf = b.pf AND a.rep = b.rep AND a.ipos = b.spos
),
-- The permuted cell of feature pf at destination row i = pf's base cell at the
-- shuffle partner row. Exactly d * n_repeats * n rows.
__rf_permcell AS MATERIALIZED (
    SELECT pm.pf, pm.rep, pm.dest_i AS i, base.v, base.lv
    FROM __rf_permmap pm
    JOIN __rf_base base ON base.i = pm.src_i AND base.col = pm.pf
),
-- Model nodes (all trees; permutation importance is NOT out-of-bag).
__rf_nodes AS MATERIALIZED (SELECT * FROM query_table(model)),
__rf_treelist AS (SELECT DISTINCT tree FROM __rf_nodes),
__rf_int AS MATERIALIZED (
    SELECT tree, node, split_feature, split_kind, threshold, cats_left, cats_right, unseen_left
    FROM __rf_nodes WHERE NOT is_leaf
),
__rf_leaf AS MATERIALIZED (
    SELECT tree, node, prediction, class_counts FROM __rf_nodes WHERE is_leaf
),
-- The descent: seed every (job, scored row, tree) at node 1, then descend. At a
-- node the effective cell is the base cell, EXCEPT when the split feature is this
-- job's permuted feature (the LEFT JOIN matches only then). Baseline (pf = NULL)
-- never matches -> base cells. Same three-way categorical rule as __rf_walk.
__rf_w AS (
    SELECT j.pf, j.rep, s.i AS rid, t.tree, 1::BIGINT AS node
    FROM __rf_jobset j
    CROSS JOIN __rf_srows s
    CROSS JOIN __rf_treelist t
    CROSS JOIN __rf_chk ck
    WHERE ck.ok
  UNION ALL
    SELECT w.pf, w.rep, w.rid, w.tree,
           w.node * 2 + CASE WHEN i.split_kind = 'num'
                             THEN CASE WHEN coalesce(pc.v, b.v) <= i.threshold THEN 0 ELSE 1 END
                             ELSE CASE WHEN list_contains(i.cats_left,  coalesce(pc.lv, b.lv)) THEN 0
                                       WHEN list_contains(i.cats_right, coalesce(pc.lv, b.lv)) THEN 1
                                       WHEN i.unseen_left                                       THEN 0
                                       ELSE 1 END
                        END AS node
    FROM __rf_w w
    JOIN __rf_int i ON i.tree = w.tree AND i.node = w.node
    JOIN __rf_base b ON b.i = w.rid AND b.col = i.split_feature
    LEFT JOIN __rf_permcell pc ON pc.pf = w.pf AND pc.rep = w.rep
                              AND pc.i = w.rid AND pc.pf = i.split_feature
),
__rf_landed AS (
    SELECT w.pf, w.rep, w.rid, w.tree, l.prediction, l.class_counts
    FROM __rf_w w JOIN __rf_leaf l ON l.tree = w.tree AND l.node = w.node
),
-- Aggregate to a forest prediction per (job, row). Regression: mean leaf value.
-- Classification: soft vote (mean of normalized leaf distributions), argmax with
-- ties to the smallest label -- exactly rf_class_predict.
--
-- DETERMINISM: every floating SUM below is an ORDERED sum
-- (list_sum(list(x ORDER BY key))), never a bare avg()/sum(). DuckDB's hash
-- aggregation accumulates a group's rows in an order that is not stable run to
-- run (even under threads=1), and float addition is non-associative, so a bare
-- avg() jitters by ~1 ULP between runs and "same seed => identical table" would
-- fail. Summing in a fixed key order (tree, then rid, then rep) makes the result
-- bit-identical. The sort keys are unique within each group, so the order is
-- total and reproducible.
__rf_regpred AS (
    SELECT pf, rep, rid, list_sum(list(prediction ORDER BY tree)) / count(*) AS yhat
    FROM __rf_landed GROUP BY pf, rep, rid
),
__rf_clsprob AS (
    SELECT pf, rep, rid, cls, list_sum(list(pp ORDER BY tree)) / count(*) AS p
    FROM (SELECT pf, rep, rid, tree,
                 unnest(map_keys(class_counts))   AS cls,
                 unnest(map_values(class_counts)) / list_sum(map_values(class_counts)) AS pp
          FROM __rf_landed)
    GROUP BY pf, rep, rid, cls
),
__rf_clspred AS (
    SELECT pf, rep, rid, (list(cls ORDER BY p DESC, cls))[1] AS pred
    FROM __rf_clsprob GROUP BY pf, rep, rid
),
-- Score per job. Regression R^2 = 1 - SSE/SST (NULL if SST = 0), SSE summed in
-- rid order. Classification accuracy = mean[pred = y] (a sum of 0/1 ints, exact
-- and order-independent). Only the model's family produces rows (the other
-- branch's predictions are all-NULL / empty and are filtered out here).
__rf_regscore AS (
    SELECT p.pf, p.rep,
           CASE WHEN (SELECT sst FROM __rf_sst) = 0 THEN NULL
                ELSE 1.0 - list_sum(list((y.yv - p.yhat) * (y.yv - p.yhat) ORDER BY p.rid))
                           / (SELECT sst FROM __rf_sst) END AS score
    FROM __rf_regpred p JOIN __rf_yset y ON y.rid = p.rid
    GROUP BY p.pf, p.rep
),
__rf_clsscore AS (
    SELECT p.pf, p.rep, avg((p.pred = y.ys)::INT) AS score
    FROM __rf_clspred p JOIN __rf_yset y ON y.rid = p.rid
    GROUP BY p.pf, p.rep
),
__rf_score AS (
    SELECT pf, rep, score FROM __rf_regscore WHERE (SELECT family FROM __rf_meta) = 'regression'
    UNION ALL
    SELECT pf, rep, score FROM __rf_clsscore WHERE (SELECT family FROM __rf_meta) = 'classification'
),
__rf_baseline AS (SELECT any_value(score) AS baseline FROM __rf_score WHERE pf IS NULL),
-- Reduce: per-repeat importance = baseline - s_{j,r}; report mean and pop std.
-- Every model feature appears (a never-split feature permutes to the baseline
-- score, so importance = 0). Mean and stddev_pop are again ORDERED sums (over
-- rep) for run-to-run bit-identity.
__rf_imprep AS (
    SELECT pf, rep, (SELECT baseline FROM __rf_baseline) - score AS imp
    FROM __rf_score WHERE pf IS NOT NULL
),
__rf_impmean AS (
    SELECT pf, list_sum(list(imp ORDER BY rep)) / count(*) AS mean_imp, count(*) AS nr
    FROM __rf_imprep GROUP BY pf
),
__rf_out AS (
    SELECT i.pf AS feature, m.mean_imp AS importance,
           sqrt(list_sum(list((i.imp - m.mean_imp) * (i.imp - m.mean_imp) ORDER BY i.rep)) / m.nr)
             AS importance_std
    FROM __rf_imprep i JOIN __rf_impmean m ON m.pf = i.pf
    GROUP BY i.pf, m.mean_imp, m.nr
)
-- Drive off __rf_chk so the guards fire even when the scored set is empty (e.g. a
-- missing outcome column makes __rf_out empty); __rf_chk always yields one row.
SELECT o.feature, o.importance, o.importance_std
FROM __rf_chk ck LEFT JOIN __rf_out o ON true
WHERE ck.ok AND o.feature IS NOT NULL
ORDER BY o.importance DESC, o.feature;
