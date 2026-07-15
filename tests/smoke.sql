-- Pure-DuckDB smoke test for the duckRF random-forest macros -- no Python needed.
-- Run from the repo root:
--     duckdb < tests/smoke.sql
-- Every check aborts the script with a non-zero exit code if it fails.
--
-- Run single-threaded so the fit is bit-for-bit reproducible (parallel float
-- sums are non-associative; see the determinism check at the end).

.bail on
PRAGMA threads=1;
.read rf_macros.sql

-- ------------------------------------------------------------------ regression
-- Deterministic signal y = 2 + 3*x1 - x2 with x1 = i (unique per row): one CART
-- grown to purity reproduces y exactly, so R^2 ~ 1 and RMSE ~ 0.
CREATE TABLE reg AS
  SELECT i AS x1, (i % 7) - 3 AS x2, 2 + 3.0 * i - ((i % 7) - 3) AS y
  FROM range(200) t(i);
CREATE TABLE reg_m AS
  SELECT * FROM rf_reg_fit('reg', 'y', n_trees := 1, sample_frac := 1.0,
                           replace_sample := false, mtry := 2, max_depth := NULL);
SELECT CASE
    WHEN (SELECT count(*) FROM rf_reg_predict('reg_m', 'reg')) = 200
     AND (SELECT max(abs(prediction - y)) FROM rf_reg_predict('reg_m', 'reg')) < 1e-6
     AND (SELECT r2 FROM rf_reg_evaluate('reg_m', 'reg', 'y')) > 0.9999
     AND (SELECT rmse FROM rf_reg_evaluate('reg_m', 'reg', 'y')) < 1e-6
    THEN 'PASS  rf_reg recovers a near-noiseless linear signal (R^2 ~ 1)'
    ELSE error('SMOKE FAIL: rf_reg did not recover the clean signal')
  END;

-- --------------------------------------------------------------- classification
-- y is a deterministic function of a feature, so a forest recovers it cleanly.
CREATE TABLE clf AS
  SELECT i AS x1, (i * 13) % 20 AS x2,
         CASE WHEN (i % 10) < 6 THEN 'yes' ELSE 'no' END AS y
  FROM range(400) t(i);
CREATE TABLE clf_m AS SELECT * FROM rf_class_fit('clf', 'y', n_trees := 40, seed := 7);
SELECT CASE
    WHEN (SELECT accuracy FROM rf_class_evaluate('clf_m', 'clf', 'y')) > 0.95
     AND (SELECT auc FROM rf_class_evaluate('clf_m', 'clf', 'y')) BETWEEN 0 AND 1
     AND (SELECT bool_and(abs(list_sum(map_values(probs)) - 1.0) < 1e-9)
          FROM rf_class_predict('clf_m', 'clf'))
     AND (SELECT bool_and(pred IN ('yes', 'no')) FROM rf_class_predict('clf_m', 'clf'))
    THEN 'PASS  rf_class recovers a clean class signal; probs normalize per row'
    ELSE error('SMOKE FAIL: rf_class did not recover the clean class signal')
  END;

-- --------------------------------------------------------- categorical subset split
-- Levels {A,B} have mean ~0, {C,D,E} have mean ~10. The optimal split is the
-- SUBSET {A,B} | {C,D,E}, which a native categorical split finds at depth 1
-- (split_kind = 'cat'); a grown tree then reproduces the group means.
CREATE TABLE cat AS
  SELECT (CASE i % 5 WHEN 0 THEN 'A' WHEN 1 THEN 'B' WHEN 2 THEN 'C'
                     WHEN 3 THEN 'D' ELSE 'E' END) AS g,
         (CASE WHEN i % 5 < 2 THEN 0.0 ELSE 10.0 END) + ((i % 3) - 1) * 0.001 AS y
  FROM range(300) t(i);
CREATE TABLE cat_m AS SELECT * FROM rf_reg_fit('cat', 'y', n_trees := 20, max_depth := 6, seed := 1);
SELECT CASE
    WHEN (SELECT split_kind FROM rf_reg_fit('cat', 'y', n_trees := 1, sample_frac := 1.0,
             replace_sample := false, mtry := 1, max_depth := 1) WHERE node = 1) = 'cat'
     AND (SELECT feature_kinds = ['cat'] FROM cat_m LIMIT 1)
     AND (SELECT r2 FROM rf_reg_evaluate('cat_m', 'cat', 'y')) > 0.999
    THEN 'PASS  categorical subset split isolates {A,B} | {C,D,E}'
    ELSE error('SMOKE FAIL: categorical subset split failed')
  END;

-- ----------------------------------------------------------- constant feature ok
-- A perfectly constant feature must be handled gracefully (never split on) and
-- carry exactly zero MDI importance.
CREATE TABLE cst AS SELECT x1, x2, 4.2 AS c, y FROM reg;
CREATE TABLE cst_m AS SELECT * FROM rf_reg_fit('cst', 'y', n_trees := 20, seed := 1, max_depth := 8);
SELECT CASE
    WHEN (SELECT count(*) FROM rf_reg_predict('cst_m', 'cst')) = 200
     AND (SELECT importance FROM rf_importance('cst_m') WHERE feature = 'c') = 0.0
    THEN 'PASS  constant feature handled; its MDI importance is exactly 0'
    ELSE error('SMOKE FAIL: constant feature not handled / importance not 0')
  END;

-- ------------------------------------------------------ summary / importance shapes
-- rf_summary returns one sane row; rf_importance lists every feature and sums to 1.
SELECT CASE
    WHEN (SELECT n_trees FROM rf_summary('cst_m')) = 20
     AND (SELECT family FROM rf_summary('cst_m')) = 'regression'
     AND (SELECT n_leaves < n_nodes AND n_nodes > 0 FROM rf_summary('cst_m'))
     AND (SELECT count(*) FROM rf_importance('cst_m')) = 3           -- x1, x2, c
     AND (SELECT abs(sum(importance) - 1.0) < 1e-9 FROM rf_importance('cst_m'))
    THEN 'PASS  rf_summary / rf_importance return sane shapes (importance sums to 1)'
    ELSE error('SMOKE FAIL: rf_summary / rf_importance shapes wrong')
  END;

-- ------------------------------------------------------ permutation importance
-- The cardinality-unbiased complement to MDI: shuffling the informative feature
-- x1 must degrade the model far more than shuffling the pure-noise feature added
-- here, so x1's permutation importance clearly outranks the noise feature's (and
-- the noise feature sits near zero). Same-seed runs are bit-identical (threads=1).
CREATE TABLE perm AS
  SELECT x1, x2, (x1 * 2654435761) % 97 AS noise, y FROM reg;
CREATE TABLE perm_m AS SELECT * FROM rf_reg_fit('perm', 'y', n_trees := 30, seed := 5, max_depth := 8);
SELECT CASE
    WHEN (SELECT importance FROM rf_permutation_importance('perm_m', 'perm', 'y', n_repeats := 10)
          WHERE feature = 'x1')
       > (SELECT importance FROM rf_permutation_importance('perm_m', 'perm', 'y', n_repeats := 10)
          WHERE feature = 'noise')
     AND abs((SELECT importance FROM rf_permutation_importance('perm_m', 'perm', 'y', n_repeats := 10)
              WHERE feature = 'noise')) < 0.05
     AND (SELECT count(*) FROM rf_permutation_importance('perm_m', 'perm', 'y')) = 3
     AND (SELECT count(*) FROM (
            (SELECT * FROM rf_permutation_importance('perm_m', 'perm', 'y', n_repeats := 8, seed := 42)
             EXCEPT
             SELECT * FROM rf_permutation_importance('perm_m', 'perm', 'y', n_repeats := 8, seed := 42))
            UNION ALL
            (SELECT * FROM rf_permutation_importance('perm_m', 'perm', 'y', n_repeats := 8, seed := 42)
             EXCEPT
             SELECT * FROM rf_permutation_importance('perm_m', 'perm', 'y', n_repeats := 8, seed := 42)))) = 0
    THEN 'PASS  permutation importance: signal x1 outranks noise; deterministic'
    ELSE error('SMOKE FAIL: permutation importance signal/noise or determinism wrong')
  END;

-- ------------------------------------------------------------------------- OOB
-- Out-of-bag runs on the exact training table: scored + excluded == n_train, and
-- the OOB metrics are finite.
CREATE TABLE oob_m AS SELECT * FROM rf_reg_fit('reg', 'y', n_trees := 50, seed := 3, max_depth := 8);
SELECT CASE
    WHEN (SELECT n + n_excluded FROM rf_reg_oob('oob_m', 'reg', 'y')) = 200
     AND (SELECT isfinite(rmse) AND isfinite(r2) FROM rf_reg_oob('oob_m', 'reg', 'y'))
     AND (SELECT count(*) FROM rf_reg_oob_predict('oob_m', 'reg')) = 200
    THEN 'PASS  out-of-bag scores the training table (n + n_excluded = n_train)'
    ELSE error('SMOKE FAIL: OOB output invalid')
  END;

-- ------------------------------------------------------ determinism (threads=1)
-- Same seed -> bit-identical model; a different seed -> a different model. Compare
-- the scalar structure columns (list/map columns are equal whenever these are).
CREATE TABLE det_a AS SELECT * FROM rf_reg_fit('reg', 'y', n_trees := 15, seed := 42, max_depth := 6);
CREATE TABLE det_b AS SELECT * FROM rf_reg_fit('reg', 'y', n_trees := 15, seed := 42, max_depth := 6);
CREATE TABLE det_c AS SELECT * FROM rf_reg_fit('reg', 'y', n_trees := 15, seed := 43, max_depth := 6);
SELECT CASE
    WHEN (SELECT count(*) FROM (
            (SELECT tree, node, is_leaf, split_feature, threshold, prediction, n_rows FROM det_a
             EXCEPT
             SELECT tree, node, is_leaf, split_feature, threshold, prediction, n_rows FROM det_b)
            UNION ALL
            (SELECT tree, node, is_leaf, split_feature, threshold, prediction, n_rows FROM det_b
             EXCEPT
             SELECT tree, node, is_leaf, split_feature, threshold, prediction, n_rows FROM det_a))) = 0
     AND (SELECT count(*) FROM (
            SELECT tree, node, threshold, prediction FROM det_a
            EXCEPT
            SELECT tree, node, threshold, prediction FROM det_c)) > 0
    THEN 'PASS  determinism: same seed -> identical model, different seed -> different'
    ELSE error('SMOKE FAIL: determinism check failed under threads=1')
  END;

SELECT 'ALL SMOKE CHECKS PASSED' AS result;
