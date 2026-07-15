"""Deterministic test suite for the duckRF DuckDB random-forest macros.

Every non-trivial fit / predict / evaluate is checked against an independent
scikit-learn or numpy reference on the same fixed-seed data, so a failure means
the macros disagree with a trusted implementation -- not merely that some
previously-recorded number drifted. This mirrors the adversarial verification
the library was developed under (see the scratchpad batteries) in a form anyone
can reproduce with `pytest`.

The strongest checks are the SINGLE-TREE CART equivalences: with
`n_trees:=1, sample_frac:=1.0, replace_sample:=false, mtry:=d` duckRF grows one
deterministic CART that must match sklearn's DecisionTree{Regressor,Classifier}
(max_features=None) to ~1e-9 on data with no tied splits -- predictions,
impurities and MDI importances all agree exactly. Forest-level behaviour is
checked statistically against sklearn's RandomForest, the categorical subset
splits are checked against a brute-force optimum, OOB membership is replayed in
numpy, and the whole error / reserved-name contract is exercised.

A handful of documented limitations are encoded honestly rather than as false
requirements:
  * K>2 multiclass categorical splits use the Breiman/Ripley K-orderings
    heuristic, so they are asserted "at least as good as the best singleton
    split", not brute-force optimal.
  * bit-identical model reproducibility needs single-threaded float sums, so the
    session runs under `PRAGMA threads=1` (predictions agree to ~1e-14 either
    way; bootstrap membership / mtry lottery / OOB are bit-stable regardless).

Run from the repo root:  .venv/bin/python -m pytest tests/ -q
"""

import itertools
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")  # silence sklearn solver/deprecation chatter

MACRO_FILE = Path(__file__).resolve().parents[1] / "rf_macros.sql"
DuckDBError = getattr(duckdb, "Error", Exception)

SINGLE = dict(n_trees=1, sample_frac=1.0, replace_sample="false")  # one CART


# --------------------------------------------------------------------------- #
# Fixtures & helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def con():
    c = duckdb.connect()
    c.execute(MACRO_FILE.read_text())
    c.execute("PRAGMA threads=1")  # bit-identical, associative float sums
    return c


def _load(con, name, df):
    con.register(f"_src_{name}", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _src_{name}")
    con.unregister(f"_src_{name}")


def _params(kw):
    return "".join(f", {k} := {v}" for k, v in kw.items())


def df_run(con, sql):
    return con.execute(sql).df()


def fit_tbl(con, macro, tbl, model, outcome="y", **kw):
    """Materialise a model table `model` from a *_fit macro; return its name."""
    con.execute(
        f"CREATE OR REPLACE TABLE {model} AS "
        f"SELECT * FROM {macro}('{tbl}', '{outcome}'{_params(kw)})"
    )
    return model


def frame(X, cols=None):
    cols = cols or [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols)


# --- an independent python scorer for the model table (used for cross-checks) --
def _descend(nodes, tree, row):
    n = 1
    while True:
        r = nodes[(tree, n)]
        if r["is_leaf"]:
            return r
        f = r["split_feature"]
        v = row[f]
        if r["split_kind"] == "num":
            go_left = float(v) <= r["threshold"]
        else:
            lv = str(v)
            cl = list(r["cats_left"]) if r["cats_left"] is not None else []
            cr = list(r["cats_right"]) if r["cats_right"] is not None else []
            go_left = True if lv in cl else False if lv in cr else bool(r["unseen_left"])
        n = 2 * n + (0 if go_left else 1)


def py_forest_predict(model_df, X):
    nodes = {(int(r.tree), int(r.node)): r._asdict()
             for r in model_df.itertuples(index=False)}
    trees = sorted({t for t, _ in nodes})
    fam = model_df["family"].iloc[0]
    classes = list(model_df["classes"].iloc[0]) if fam == "classification" else None
    out = []
    for _, row in X.iterrows():
        if fam == "regression":
            out.append(np.mean([_descend(nodes, t, row)["prediction"] for t in trees]))
        else:
            acc = np.zeros(len(classes))
            for t in trees:
                cc = _descend(nodes, t, row)["class_counts"]
                v = np.array([float(cc[k]) for k in classes])
                acc += v / v.sum()
            out.append(acc / len(trees))
    return np.array(out)


def cart_reg(con, X, y, **params):
    """Fit one deterministic duckRF CART; return (duck_pred, sklearn_pred) aligned
    to X's row order. Training table carries only features + y (no index column
    that could leak in as a splitter); scoring uses an rid passthrough for order."""
    md = params.pop("max_depth", "NULL")
    _load(con, "tr", frame(X).assign(y=y))
    fit_tbl(con, "rf_reg_fit", "tr", "m", mtry=X.shape[1], max_depth=md,
            criterion="'mse'", **SINGLE, **params)
    _load(con, "sc", frame(X).assign(rid=np.arange(len(X))))
    out = df_run(con, "SELECT rid, prediction FROM rf_reg_predict('m','sc')").sort_values("rid")
    duck = out["prediction"].to_numpy()
    sk = DecisionTreeRegressor(max_features=None, random_state=0,
                               max_depth=None if md == "NULL" else md,
                               **{k: v for k, v in params.items()
                                  if k in ("min_samples_split", "min_samples_leaf",
                                           "min_impurity_decrease")}).fit(X, y)
    return duck, sk.predict(X)


def cart_cls(con, X, y, criterion="gini", max_depth="NULL", weight=None, **params):
    """Fit one duckRF classification CART; return (duck_labels, duck_proba,
    sklearn_tree, classes) aligned to X's row order."""
    classes = sorted(set(str(v) for v in y))
    _load(con, "tc", frame(X).assign(y=[str(v) for v in y]))
    fit_tbl(con, "rf_class_fit", "tc", "mc", mtry=X.shape[1], max_depth=max_depth,
            criterion=f"'{criterion}'", **SINGLE, **params)
    _load(con, "scc", frame(X).assign(rid=np.arange(len(X))))
    out = df_run(con, "SELECT rid, pred, probs FROM rf_class_predict('mc','scc')").sort_values("rid")
    lab = out["pred"].to_numpy()
    proba = np.array([[row[c] for c in classes] for row in out["probs"]])
    sk = DecisionTreeClassifier(
        criterion="gini" if criterion == "gini" else "entropy", max_features=None,
        random_state=0, max_depth=None if max_depth == "NULL" else max_depth,
        class_weight=weight,
        **{k: v for k, v in params.items()
           if k in ("min_samples_split", "min_samples_leaf", "min_impurity_decrease")}
    ).fit(X, [str(v) for v in y])
    assert list(sk.classes_) == classes
    return lab, proba, sk, classes


def _sse(y, p):
    return float(np.sum((np.asarray(y) - np.asarray(p)) ** 2))


# ===========================================================================
# 1. Single-tree CART equivalence vs scikit-learn (the headline check)
#
# On tie-free data duckRF grows exactly sklearn's CART (predictions match to
# ~1e-9). Where equally-good tied splits exist, sklearn and duckRF may break the
# tie differently -- both valid -- so those configurations assert the weaker but
# still-adversarial property "duckRF's realized objective is no WORSE than
# sklearn's" (SSE for regression, training accuracy for classification).
# ===========================================================================
class TestSingleTreeParity:
    @pytest.mark.parametrize("max_depth", [1, 2, 3, "NULL"])
    def test_regression_exact_on_untied_data(self, con, max_depth):
        # shallow (structurally forced) or grown-to-purity -> exact equivalence.
        rng = np.random.default_rng(12345)
        X = rng.standard_normal((300, 4))
        y = X @ [1.5, -2.0, 0.7, 3.1] + 0.3 * rng.standard_normal(300)
        duck, sk = cart_reg(con, X, y, max_depth=max_depth)
        assert np.max(np.abs(duck - sk)) < 1e-9

    @pytest.mark.parametrize("max_depth", [5, 10])
    def test_regression_never_worse_when_depth_limited(self, con, max_depth):
        rng = np.random.default_rng(12345)
        X = rng.standard_normal((300, 4))
        y = X @ [1.5, -2.0, 0.7, 3.1] + 0.3 * rng.standard_normal(300)
        duck, sk = cart_reg(con, X, y, max_depth=max_depth)
        assert np.max(np.abs(duck - sk)) < 1e-9 or _sse(y, duck) <= _sse(y, sk) * (1 + 1e-9) + 1e-6

    @pytest.mark.parametrize("msl", [1, 3, 7, 20])
    def test_regression_min_samples_leaf(self, con, msl):
        rng = np.random.default_rng(21)
        X = rng.standard_normal((300, 4))
        y = X @ [1.0, -2.0, 0.5, 1.3] + 0.3 * rng.standard_normal(300)
        duck, sk = cart_reg(con, X, y, min_samples_leaf=msl)
        assert np.max(np.abs(duck - sk)) < 1e-9 or _sse(y, duck) <= _sse(y, sk) * (1 + 1e-9) + 1e-6

    @pytest.mark.parametrize("mi", [0.0, 0.01, 0.1, 1.0])
    def test_regression_min_impurity_decrease(self, con, mi):
        rng = np.random.default_rng(22)
        X = rng.standard_normal((300, 4))
        y = X @ [1.5, -2.0, 0.7, 3.1] + 0.3 * rng.standard_normal(300)
        duck, sk = cart_reg(con, X, y, min_impurity_decrease=mi)
        assert np.max(np.abs(duck - sk)) < 1e-9 or _sse(y, duck) <= _sse(y, sk) * (1 + 1e-9) + 1e-6

    def test_regression_large_mean_target(self, con):
        # y centred at 1e6: un-centering of the leaf mean must stay exact.
        rng = np.random.default_rng(23)
        X = rng.standard_normal((250, 3))
        y = 1e6 + X @ [2.0, -1.0, 0.5] + 0.01 * rng.standard_normal(250)
        duck, sk = cart_reg(con, X, y, max_depth=6)
        assert np.max(np.abs(duck - sk)) < 1e-3 or _sse(y, duck) <= _sse(y, sk) * (1 + 1e-9) + 1e-3

    @pytest.mark.parametrize("criterion", ["gini", "entropy"])
    @pytest.mark.parametrize("max_depth", [1, 2, 3])
    def test_binary_classification_exact_on_untied_data(self, con, criterion, max_depth):
        rng = np.random.default_rng(31)
        X = rng.standard_normal((300, 4))
        y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0).astype(int)
        lab, p, sk, _ = cart_cls(con, X, y, criterion, max_depth)
        assert np.array_equal(lab, sk.predict(X))
        assert np.max(np.abs(p - sk.predict_proba(X))) < 1e-9

    @pytest.mark.parametrize("criterion", ["gini", "entropy"])
    def test_binary_classification_never_worse_when_grown(self, con, criterion):
        rng = np.random.default_rng(31)
        X = rng.standard_normal((300, 4))
        y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0).astype(int)
        lab, p, sk, _ = cart_cls(con, X, y, criterion, "NULL")
        ystr = np.array([str(v) for v in y])
        assert (lab == ystr).mean() >= sk.score(X, ystr) - 1e-12

    @pytest.mark.parametrize("criterion", ["gini", "entropy"])
    def test_multiclass_probs_never_worse(self, con, criterion):
        rng = np.random.default_rng(32)
        X = rng.standard_normal((600, 4))
        y = np.digitize(X[:, 0] + 0.2 * X[:, 1] - 0.3 * X[:, 2], [-1.0, -0.3, 0.3, 1.0])
        lab, p, sk, classes = cart_cls(con, X, y, criterion, "NULL", min_samples_leaf=40)
        ystr = np.array([str(v) for v in y])
        # probs are dense over all classes and normalized
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
        # never worse than sklearn's tree on the same weighting
        assert (lab == ystr).mean() >= sk.score(X, ystr) - 1e-12

    def test_entropy_impurity_is_in_bits(self, con):
        # A wrong log base would show up in the reported impurity column.
        rng = np.random.default_rng(33)
        X = rng.standard_normal((200, 3))
        y = np.digitize(X[:, 0] + 0.3 * X[:, 1], [-0.5, 0.4])
        cart_cls(con, X, y, "entropy", 3)
        duck = df_run(con, "SELECT impurity FROM mc WHERE node=1").impurity[0]
        sk = DecisionTreeClassifier(criterion="entropy", max_depth=3,
                                    max_features=None, random_state=0
                                    ).fit(X, [str(v) for v in y])
        assert abs(duck - sk.tree_.impurity[0]) < 1e-12

    def test_mse_root_impurity_is_variance(self, con):
        rng = np.random.default_rng(34)
        X = rng.standard_normal((200, 3))
        y = X @ [2.0, -1.0, 0.5] + 0.3 * rng.standard_normal(200)
        cart_reg(con, X, y, max_depth=3)
        duck = df_run(con, "SELECT impurity FROM m WHERE node=1").impurity[0]
        sk = DecisionTreeRegressor(max_depth=3, max_features=None,
                                   random_state=0).fit(X, y)
        assert abs(duck - sk.tree_.impurity[0]) < 1e-9

    def test_xor_zero_gain_root_split_accepted(self, con):
        # sklearn takes a zero-gain root split on XOR; duckRF must too (train acc 1).
        xs = np.array([[a, b] for a in (0, 1) for b in (0, 1)] * 25, dtype=float)
        y = (xs[:, 0].astype(int) ^ xs[:, 1].astype(int)).astype(str)
        df = pd.DataFrame(xs, columns=["a", "b"]).assign(y=y)
        _load(con, "xr", df)
        fit_tbl(con, "rf_class_fit", "xr", "mx", mtry=2, **SINGLE)
        m = df_run(con, "SELECT * FROM mx")
        p = py_forest_predict(m, df[["a", "b"]])
        lab = np.array(list(m["classes"].iloc[0]))[p.argmax(1)]
        assert (lab == y).mean() == 1.0


# ===========================================================================
# 2. min_impurity_decrease pruning leaf counts (log-base sensitive)
# ===========================================================================
class TestPruning:
    @pytest.mark.parametrize("criterion", ["gini", "entropy"])
    @pytest.mark.parametrize("mi", [0.0, 0.001, 0.005, 0.01, 0.02, 0.05])
    def test_classification_leaf_count_matches_sklearn(self, con, criterion, mi):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((400, 4))
        y = np.digitize(X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2], [-0.6, 0.0, 0.6])
        df = frame(X).assign(y=[str(v) for v in y])
        _load(con, "tr", df)
        fit_tbl(con, "rf_class_fit", "tr", "m", mtry=4, max_depth="NULL",
                min_impurity_decrease=mi, criterion=f"'{criterion}'", **SINGLE)
        nl = df_run(con, "SELECT count(*) c FROM m WHERE is_leaf").c[0]
        sk = DecisionTreeClassifier(criterion="gini" if criterion == "gini" else "entropy",
                                    min_impurity_decrease=mi, max_features=None,
                                    random_state=0).fit(X, [str(v) for v in y])
        # exact on well-separated pruning; +-2 slack for tie-break at fully-grown
        # trees. A wrong entropy log base would shift every count by many leaves.
        assert abs(nl - sk.get_n_leaves()) <= 2

    @pytest.mark.parametrize("mi", [0.0, 0.05, 0.2, 0.5, 1.0, 2.0])
    def test_regression_leaf_count_matches_sklearn(self, con, mi):
        rng = np.random.default_rng(12)
        X = rng.standard_normal((400, 4))
        y = X @ [1.0, -2.0, 0.5, 3.0] + 0.3 * rng.standard_normal(400)
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", mtry=4, max_depth="NULL",
                min_impurity_decrease=mi, criterion="'mse'", **SINGLE)
        nl = df_run(con, "SELECT count(*) c FROM m WHERE is_leaf").c[0]
        sk = DecisionTreeRegressor(min_impurity_decrease=mi, max_features=None,
                                   random_state=0).fit(X, y)
        assert abs(nl - sk.get_n_leaves()) <= 2


# ===========================================================================
# 3. Forest-level statistical parity vs sklearn RandomForest
# ===========================================================================
class TestForest:
    def test_regression_r2_near_sklearn(self, con):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(1200, 6))
        y = np.sin(X[:, 0] * 2) * 3 + X[:, 1] ** 2 - X[:, 2] + rng.normal(0, 0.5, 1200)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
        _load(con, "tr", frame(Xtr).assign(y=ytr))
        _load(con, "te", frame(Xte))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=100, mtry=2, max_depth=12, seed=0)
        p = df_run(con, "SELECT * FROM rf_reg_predict('m','te')")
        # predict passes columns through in input order here (single-threaded scan)
        r2_rf = r2_score(yte, p["prediction"].to_numpy())
        r2_sk = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=12,
                                      random_state=0).fit(Xtr, ytr).score(Xte, yte)
        assert abs(r2_rf - r2_sk) < 0.08

    def test_classification_accuracy_near_sklearn(self, con):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(1000, 6))
        y = np.where(X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(0, 0.5, 1000) > 0, "yes", "no")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
        _load(con, "tr", frame(Xtr).assign(y=ytr))
        _load(con, "te", frame(Xte))
        fit_tbl(con, "rf_class_fit", "tr", "m", n_trees=100, max_depth=12, seed=0)
        acc_rf = df_run(con, "SELECT * FROM rf_class_predict('m','te')")["pred"].to_numpy()
        acc_rf = (acc_rf == yte).mean()
        acc_sk = RandomForestClassifier(n_estimators=100, max_depth=12,
                                        random_state=0).fit(Xtr, ytr).score(Xte, yte)
        assert abs(acc_rf - acc_sk) < 0.08

    def test_bootstrap_draws_632_distinct(self, con):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(800, 4))
        y = X @ [1.0, -1.0, 0.5, 0.2] + rng.normal(0, 0.3, 800)
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=100, seed=1, max_depth=8)
        frac = df_run(con, """
            SELECT avg(u)/800.0 f FROM (
              SELECT tree, sum(n_rows) FILTER (is_leaf) AS u FROM m GROUP BY tree)""").f[0]
        assert abs(frac - 0.632) < 0.02

    def test_python_scorer_matches_predict(self, con):
        # rf_*_predict must agree with an independent tree-walk of the model table.
        rng = np.random.default_rng(6)
        X = rng.normal(size=(200, 4))
        y = X @ [1.0, -1.0, 0.5, 0.2] + rng.normal(0, 0.3, 200)
        df = frame(X).assign(y=y)
        _load(con, "tr", df)
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=25, seed=2, max_depth=6)
        model = df_run(con, "SELECT * FROM m")
        out = df_run(con, "SELECT * FROM rf_reg_predict('m','tr')")
        ref = py_forest_predict(model, frame(X))
        assert np.max(np.abs(out["prediction"].to_numpy() - ref)) < 1e-9


# ===========================================================================
# 4. Predict semantics: proba round-trip, predict_trees, na_action
# ===========================================================================
class TestPredict:
    def _reg_forest(self, con, seed=41, **kw):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(300, 4))
        y = X @ [2.0, -1.0, 0.5, 1.3] + rng.normal(0, 0.3, 300)
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", **kw)
        return X, y

    def test_probs_are_normalized(self, con):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(300, 4))
        y = np.where(X[:, 0] - X[:, 1] > 0, "hi", "lo")
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_class_fit", "tr", "m", n_trees=20, seed=1)
        s = df_run(con, """SELECT bool_and(abs(list_sum(map_values(probs))-1.0)<1e-9) ok,
                                  bool_and(pred = ANY(classes)) lbl
                           FROM rf_class_predict('m','tr'), (SELECT any_value(classes) classes FROM m)""")
        assert bool(s.ok[0]) and bool(s.lbl[0])

    def test_predict_trees_one_row_per_row_tree(self, con):
        self._reg_forest(con, n_trees=15, seed=7, max_depth=6)
        out = df_run(con, "SELECT count(*) c, count(DISTINCT tree) t FROM rf_reg_predict_trees('m','tr')")
        assert out.t[0] == 15 and out.c[0] == 15 * 300

    def test_n_trees_scoring_cap(self, con):
        self._reg_forest(con, n_trees=40, seed=7)
        t = df_run(con, "SELECT count(DISTINCT tree) t FROM rf_reg_predict_trees('m','tr', n_trees:=5)").t[0]
        assert t == 5

    def test_null_feature_gives_null_prediction(self, con):
        X, y = self._reg_forest(con, n_trees=1, seed=8, mtry=4, max_depth=4,
                                sample_frac=1.0, replace_sample="false")
        sf = df_run(con, "SELECT any_value(split_feature) f FROM m WHERE node=1").f[0]
        d = frame(X[:3]).copy()
        d.loc[1, sf] = np.nan  # null the dominant split feature on row 1
        _load(con, "sc", d)
        r = df_run(con, "SELECT prediction FROM rf_reg_predict('m','sc')").prediction
        assert pd.isna(r[1]) and not pd.isna(r[0])

    def test_skip_tree_recovers_null_row(self, con):
        # With mtry=1 many trees never test the nulled feature and still land on a leaf.
        rng = np.random.default_rng(9)
        X = rng.normal(size=(300, 4))
        y = X @ [2.0, -1.0, 0.5, 1.3] + rng.normal(0, 0.3, 300)
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=60, mtry=1, max_depth=6, seed=11)
        d = frame(X[:3]).copy()
        d.loc[1, "x2"] = np.nan
        _load(con, "sc", d)
        null_a = df_run(con, "SELECT prediction FROM rf_reg_predict('m','sc')").prediction
        skip = df_run(con, "SELECT prediction FROM rf_reg_predict('m','sc', na_action:='skip_tree')").prediction
        assert pd.isna(null_a[1]) and not pd.isna(skip[1])


# ===========================================================================
# 5. Evaluate metrics vs the sklearn metric functions (single-tree = exact)
# ===========================================================================
class TestEvaluate:
    def test_regression_metrics_match_sklearn(self, con):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 4))
        y = X @ [2.0, -1.0, 0.5, 1.3] + rng.normal(0, 0.3, 300) + 100.0
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", mtry=4, max_depth=3, seed=42, **SINGLE)
        ev = df_run(con, "SELECT * FROM rf_reg_evaluate('m','tr','y')").iloc[0]
        yh = DecisionTreeRegressor(max_depth=3, max_features=None, random_state=0).fit(X, y).predict(X)
        assert ev["n"] == 300
        assert abs(ev["rmse"] - np.sqrt(mean_squared_error(y, yh))) < 1e-9
        assert abs(ev["mae"] - mean_absolute_error(y, yh)) < 1e-9
        assert abs(ev["r2"] - r2_score(y, yh)) < 1e-9

    @pytest.mark.parametrize("kind,crit", [("bin", "gini"), ("multi", "entropy")])
    def test_classification_metrics_match_sklearn(self, con, kind, crit):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 4))
        if kind == "bin":
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
        else:
            y = np.digitize(X[:, 0] + 0.5 * X[:, 1], [-0.5, 0.5])
        ystr = y.astype(str)
        _load(con, "tc", frame(X).assign(y=ystr))
        fit_tbl(con, "rf_class_fit", "tc", "mc", mtry=4, max_depth=3,
                criterion=f"'{crit}'", seed=42, **SINGLE)
        ev = df_run(con, "SELECT * FROM rf_class_evaluate('mc','tc','y')").iloc[0]
        sk = DecisionTreeClassifier(max_depth=3, criterion=crit, max_features=None,
                                    random_state=0).fit(X, ystr)
        classes = list(sk.classes_)
        P = sk.predict_proba(X)
        assert abs(ev["accuracy"] - accuracy_score(ystr, sk.predict(X))) < 1e-9
        assert abs(ev["log_loss"] - log_loss(ystr, P, labels=classes)) < 1e-9
        assert abs(ev["brier"] - brier_score_loss(ystr, P, labels=classes)) < 1e-9
        if kind == "bin":
            auc = roc_auc_score((y == 1).astype(int), P[:, classes.index("1")])
            assert abs(ev["auc"] - auc) < 1e-9
        else:
            assert pd.isna(ev["auc"])  # AUC is binary-only

    def test_evaluate_drops_null_outcome_rows(self, con):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 4))
        y = X @ [1.0, -1.0, 0.5, 0.2] + rng.normal(0, 0.3, 300)
        df = frame(X).assign(y=y)
        _load(con, "tr", df)
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=5, seed=1)
        df.loc[[1, 2, 3], "y"] = np.nan
        _load(con, "te", df)
        assert df_run(con, "SELECT n FROM rf_reg_evaluate('m','te','y')").n[0] == 297


# ===========================================================================
# 6. Categorical: subset-split optimality, ENUM, headline cat-outcome use case
# ===========================================================================
class TestCategorical:
    def _stump(self, con, df, macro, criterion):
        _load(con, "tr", df)
        r = con.execute(f"""SELECT is_leaf, split_kind, cats_left, cats_right, imp_decrease
            FROM {macro}('tr','y', n_trees:=1, sample_frac:=1.0, replace_sample:=false,
                         mtry:=1, max_depth:=1, min_impurity_decrease:=0.0,
                         criterion:='{criterion}') WHERE node=1""").fetchone()
        return r  # (is_leaf, split_kind, cats_left, cats_right, imp_decrease)

    @staticmethod
    def _brute_reg(levels, y):
        L = sorted(set(levels)); yv = np.array(y); tot = yv.sum(); n = len(yv)
        best = -np.inf
        for r in range(1, len(L)):
            for sub in itertools.combinations(L, r):
                mask = np.isin(levels, sub)
                if mask.sum() == 0 or (~mask).sum() == 0:
                    continue
                best = max(best, yv[mask].sum() ** 2 / mask.sum()
                           + yv[~mask].sum() ** 2 / (~mask).sum() - tot ** 2 / n)
        return best

    @staticmethod
    def _brute_gini(levels, y):
        L = sorted(set(levels)); classes = sorted(set(y)); yv = np.array(y); n = len(yv)
        def q(mask):
            m = mask.sum()
            return sum((yv[mask] == c).sum() ** 2 for c in classes) / m if m else 0.0
        parent = q(np.ones(n, bool)); best = -np.inf
        for r in range(1, len(L)):
            for sub in itertools.combinations(L, r):
                mask = np.isin(levels, sub)
                if mask.sum() == 0 or (~mask).sum() == 0:
                    continue
                best = max(best, q(mask) + q(~mask) - parent)
        return best

    def test_regression_subset_split_is_brute_force_optimal(self, con):
        rng = np.random.default_rng(7)
        for _ in range(25):
            L = rng.integers(3, 7); n = rng.integers(15, 60)
            levels = np.array([chr(65 + i) for i in rng.integers(0, L, n)])
            if len(set(levels)) < 2:
                continue
            y = rng.standard_normal(n)
            r = self._stump(con, pd.DataFrame({"x": levels, "y": y}), "rf_reg_fit", "mse")
            if r[0]:  # leaf
                assert self._brute_reg(levels, y) < 1e-9
                continue
            assert r[1] == "cat"
            assert abs(r[4] - self._brute_reg(levels, y)) < 1e-6

    def test_binary_gini_subset_split_is_brute_force_optimal(self, con):
        rng = np.random.default_rng(8)
        for _ in range(25):
            L = rng.integers(3, 7); n = rng.integers(20, 60)
            levels = np.array([chr(65 + i) for i in rng.integers(0, L, n)])
            y = np.array([str(v) for v in rng.integers(0, 2, n)])
            if len(set(levels)) < 2 or len(set(y)) < 2:
                continue
            r = self._stump(con, pd.DataFrame({"x": levels, "y": y}), "rf_class_fit", "gini")
            if r[0]:
                continue
            assert r[1] == "cat"
            assert abs(r[4] - self._brute_gini(levels, y)) < 1e-6

    def test_multiclass_at_least_as_good_as_best_singleton(self, con):
        # K>2 uses the K-orderings heuristic: not guaranteed optimal, but must beat
        # every one-level-vs-rest singleton split (documented lower bound).
        rng = np.random.default_rng(12345)
        checked = 0
        for _ in range(120):
            K = rng.integers(3, 6); L = rng.integers(3, 8)
            levels = [f"lv{i}" for i in range(L)]
            clab = [f"c{i}" for i in range(K)]
            n = int(rng.integers(L * 4, L * 12))
            x = rng.choice(levels, size=n)
            for i, lv in enumerate(levels):
                x[i % n] = lv
            dist = {lv: rng.dirichlet(np.ones(K) * rng.uniform(0.3, 2)) for lv in levels}
            y = np.array([clab[rng.choice(K, p=dist[xx])] for xx in x])
            if len(np.unique(y)) < 2:
                continue
            r = self._stump(con, pd.DataFrame({"x": x, "y": y}), "rf_class_fit", "gini")
            if r[0] or r[1] != "cat":
                continue
            present = list(np.unique(x)); classes = sorted(set(y))
            def q(mask):
                m = mask.sum()
                return sum((y[mask] == c).sum() ** 2 for c in classes) / m if m else 0.0
            parent = q(np.ones(n, bool))
            best_single = max(q(x == lv) + q(x != lv) - parent for lv in present)
            assert r[4] >= best_single - 1e-9
            checked += 1
        assert checked > 30  # the battery actually exercised the heuristic

    def test_numeric_looking_levels_stay_categorical(self, con):
        # optimal grouping {'1','10'} vs {'2'} is impossible for a numeric threshold.
        rng = np.random.default_rng(1)
        rows = []
        for _ in range(40):
            rows += [("1", rng.normal(0, 0.3)), ("10", rng.normal(0, 0.3)),
                     ("2", rng.normal(10, 0.3))]
        df = pd.DataFrame(rows, columns=["x", "y"])
        r = self._stump(con, df, "rf_reg_fit", "mse")
        assert r[1] == "cat"
        fk = df_run(con, "SELECT any_value(feature_kinds) k FROM rf_reg_fit('tr','y',n_trees:=1,max_depth:=1)").k[0]
        assert list(fk) == ["cat"]

    def test_enum_feature_and_outcome(self, con):
        con.execute("DROP TYPE IF EXISTS mood")
        con.execute("CREATE TYPE mood AS ENUM ('lo','mid','hi')")
        con.execute("""CREATE OR REPLACE TABLE etr AS
            SELECT (CASE WHEN i%3=0 THEN 'lo' WHEN i%3=1 THEN 'mid' ELSE 'hi' END)::mood AS m,
                   (i%3)::DOUBLE + (i%7)*0.01 AS y FROM range(90) t(i)""")
        fk = con.execute("SELECT any_value(feature_kinds) FROM rf_reg_fit('etr','y',n_trees:=1,max_depth:=1)").fetchone()[0]
        assert list(fk) == ["cat"]
        r = con.execute("""SELECT split_kind FROM rf_reg_fit('etr','y',n_trees:=1,max_depth:=1,
                           mtry:=1,replace_sample:=false,sample_frac:=1.0) WHERE node=1""").fetchone()
        assert r[0] == "cat"
        con.execute("""CREATE OR REPLACE TABLE ecl AS
            SELECT (i%5)::DOUBLE AS x,
                   (CASE WHEN i%2=0 THEN 'lo' ELSE 'hi' END)::mood AS y FROM range(80) t(i)""")
        cls = con.execute("SELECT any_value(classes) FROM rf_class_fit('ecl','y',n_trees:=1)").fetchone()[0]
        assert set(cls) == {"lo", "hi"}

    def test_categorical_outcome_with_categorical_features(self, con):
        # The headline use case: both features and outcome are categorical.
        rng = np.random.default_rng(7)
        n = 400
        color = rng.choice(["red", "green", "blue"], size=n)
        size = rng.choice(["S", "M", "L", "XL"], size=n)
        def label(c, s):
            base = {"red": 0, "green": 1, "blue": 2}[c] + {"S": 0, "M": 0, "L": 1, "XL": 1}[s]
            return ["catA", "catB", "catC", "catD"][base % 4]
        y = np.array([label(c, s) for c, s in zip(color, size)])
        df = pd.DataFrame({"color": color, "size": size, "y": y})
        _load(con, "tr", df)
        fit_tbl(con, "rf_class_fit", "tr", "m", n_trees=50, seed=7)
        fk = df_run(con, "SELECT any_value(feature_kinds) k FROM m").k[0]
        assert set(fk) == {"cat"}
        sc = pd.DataFrame({"color": color, "size": size, "rid": np.arange(n)})
        _load(con, "sc", sc)
        out = df_run(con, "SELECT rid, pred, probs FROM rf_class_predict('m','sc')").sort_values("rid")
        assert (out["pred"].to_numpy() == y).mean() > 0.9
        prow = out["probs"].iloc[0]
        assert set(prow.keys()) == {"catA", "catB", "catC", "catD"}
        assert abs(sum(prow.values()) - 1.0) < 1e-6

    def test_cats_left_right_partition_present_levels(self, con):
        rng = np.random.default_rng(3)
        for _ in range(40):
            L = rng.integers(2, 8)
            levels = np.array([f"g{i}" for i in rng.integers(0, L, int(rng.integers(L * 3, L * 8)))])
            for i, lv in enumerate([f"g{i}" for i in range(L)]):
                levels[i % len(levels)] = lv
            y = rng.normal(size=len(levels)) * 3 + 5
            r = self._stump(con, pd.DataFrame({"x": levels, "y": y}), "rf_reg_fit", "mse")
            if r[0] or r[1] != "cat":
                continue
            cl, cr = set(r[2]), set(r[3])
            present = set(np.unique(levels))
            assert not (cl & cr)
            assert (cl | cr) == present

    def test_unseen_level_at_predict_routes_not_null(self, con):
        rng = np.random.default_rng(1)
        tr = pd.DataFrame({"x": ["a"] * 40 + ["b"] * 40,
                           "y": list(rng.normal(0, 0.1, 40)) + list(rng.normal(5, 0.1, 40))})
        _load(con, "tr", tr)
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=10, max_depth=3, seed=1)
        _load(con, "sc", pd.DataFrame({"x": ["a", "b", "NEVERSEEN", "zzz"]}))
        pr = df_run(con, "SELECT x, prediction FROM rf_reg_predict('m','sc')")
        assert not pr["prediction"].isna().any()


# ===========================================================================
# 7. MDI importance vs sklearn feature_importances_
# ===========================================================================
class TestImportance:
    @pytest.mark.parametrize("cls,crit,depth", [(True, "gini", 3), (True, "entropy", 3),
                                                (False, "mse", 3)])
    def test_single_tree_importance_matches_sklearn(self, con, cls, crit, depth):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((300, 3))
        if cls:
            y = (X[:, 0] - 0.5 * X[:, 1] > 0).astype(int)
            df = frame(X).assign(y=[str(v) for v in y])
            macro = "rf_class_fit"
        else:
            y = X @ [2.0, -1.0, 0.5] + 0.3 * rng.standard_normal(300)
            df = frame(X).assign(y=y)
            macro = "rf_reg_fit"
        _load(con, "tr", df)
        fit_tbl(con, macro, "tr", "m", mtry=3, max_depth=depth, criterion=f"'{crit}'", **SINGLE)
        imp = df_run(con, "SELECT feature, importance FROM rf_importance('m') ORDER BY feature")
        duckv = imp["importance"].to_numpy()
        if cls:
            sk = DecisionTreeClassifier(criterion=crit, max_depth=depth, max_features=None,
                                        random_state=0).fit(X, [str(v) for v in y])
        else:
            sk = DecisionTreeRegressor(max_depth=depth, max_features=None,
                                       random_state=0).fit(X, y)
        assert abs(duckv.sum() - 1.0) < 1e-9
        assert np.max(np.abs(duckv - sk.feature_importances_)) < 1e-9

    def test_importance_lists_every_feature_including_unused(self, con):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((300, 4))
        y = X[:, 0] * 2 + 0.1 * rng.standard_normal(300)  # only x0 informative
        X[:, 3] = 5.0  # x3 constant -> can never be used
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=20, seed=1, max_depth=5)
        imp = df_run(con, "SELECT feature, importance FROM rf_importance('m')")
        assert set(imp["feature"]) == {"x0", "x1", "x2", "x3"}
        assert abs(imp["importance"].sum() - 1.0) < 1e-9
        assert imp.set_index("feature").loc["x3", "importance"] == 0.0


# ===========================================================================
# 8. Out-of-bag: numpy membership replay, near-holdout score, guards
# ===========================================================================
import hashlib
import math


def _md5num(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest(), "little")


def _bags_replace(seed, n_trees, n, m):
    return [set((_md5num(f"{seed}:{t}:{k}") % n) + 1 for k in range(1, m + 1))
            for t in range(1, n_trees + 1)]


class TestOOB:
    def test_oob_predictions_match_numpy_bootstrap_replay(self, con):
        rng = np.random.default_rng(0)
        n, seed, NT = 400, 42, 30
        X = rng.normal(size=(n, 4))
        y = X[:, 0] * 2 - X[:, 1] + rng.normal(0, 0.3, n) + 100.0
        _load(con, "tr", frame(X, ["a", "b", "c", "d"]).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=NT, seed=seed, max_depth=8)
        pt = df_run(con, "SELECT __rf_rid__ rid, tree, prediction FROM rf_reg_predict_trees('m','tr')")
        pred = {}
        for r in pt.itertuples():
            pred.setdefault(int(r.rid), {})[int(r.tree)] = r.prediction
        bags = _bags_replace(seed, NT, n, max(1, math.ceil(1.0 * n)))
        manual, excluded = {}, []
        for rid in range(1, n + 1):
            trees = [t + 1 for t in range(NT) if rid not in bags[t]]
            (manual.__setitem__(rid, np.mean([pred[rid][t] for t in trees]))
             if trees else excluded.append(rid))
        oob = df_run(con, "SELECT row_number() OVER () rid, prediction FROM rf_reg_oob_predict('m','tr')")
        duck = {int(r.rid): r.prediction for r in oob.itertuples()}
        maxd, nullmis = 0.0, 0
        for rid in range(1, n + 1):
            dv = duck.get(rid)
            isnull = dv is None or (isinstance(dv, float) and math.isnan(dv))
            if rid in manual:
                nullmis += isnull
                maxd = max(maxd, 0 if isnull else abs(dv - manual[rid]))
            else:
                nullmis += not isnull
        assert maxd < 1e-9 and nullmis == 0
        n_excl = df_run(con, "SELECT n_excluded FROM rf_reg_oob('m','tr','y')").n_excluded[0]
        assert int(n_excl) == len(excluded)

    def test_oob_score_near_holdout(self, con):
        rng = np.random.default_rng(0)
        n = 400
        X = rng.normal(size=(n, 4))
        y = X[:, 0] * 2 - X[:, 1] + rng.normal(0, 0.3, n) + 100.0
        df = frame(X, ["a", "b", "c", "d"]).assign(y=y)
        _load(con, "trn", df.iloc[:280])
        _load(con, "hol", df.iloc[280:])
        fit_tbl(con, "rf_reg_fit", "trn", "m", n_trees=100, seed=1, max_depth=10)
        oob = df_run(con, "SELECT rmse FROM rf_reg_oob('m','trn','y')").rmse[0]
        hol = df_run(con, "SELECT rmse FROM rf_reg_evaluate('m','hol','y')").rmse[0]
        assert abs(oob - hol) / hol < 0.25

    def test_oob_rejects_filtered_table(self, con):
        _load(con, "tr", frame(np.random.default_rng(1).normal(size=(200, 3))).assign(
            y=np.random.default_rng(2).normal(size=200)))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=10, seed=1)
        con.execute("CREATE OR REPLACE TABLE trsub AS SELECT * FROM tr LIMIT 100")
        with pytest.raises(DuckDBError):
            df_run(con, "SELECT * FROM rf_reg_oob_predict('m','trsub')")

    def test_classification_oob_n_excluded_matches_numpy(self, con):
        rng = np.random.default_rng(0)
        n, seed, NT = 400, 42, 25
        X = rng.normal(size=(n, 4))
        lin = X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
        y = np.where(lin > 0.3, "hi", np.where(lin < -0.3, "lo", "mid"))
        _load(con, "tr", frame(X, ["a", "b", "c", "d"]).assign(y=y))
        fit_tbl(con, "rf_class_fit", "tr", "m", n_trees=NT, seed=seed, max_depth=8)
        bags = _bags_replace(seed, NT, n, n)
        excl = [rid for rid in range(1, n + 1) if all(rid in bags[t] for t in range(NT))]
        n_excl = df_run(con, "SELECT n_excluded FROM rf_class_oob('m','tr','y')").n_excluded[0]
        assert int(n_excl) == len(excl)


# ===========================================================================
# 9. Determinism (threads=1) and RNG replay
# ===========================================================================
class TestDeterminism:
    def test_same_seed_identical_model(self, con):
        _load(con, "t", frame(np.random.default_rng(1).normal(size=(300, 4))).assign(
            y=np.random.default_rng(2).normal(size=300)))
        fit_tbl(con, "rf_reg_fit", "t", "ma", n_trees=15, seed=42, max_depth=6)
        fit_tbl(con, "rf_reg_fit", "t", "mb", n_trees=15, seed=42, max_depth=6)
        d = df_run(con, "SELECT count(*) c FROM ((SELECT * FROM ma EXCEPT SELECT * FROM mb) "
                        "UNION ALL (SELECT * FROM mb EXCEPT SELECT * FROM ma))").c[0]
        assert d == 0

    def test_different_seed_differs(self, con):
        _load(con, "t", frame(np.random.default_rng(3).normal(size=(300, 4))).assign(
            y=np.random.default_rng(4).normal(size=300)))
        fit_tbl(con, "rf_reg_fit", "t", "ma", n_trees=15, seed=42, max_depth=6)
        fit_tbl(con, "rf_reg_fit", "t", "mb", n_trees=15, seed=43, max_depth=6)
        d = df_run(con, "SELECT count(*) c FROM (SELECT * FROM ma EXCEPT SELECT * FROM mb)").c[0]
        assert d > 0

    def test_single_cart_is_row_order_invariant(self, con):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(300, 4))
        y = X @ [2.0, -1.0, 0.5, 1.3] + rng.normal(0, 0.3, 300) + 100.0
        df = frame(X, ["a", "b", "c", "d"]).assign(y=y)
        _load(con, "o1", df)
        _load(con, "o2", df.sort_values("y", ascending=False))
        q = ("SELECT node,is_leaf,split_feature,round(threshold,9) th,round(prediction,9) p,n_rows "
             "FROM rf_reg_fit('{}','y', n_trees:=1, mtry:=4, sample_frac:=1.0, "
             "replace_sample:=false, seed:=42) ORDER BY node")
        a = df_run(con, q.format("o1")).to_json()
        b = df_run(con, q.format("o2")).to_json()
        assert a == b

    def test_growing_n_trees_keeps_first_trees_identical(self, con):
        _load(con, "t", frame(np.random.default_rng(6).normal(size=(400, 4))).assign(
            y=np.random.default_rng(7).normal(size=400)))
        small = df_run(con, "SELECT * FROM rf_reg_fit('t','y', n_trees:=5, seed:=42) WHERE tree<=5 ORDER BY tree,node")
        big = df_run(con, "SELECT * FROM rf_reg_fit('t','y', n_trees:=20, seed:=42) WHERE tree<=5 ORDER BY tree,node")
        cols = [c for c in small.columns if c != "n_trees"]
        assert small[cols].to_json() == big[cols].to_json()

    def test_bootstrap_draw_matches_md5_replay(self, con):
        # duckdb's md5_number % n must equal the documented numpy replay bit-for-bit.
        inputs = [f"42:{t}:{k}" for t in range(1, 6) for k in range(1, 50)]
        duck = con.execute(
            "SELECT s, (md5_number(s) % 400::UHUGEINT)::BIGINT r FROM (SELECT unnest(?::VARCHAR[]) s)",
            [inputs]).df()
        dm = dict(zip(duck.s, duck.r))
        assert all(dm[s] == _md5num(s) % 400 for s in inputs)


# ===========================================================================
# 10. Sample weights and class_weight := 'balanced'
# ===========================================================================
class TestWeights:
    def test_constant_weight_equals_unweighted(self, con):
        # A constant weight column must reproduce the unweighted fit exactly.
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 4))
        y = X @ [2.0, -1.0, 0.5, 1.3] + rng.normal(0, 0.3, 300)
        _load(con, "tr0", frame(X).assign(y=y))
        _load(con, "trw", frame(X).assign(w=3.0, y=y))
        fit_tbl(con, "rf_reg_fit", "tr0", "m0", mtry=4, max_depth=5, **SINGLE)
        fit_tbl(con, "rf_reg_fit", "trw", "mw", mtry=4, max_depth=5, weights_col="'w'", **SINGLE)
        _load(con, "sc", frame(X).assign(rid=np.arange(300)))
        a = df_run(con, "SELECT rid, prediction FROM rf_reg_predict('m0','sc')").sort_values("rid")["prediction"].to_numpy()
        b = df_run(con, "SELECT rid, prediction FROM rf_reg_predict('mw','sc')").sort_values("rid")["prediction"].to_numpy()
        assert np.max(np.abs(a - b)) < 1e-12

    def test_weighted_regression_matches_sklearn(self, con):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(400, 4))
        y = X @ [1.5, -1.0, 0.4, 0.8] + rng.normal(0, 0.3, 400)
        w = rng.uniform(0.2, 5, 400)
        _load(con, "trw", frame(X).assign(w=w, y=y))
        fit_tbl(con, "rf_reg_fit", "trw", "m", mtry=4, max_depth=4, weights_col="'w'", **SINGLE)
        _load(con, "sc", frame(X).assign(rid=np.arange(400)))
        duck = df_run(con, "SELECT rid, prediction FROM rf_reg_predict('m','sc')").sort_values("rid")["prediction"].to_numpy()
        sk = DecisionTreeRegressor(max_depth=4, max_features=None, random_state=0).fit(X, y, sample_weight=w)
        # exact where the weighted split search is unambiguous; never worse otherwise
        assert (np.max(np.abs(duck - sk.predict(X))) < 1e-9
                or _sse(y, duck) <= _sse(y, sk.predict(X)) * (1 + 1e-9) + 1e-6)

    def test_class_weight_balanced_matches_sklearn(self, con):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(500, 4))
        # imbalanced 3-class target
        s = X[:, 0] + 0.5 * X[:, 1]
        y = np.where(s > 1.0, "rare", np.where(s > -0.2, "mid", "common"))
        # duckRF's 'balanced' uses sklearn's exact n/(K*n_k); check the reweighted
        # tree is never worse than sklearn's on the SAME balanced weighting.
        lab, p, sk, classes = cart_cls(con, X, y, "gini", 4, weight="balanced",
                                       class_weight="'balanced'")
        ystr = np.array([str(v) for v in y])
        bincount = {c: (ystr == c).sum() for c in classes}
        w = np.array([len(ystr) / (len(classes) * bincount[c]) for c in ystr])
        acc_duck = np.average(lab == ystr, weights=w)
        acc_sk = np.average(sk.predict(X) == ystr, weights=w)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
        assert acc_duck >= acc_sk - 1e-12


# ===========================================================================
# 11. Cross-validation tuning
# ===========================================================================
class TestCV:
    def test_rf_cv_regression_grid(self, con):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 4))
        y = X @ [2.0, -1.0, 0.5, 1.3] + rng.normal(0, 0.3, 300)
        _load(con, "tr", frame(X).assign(y=y))
        cv = df_run(con, "SELECT * FROM rf_cv('tr','y','regression',[1,2,4], k:=4, n_trees:=15, seed:=1)")
        assert len(cv) == 3 and np.isfinite(cv["cv_error"]).all()
        assert set(cv["mtry"]) == {1, 2, 4}

    def test_rf_cv_classification_error_in_unit_interval(self, con):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(300, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(str)
        _load(con, "tc", frame(X).assign(y=y))
        cv = df_run(con, "SELECT * FROM rf_cv('tc','y','classification',[1,2], k:=4, n_trees:=15, seed:=1)")
        assert len(cv) == 2 and ((cv["cv_error"] >= 0) & (cv["cv_error"] <= 1)).all()

    def test_rf_cv_depth_grid(self, con):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(300, 4))
        y = X @ [2.0, -1.0, 0.5, 1.3] + rng.normal(0, 0.3, 300)
        _load(con, "tr", frame(X).assign(y=y))
        cv = df_run(con, "SELECT * FROM rf_cv_depth('tr','y','regression',[2,5], k:=4, n_trees:=15, seed:=1)")
        assert len(cv) == 2 and np.isfinite(cv["cv_error"]).all()
        assert set(cv["max_depth"]) == {2, 5}

    def test_rf_cv_k_guard(self, con):
        _load(con, "tc", frame(np.random.default_rng(4).normal(size=(100, 3))).assign(
            y=(np.random.default_rng(5).normal(size=100) > 0).astype(str)))
        with pytest.raises(DuckDBError, match="k must be"):
            df_run(con, "SELECT * FROM rf_cv('tc','y','classification',[1],k:=1)")

    def test_rf_cv_illegal_family(self, con):
        _load(con, "tc", frame(np.random.default_rng(6).normal(size=(100, 3))).assign(
            y=(np.random.default_rng(7).normal(size=100) > 0).astype(str)))
        with pytest.raises(DuckDBError, match="family must be"):
            df_run(con, "SELECT * FROM rf_cv('tc','y','xx',[1,2],k:=3,n_trees:=3)")


# ===========================================================================
# 12. rf_summary shape
# ===========================================================================
class TestSummary:
    def test_summary_reports_forest_shape(self, con):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 4))
        y = X @ [2.0, -1.0, 0.5, 1.3] + rng.normal(0, 0.3, 300)
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=20, seed=1, max_depth=6)
        s = df_run(con, "SELECT * FROM rf_summary('m')").iloc[0]
        assert s["family"] == "regression"
        assert s["n_trees"] == 20
        assert s["n_nodes"] > 0 and s["n_leaves"] > 0
        assert s["n_leaves"] < s["n_nodes"]
        assert s["max_depth_reached"] <= 6

    def test_depth_cap_hit_flag(self, con):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(400, 4))
        y = X @ [2.0, -1.0, 0.5, 1.3] + rng.normal(0, 0.3, 400)
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=5, seed=1, max_depth=2)
        s = df_run(con, "SELECT depth_cap_hit, max_depth_reached FROM rf_summary('m')").iloc[0]
        assert bool(s["depth_cap_hit"]) and s["max_depth_reached"] == 2


# ===========================================================================
# 13. Contract: NULLs, degenerate inputs, reserved names, guards
#     (the heart of the suite -- every edge exercised against real DuckDB)
# ===========================================================================
def _xor(seed, n=200, cols=("a", "b")):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    return pd.DataFrame({cols[0]: a, cols[1]: b, "y": ((a > 0) ^ (b > 0)).astype(int)})


def _reg(seed, n=200):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 3))
    d = frame(X, ["a", "b", "cc"])
    d["y"] = X[:, 0] * 2 - X[:, 1] + rng.normal(0, 0.3, n)
    return d


class TestNullContract:
    def test_null_outcomes_dropped(self, con):
        d = _xor(1); d.loc[[0, 1, 2], "y"] = np.nan
        _load(con, "t", d)
        assert df_run(con, "SELECT any_value(n_train) v FROM rf_class_fit('t','y',n_trees:=3)").v[0] == len(d) - 3

    def test_null_feature_cells_dropped(self, con):
        d = _xor(2); d.loc[[0, 1, 2], "a"] = np.nan
        _load(con, "t", d)
        assert df_run(con, "SELECT any_value(n_train) v FROM rf_class_fit('t','y',n_trees:=3)").v[0] == len(d) - 3

    def test_all_null_row_dropped(self, con):
        d = _xor(3); d.loc[5, :] = np.nan
        _load(con, "t", d)
        assert df_run(con, "SELECT any_value(n_train) v FROM rf_class_fit('t','y',n_trees:=3)").v[0] == len(d) - 1

    def test_entirely_null_feature_errors(self, con):
        d = _xor(4); d["a"] = np.nan
        _load(con, "t", d)
        with pytest.raises(DuckDBError, match="entirely NULL"):
            df_run(con, "SELECT * FROM rf_class_fit('t','y')")

    def test_nan_inf_feature_errors(self, con):
        d = _xor(6).astype({"y": int}); d.loc[0, "a"] = np.inf
        _load(con, "t", d)
        with pytest.raises(DuckDBError, match="NaN or Inf"):
            df_run(con, "SELECT * FROM rf_class_fit('t','y')")

    def test_nan_inf_reg_outcome_errors(self, con):
        d = _reg(7); d.loc[0, "y"] = np.inf
        _load(con, "t", d)
        with pytest.raises(DuckDBError, match="NaN or Inf"):
            df_run(con, "SELECT * FROM rf_reg_fit('t','y')")


class TestDegenerate:
    def test_empty_table_errors(self, con):
        _load(con, "t", _xor(10))
        con.execute("CREATE OR REPLACE TABLE te AS SELECT * FROM t WHERE false")
        with pytest.raises(DuckDBError, match="is empty"):
            df_run(con, "SELECT * FROM rf_class_fit('te','y')")

    def test_predict_empty_returns_zero_rows(self, con):
        _load(con, "t", _xor(11))
        fit_tbl(con, "rf_class_fit", "t", "m", n_trees=5)
        con.execute("CREATE OR REPLACE TABLE te AS SELECT * FROM t WHERE false")
        assert len(df_run(con, "SELECT * FROM rf_class_predict('m','te')")) == 0

    def test_single_row_class_single_class_errors(self, con):
        _load(con, "t", _xor(20).iloc[[0]])
        with pytest.raises(DuckDBError, match="single class"):
            df_run(con, "SELECT * FROM rf_class_fit('t','y')")

    def test_single_row_reg_is_leaf(self, con):
        _load(con, "t", _reg(21).iloc[[0]])
        assert df_run(con, "SELECT is_leaf FROM rf_reg_fit('t','y',n_trees:=3)").is_leaf.all()

    def test_single_feature_gets_mtry_one(self, con):
        _load(con, "t", _xor(22)[["a", "y"]])
        assert df_run(con, "SELECT any_value(mtry) v FROM rf_class_fit('t','y',n_trees:=3)").v[0] == 1

    def test_single_constant_feature_all_stumps(self, con):
        d = _xor(24)[["a", "y"]]; d["a"] = 3.0
        _load(con, "t", d)
        out = df_run(con, "SELECT count(*) c, sum(is_leaf::int) l FROM rf_class_fit('t','y',n_trees:=5)")
        assert out.c[0] == out.l[0]

    def test_constant_reg_outcome_predicts_constant(self, con):
        d = _reg(25); d["y"] = 4.2
        _load(con, "t", d)
        fit_tbl(con, "rf_reg_fit", "t", "m", n_trees=5)
        preds = df_run(con, "SELECT DISTINCT round(prediction,9) p FROM rf_reg_predict('m','t')")
        assert list(preds.p) == [4.2]

    def test_one_column_table_errors(self, con):
        _load(con, "t", _xor(52)[["y"]])
        with pytest.raises(DuckDBError, match="no feature columns"):
            df_run(con, "SELECT * FROM rf_class_fit('t','y')")

    def test_bad_column_type_date_errors(self, con):
        con.execute("""CREATE OR REPLACE TABLE t AS
            SELECT i::DOUBLE x, DATE '2020-01-01' + i::INTEGER dt, i*1.0 y FROM range(10) t(i)""")
        with pytest.raises(DuckDBError, match="DATE"):
            df_run(con, "SELECT * FROM rf_reg_fit('t','y')")

    def test_non_numeric_reg_outcome_errors(self, con):
        con.execute("CREATE OR REPLACE TABLE t AS SELECT i::DOUBLE x, 'lab'||(i%3) y FROM range(10) t(i)")
        with pytest.raises(DuckDBError, match="must be numeric"):
            df_run(con, "SELECT * FROM rf_reg_fit('t','y')")


class TestReservedNames:
    def test_reserved_table_name(self, con):
        _load(con, "__rf_x", _xor(30))
        with pytest.raises(DuckDBError, match="reserved"):
            df_run(con, "SELECT * FROM rf_class_fit('__rf_x','y')")

    def test_reserved_column_name(self, con):
        _load(con, "t", _xor(31).rename(columns={"a": "__rf_a"}))
        with pytest.raises(DuckDBError, match="reserved"):
            df_run(con, "SELECT * FROM rf_class_fit('t','y')")

    def test_output_name_collision_rejected_at_fit(self, con):
        _load(con, "t", _xor(40).rename(columns={"a": "pred"}))
        with pytest.raises(DuckDBError, match="collides"):
            fit_tbl(con, "rf_class_fit", "t", "m", n_trees=5)

    def test_reserved_passthrough_column_at_predict_rejected(self, con):
        _load(con, "t", _xor(62))
        fit_tbl(con, "rf_class_fit", "t", "m", n_trees=3)
        con.execute("CREATE OR REPLACE TABLE ts AS SELECT *, row_number() OVER () __rf_note FROM t")
        with pytest.raises(DuckDBError, match="reserved"):
            df_run(con, "SELECT * FROM rf_class_predict('m','ts')")


class TestTableForms:
    def test_schema_qualified_table(self, con):
        _load(con, "t", _xor(50))
        con.execute("CREATE SCHEMA IF NOT EXISTS s1")
        con.execute("CREATE OR REPLACE TABLE s1.tt AS SELECT * FROM t")
        assert df_run(con, "SELECT count(*) c FROM rf_class_fit('s1.tt','y',n_trees:=3)").c[0] > 0

    def test_fit_and_predict_on_view(self, con):
        _load(con, "t", _xor(51))
        con.execute("CREATE OR REPLACE VIEW v AS SELECT * FROM t")
        fit_tbl(con, "rf_class_fit", "v", "m", n_trees=3)
        assert df_run(con, "SELECT count(*) c FROM rf_class_predict('m','v')").c[0] == 200


class TestPredictContract:
    def test_missing_feature_column_yields_null(self, con):
        _load(con, "t", _xor(60))
        fit_tbl(con, "rf_class_fit", "t", "m", n_trees=5)
        con.execute("CREATE OR REPLACE TABLE tmiss AS SELECT a, y FROM t")
        out = df_run(con, "SELECT count(*) tot, count(pred) hp FROM rf_class_predict('m','tmiss')")
        assert out.tot[0] == 200 and out.hp[0] == 0

    def test_extra_columns_passthrough_and_order(self, con):
        d = _xor(61); d.insert(0, "id", range(len(d))); d["label"] = [f"r{i}" for i in range(len(d))]
        _load(con, "t", d)
        fit_tbl(con, "rf_class_fit", "t", "m", n_trees=5)
        out = df_run(con, "SELECT id, label, pred FROM rf_class_predict('m','t')")
        assert out.id.tolist() == list(range(len(d)))
        assert out.label.tolist() == d.label.tolist()


class TestModelContract:
    def _cls_model(self, con, seed=70):
        _load(con, "t", _xor(seed))
        fit_tbl(con, "rf_class_fit", "t", "m", n_trees=10)

    def test_empty_model_errors(self, con):
        self._cls_model(con)
        con.execute("CREATE OR REPLACE TABLE mempty AS SELECT * FROM m WHERE false")
        with pytest.raises(DuckDBError, match="empty"):
            df_run(con, "SELECT * FROM rf_class_predict('mempty','t')")

    def test_family_mismatch_errors(self, con):
        self._cls_model(con)
        _load(con, "tr", _reg(71))
        fit_tbl(con, "rf_reg_fit", "tr", "mr", n_trees=5)
        with pytest.raises(DuckDBError, match="regression forest"):
            df_run(con, "SELECT * FROM rf_class_predict('mr','t')")
        with pytest.raises(DuckDBError, match="classification forest"):
            df_run(con, "SELECT * FROM rf_reg_predict('m','tr')")

    def test_evaluate_nonexistent_outcome_errors(self, con):
        self._cls_model(con, 90)
        with pytest.raises(DuckDBError, match="no rows"):
            df_run(con, "SELECT * FROM rf_class_evaluate('m','t','nope')")


class TestParamGuards:
    @pytest.fixture(autouse=True)
    def _data(self, con):
        _load(con, "t", _xor(110))
        _load(con, "tr", _reg(111))

    @pytest.mark.parametrize("kw", ["n_trees:=1", "mtry:=1", "mtry:=2", "max_depth:=1",
                                    "max_depth:=60", "max_depth:=NULL", "min_samples_split:=2",
                                    "sample_frac:=0.01"])
    def test_boundary_params_ok(self, con, kw):
        assert df_run(con, f"SELECT count(*) c FROM rf_class_fit('t','y',{kw})").c[0] > 0

    @pytest.mark.parametrize("kw,msg", [
        ("n_trees:=0", "n_trees must be"),
        ("mtry:=0", "mtry must be"),
        ("mtry:=99", "mtry must be"),
        ("max_depth:=0", "max_depth must be"),
        ("max_depth:=61", "max_depth must be"),
        ("min_samples_split:=1", "min_samples_split must be"),
        ("min_samples_leaf:=0", "min_samples_leaf must be"),
        ("min_impurity_decrease:=-1", "min_impurity_decrease must be"),
        ("sample_frac:=0", "sample_frac must be"),
        ("sample_frac:=1.5", "sample_frac must be"),
        ("criterion:='xx'", "criterion must be"),
        ("seed:=NULL", "seed must not be NULL"),
        ("class_weight:='x'", "class_weight must be"),
        ("weights_col:='nope'", "weights column"),
    ])
    def test_illegal_params_error(self, con, kw, msg):
        with pytest.raises(DuckDBError, match=msg):
            df_run(con, f"SELECT * FROM rf_class_fit('t','y',{kw})")

    def test_reg_criterion_gini_illegal(self, con):
        with pytest.raises(DuckDBError, match="criterion must be 'mse'"):
            df_run(con, "SELECT * FROM rf_reg_fit('tr','y',criterion:='gini')")


# ===========================================================================
# 14. Permutation importance vs sklearn.inspection.permutation_importance
#
# The cardinality-unbiased complement to MDI: shuffle a feature's column, watch
# the model's score (R^2 / accuracy) drop. duckRF's permutation is md5-seeded, not
# numpy's, so it matches sklearn STATISTICALLY (rank / top-k), not to 1e-9 -- but
# for the SAME md5 permutation replayed in numpy the per-repeat number is pinned
# exactly, isolating the only difference from sklearn to the RNG.
# ===========================================================================
def _perm_replay_column(values, feat, rep, seed):
    """Replicate duckRF's md5 shuffle for (feat, rep): the row at destination
    ordinal i (1..n) receives the value from the row whose md5 key ranks i-th."""
    n = len(values)
    keys = [(_md5num(f"{seed}:P:{feat}:{rep}:{i}"), i) for i in range(1, n + 1)]
    order = sorted(range(n), key=lambda p: keys[p])
    return np.asarray(values)[order]


class TestPermutationImportance:
    def _perm(self, con, model, tbl, outcome, n_repeats, seed=42):
        return df_run(con, f"SELECT feature, importance, importance_std FROM "
                           f"rf_permutation_importance('{model}','{tbl}','{outcome}',"
                           f" n_repeats:={n_repeats}, seed:={seed})")

    def test_signal_outranks_noise_regression(self, con):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((400, 5))
        y = 3 * X[:, 0] - 2 * X[:, 1] + 0.3 * rng.standard_normal(400)  # x2..x4 noise
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=40, seed=1, max_depth=12)
        p = self._perm(con, "m", "tr", "y", 12).set_index("feature")["importance"]
        assert min(p["x0"], p["x1"]) > max(p["x2"], p["x3"], p["x4"])
        assert max(abs(p["x2"]), abs(p["x3"]), abs(p["x4"])) < 0.05
        # importance_std is non-negative
        s = self._perm(con, "m", "tr", "y", 12).set_index("feature")["importance_std"]
        assert (s >= 0).all()

    def test_signal_outranks_noise_classification(self, con):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 5))
        y = np.where(2.5 * X[:, 0] - 1.5 * X[:, 1] > 0, "pos", "neg")
        _load(con, "tc", frame(X).assign(y=y))
        fit_tbl(con, "rf_class_fit", "tc", "mc", n_trees=40, seed=1, max_depth=12)
        p = self._perm(con, "mc", "tc", "y", 12).set_index("feature")["importance"]
        assert min(p["x0"], p["x1"]) > max(p["x2"], p["x3"], p["x4"])
        assert max(abs(p["x2"]), abs(p["x3"]), abs(p["x4"])) < 0.03

    def test_rank_correlates_with_sklearn(self, con):
        # graded signal on every feature -> a well-defined ranking to correlate.
        rng = np.random.default_rng(42)
        n, d = 500, 6
        X = rng.standard_normal((n, d))
        y = X @ np.array([3.0, 2.0, 1.2, 0.7, 0.4, 0.2]) + 0.3 * rng.standard_normal(n)
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=60, seed=7, max_depth=14)
        NR = 25
        duck = (self._perm(con, "m", "tr", "y", NR).set_index("feature")["importance"]
                .reindex([f"x{i}" for i in range(d)]).to_numpy())
        sk = RandomForestRegressor(n_estimators=60, max_depth=14, random_state=7).fit(X, y)
        skimp = permutation_importance(sk, X, y, n_repeats=NR, random_state=0).importances_mean
        assert spearmanr(duck, skimp).statistic >= 0.9
        assert set(np.argsort(duck)[::-1][:3]) == set(np.argsort(skimp)[::-1][:3])
        # and rank-agrees with duckRF's own MDI on this unbiased (all-numeric) data
        mdi = (df_run(con, "SELECT feature, importance FROM rf_importance('m')")
               .set_index("feature")["importance"].reindex([f"x{i}" for i in range(d)]).to_numpy())
        assert spearmanr(duck, mdi).statistic >= 0.9

    def test_definition_replay_pins_per_repeat(self, con):
        # SAME md5 permutation replayed in numpy, scored via rf_reg_predict, must
        # equal duckRF's per-repeat (n_repeats=1) importance to ~1e-9.
        SEED = 12345
        rng = np.random.default_rng(1)
        X = rng.standard_normal((300, 4))
        y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + 0.3 * rng.standard_normal(300)
        df = frame(X).assign(y=y)
        _load(con, "tr", df)
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=20, seed=5, max_depth=10)
        duck1 = self._perm(con, "m", "tr", "y", 1, seed=SEED).set_index("feature")["importance"]
        yhat = df_run(con, "SELECT prediction FROM rf_reg_predict('m','tr')")["prediction"].to_numpy()
        yv = df["y"].to_numpy()
        sst = float(np.sum((yv - yv.mean()) ** 2))
        r2_base = 1.0 - float(np.sum((yv - yhat) ** 2)) / sst
        for j, feat in enumerate([f"x{i}" for i in range(4)]):
            dfp = df.copy()
            dfp[feat] = _perm_replay_column(X[:, j], feat, 1, SEED)
            _load(con, "tp", dfp)
            yhp = df_run(con, "SELECT prediction FROM rf_reg_predict('m','tp')")["prediction"].to_numpy()
            r2_p = 1.0 - float(np.sum((yv - yhp) ** 2)) / sst
            assert abs((r2_base - r2_p) - duck1[feat]) < 1e-9

    def test_deterministic_same_seed(self, con):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((300, 4))
        y = 2 * X[:, 0] - X[:, 1] + 0.3 * rng.standard_normal(300)
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=30, seed=1, max_depth=10)
        a = self._perm(con, "m", "tr", "y", 10, seed=42)
        b = self._perm(con, "m", "tr", "y", 10, seed=42)
        assert a.equals(b)  # bit-identical under threads=1
        d = self._perm(con, "m", "tr", "y", 10, seed=43)
        assert (np.abs(a["importance"].to_numpy() - d["importance"].to_numpy()) > 1e-12).any()

    def test_lists_every_feature_including_constant(self, con):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((300, 4))
        y = 2 * X[:, 0] - X[:, 1] + 0.3 * rng.standard_normal(300)
        X[:, 3] = 7.0  # constant -> never split on -> importance exactly 0
        _load(con, "tr", frame(X).assign(y=y))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=20, seed=1, max_depth=8)
        p = self._perm(con, "m", "tr", "y", 8).set_index("feature")["importance"]
        assert set(p.index) == {"x0", "x1", "x2", "x3"}
        assert p["x3"] == 0.0

    def test_works_on_holdout_table(self, con):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((400, 4))
        y = 3 * X[:, 0] - X[:, 1] + 0.3 * rng.standard_normal(400)
        df = frame(X).assign(y=y)
        _load(con, "trn", df.iloc[:300])
        _load(con, "hol", df.iloc[300:])
        fit_tbl(con, "rf_reg_fit", "trn", "m", n_trees=40, seed=1, max_depth=10)
        p = self._perm(con, "m", "hol", "y", 10).set_index("feature")["importance"]
        assert p["x0"] > p["x2"] and p["x0"] > p["x3"]

    def test_unknown_outcome_errors(self, con):
        _load(con, "tr", _reg(80))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=5)
        with pytest.raises(DuckDBError, match="not found"):
            df_run(con, "SELECT * FROM rf_permutation_importance('m','tr','nope')")

    def test_n_repeats_zero_errors(self, con):
        _load(con, "tr", _reg(81))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=5)
        with pytest.raises(DuckDBError, match="n_repeats must be"):
            df_run(con, "SELECT * FROM rf_permutation_importance('m','tr','y', n_repeats:=0)")

    def test_reserved_table_name_errors(self, con):
        d = _reg(82)
        _load(con, "tr", d)
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=5)
        _load(con, "__rf_bad", d)
        with pytest.raises(DuckDBError, match="reserved"):
            df_run(con, "SELECT * FROM rf_permutation_importance('m','__rf_bad','y')")

    def test_empty_model_errors(self, con):
        _load(con, "tr", _reg(83))
        fit_tbl(con, "rf_reg_fit", "tr", "m", n_trees=5)
        con.execute("CREATE OR REPLACE TABLE mempty AS SELECT * FROM m WHERE false")
        with pytest.raises(DuckDBError, match="empty"):
            df_run(con, "SELECT * FROM rf_permutation_importance('mempty','tr','y')")
