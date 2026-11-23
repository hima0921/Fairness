"""
Classical bias detectors on the fixed 1158-number feature vectors (proposal RQ2):
logistic regression, random forest and XGBoost, all trained on the same features.

Features come from src/features/featurize.py (or the Colab notebook 03; same files).
Settings:
- Hyperparameters are tuned with 5-fold CV on the train split only. Folds are
  grouped by template_id (no template is in both the fit and score part of a fold)
  and stratified by category, so every fold has some of each category.
- The tuning score is average precision (PR-AUC). Only ~25% of rows are biased.
- Class imbalance: class_weight="balanced" (LR, RF) and scale_pos_weight (XGB).
- The 0.5 cutoff is reported for comparison with docs/detector_results.md. A second
  cutoff (max F1) is picked on val and then applied unchanged to test.

Feature sets (all built from the same 1158 columns):
- nli_sim:    the 3 similarities + 3 NLI probabilities (6 numbers)
- emb:        the 3 embeddings only (1152)
- full:       everything (1158), as in the proposal
- full_inter: full + answer*question, |answer-question|, answer*context, |answer-context|
              (2694). A linear model on [c; q; a] adds the answer's effect the same way
              for every question, so it cannot learn "this answer is biased for THIS
              question". The product/difference terms give it that interaction.

Run from the repo root (after building features):
    .venv/bin/python -m src.detector.train_classical                      # everything
    .venv/bin/python -m src.detector.train_classical --models lr --sets full
Outputs (gitignored): outputs/classical/<set>_<model>/{val,test}_preds.csv, report.json,
model.joblib, and outputs/classical/summary.csv.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.detector.metrics import binary_metrics, breakdown, signal_flag_rates
from src.features.featurize import ANS, CTX, NLI, QUE, SIMS, load_features

FEATURE_SETS = ["nli_sim", "emb", "full", "full_inter"]
MODELS = ["lr", "rf", "xgb"]
PRED_COLS = ["row_id", "question_id", "template_id", "category", "context_condition",
             "question_polarity", "answer_group", "is_correct", "is_unknown",
             "anti_stereotype", "is_biased"]


def build_features(X, name):
    """Column subsets / interaction terms of the 1158-wide matrix."""
    if name == "nli_sim":
        return np.hstack([X[:, SIMS], X[:, NLI]])
    if name == "emb":
        return np.hstack([X[:, CTX], X[:, QUE], X[:, ANS]])
    if name == "full":
        return X
    if name == "full_inter":
        c, q, a = X[:, CTX], X[:, QUE], X[:, ANS]
        return np.hstack([X, a * q, np.abs(a - q), a * c, np.abs(a - c)]).astype(np.float32)
    raise ValueError(name)


def model_and_grid(name, pos_weight, seed, threads):
    """Estimator + search space. Ranges are common practice, not tuned recommendations."""
    if name == "lr":
        est = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=3000,
                                       random_state=seed)),
        ])
        grid = {"clf__C": np.logspace(-4, 1, 6).tolist()}
        return est, grid
    if name == "rf":
        est = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     n_jobs=threads, random_state=seed)
        grid = {"min_samples_leaf": [1, 5, 20], "max_features": ["sqrt", 0.1], "max_depth": [None, 20]}
        return est, grid
    if name == "xgb":
        est = XGBClassifier(tree_method="hist", objective="binary:logistic", eval_metric="aucpr",
                            scale_pos_weight=pos_weight, n_jobs=threads, random_state=seed,
                            verbosity=0)
        grid = {
            "n_estimators": [200, 400, 800],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 4, 6, 8],
            "min_child_weight": [1, 5, 10],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.3, 0.5, 0.8],
            "reg_lambda": [1, 5, 10],
        }
        return est, grid
    raise ValueError(name)


def grouped_folds(train, n_splits, seed):
    """5 folds on train, grouped by template, stratified by category (not label:
    by BBQ's design every template is ~25% biased, so label strata add little)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(train)), train["category"], train["template_id"]))


def best_f1_threshold(y, prob):
    """Cutoff with the highest F1 (picked on val, applied unchanged to test)."""
    p, r, t = precision_recall_curve(y, prob)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i])


def permutation_test(est, Xtr, ytr, Xev, yev, n_perm, seed):
    """Ojala & Garriga Test 1: refit on shuffled train labels, score PR-AUC on the eval
    split. Labels are shuffled freely across train rows, which ignores the 3-rows-per-question
    structure, so treat it as a sanity check that the model learned
    something real, not as the main significance test (that is the cluster bootstrap)."""
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    real = average_precision_score(yev, clone(est).fit(Xtr, ytr).predict_proba(Xev)[:, 1])
    null = []
    for _ in range(n_perm):
        m = clone(est).fit(Xtr, rng.permutation(ytr))
        null.append(average_precision_score(yev, m.predict_proba(Xev)[:, 1]))
    null = np.array(null)
    return {"real_pr_auc": real, "null_mean": float(null.mean()), "null_max": float(null.max()),
            "n_perm": n_perm, "p_value": float((1 + (null >= real).sum()) / (n_perm + 1))}


def run_one(set_name, model_name, X, table, folds, args):
    out_dir = Path(args.out) / f"{set_name}_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    F = build_features(X, set_name)
    tr = (table["split"] == "train").to_numpy()
    va = (table["split"] == "val").to_numpy()
    te = (table["split"] == "test").to_numpy()
    ytr = table.loc[tr, "is_biased"].to_numpy()
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())

    est, grid = model_and_grid(model_name, pos_weight, args.seed, args.threads)
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    n_iter = min(n_combos, args.n_iter)
    # LR fits are single-threaded, so parallelize the search; RF/XGB use threads inside.
    search = RandomizedSearchCV(est, grid, n_iter=n_iter, scoring="average_precision", cv=folds,
                                n_jobs=args.threads if model_name == "lr" else 1,
                                refit=True, random_state=args.seed, error_score="raise")
    t0 = time.time()
    search.fit(F[tr], ytr)
    fit_min = (time.time() - t0) / 60
    model = search.best_estimator_
    print(f"[{set_name}/{model_name}] dims={F.shape[1]} cv_pr_auc={search.best_score_:.4f} "
          f"best={search.best_params_} ({n_iter} configs, {fit_min:.1f} min)", flush=True)

    report = {"feature_set": set_name, "model": model_name, "dims": int(F.shape[1]),
              "cv_pr_auc": float(search.best_score_), "best_params": search.best_params_,
              "n_configs": n_iter, "fit_minutes": fit_min, "scale_pos_weight": pos_weight,
              "cv_results": pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score"]]
              .sort_values("mean_test_score", ascending=False).head(10).to_dict("records")}

    thr = None
    for split, mask in [("val", va), ("test", te)]:
        df = table.loc[mask, PRED_COLS].copy()
        df["prob"] = model.predict_proba(F[mask])[:, 1]
        df.to_csv(out_dir / f"{split}_preds.csv", index=False)
        if split == "val":
            thr, val_f1 = best_f1_threshold(df["is_biased"], df["prob"])
            report["val_threshold"] = {"threshold": thr, "val_f1": val_f1}
        report[split] = {
            "at_0.5": binary_metrics(df["is_biased"], df["prob"], 0.5),
            "at_val_threshold": binary_metrics(df["is_biased"], df["prob"], thr),
            "signal_flag_rates_0.5": signal_flag_rates(df, threshold=0.5),
            "breakdown_0.5": breakdown(df).to_dict("records"),
        }

    if args.perm and model_name == "lr":
        report["permutation_test_val"] = permutation_test(model, F[tr], ytr, F[va],
                                                          table.loc[va, "is_biased"].to_numpy(),
                                                          args.perm, args.seed)
        print(f"[{set_name}/{model_name}] permutation test: {report['permutation_test_val']}", flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    joblib.dump({"model": model, "feature_set": set_name, "threshold": thr}, out_dir / "model.joblib")
    return report


def summarize(reports):
    rows = []
    for r in reports:
        for split in ("val", "test"):
            m5, mt = r[split]["at_0.5"], r[split]["at_val_threshold"]
            s = r[split]["signal_flag_rates_0.5"]
            rows.append({
                "feature_set": r["feature_set"], "model": r["model"], "split": split, "dims": r["dims"],
                "cv_pr_auc": r["cv_pr_auc"], "pr_auc": m5.get("pr_auc"), "roc_auc": m5.get("roc_auc"),
                "precision@0.5": m5["precision"], "recall@0.5": m5["recall"], "f1@0.5": m5["f1"],
                "threshold": r["val_threshold"]["threshold"], "f1@thr": mt["f1"],
                "flag_correct": s["A: flag rate, correct answers"],
                "flag_unknown": s["A: flag rate, unknown answers"],
                "flag_biased": s["B: flag rate, wrong & biased"],
                "flag_anti": s["B: flag rate, wrong & anti-stereotype"],
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="outputs/features")
    p.add_argument("--out", default="outputs/classical")
    p.add_argument("--sets", nargs="+", default=FEATURE_SETS, choices=FEATURE_SETS)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--n-iter", type=int, default=20, help="Max configs per randomized search.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--perm", type=int, default=0, help="Permutation-test refits for LR (0 = skip).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X, table = load_features(args.features)
    train = table[table["split"] == "train"]
    folds = grouped_folds(train, args.folds, args.seed)
    for i, (a, b) in enumerate(folds):
        assert not set(train["template_id"].iloc[a]) & set(train["template_id"].iloc[b])
        print(f"fold {i}: fit={len(a)} score={len(b)} score_templates={train['template_id'].iloc[b].nunique()} "
              f"biased={train['is_biased'].iloc[b].mean():.3f}")

    reports = []
    out = Path(args.out)
    for set_name in args.sets:
        for model_name in args.models:
            reports.append(run_one(set_name, model_name, X, table, folds, args))
            # Merge with earlier runs so partial reruns keep the full table.
            summ = summarize(reports)
            old = out / "summary.csv"
            if old.exists():
                prev = pd.read_csv(old)
                keys = set(zip(summ["feature_set"], summ["model"]))
                prev = prev[[k not in keys for k in zip(prev["feature_set"], prev["model"])]]
                summ = pd.concat([prev, summ], ignore_index=True)
            summ.to_csv(old, index=False)

    pd.set_option("display.width", 250)
    print(pd.read_csv(out / "summary.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
