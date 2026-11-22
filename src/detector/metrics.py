"""
Detector metrics and reference baselines, shared by every detector we train.

All functions take per-row binary labels (`is_biased`) and the detector's
probability that the row is biased, so DeBERTa, the classical models and an
LLM judge can be scored the same way.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, prob, threshold=0.5):
    """Threshold metrics plus threshold-free PR-AUC / ROC-AUC.

    PR-AUC (average precision) is the main number: only ~25% of rows are
    biased, so accuracy alone is misleading (always predicting 0 gets 0.75).
    """
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    out = {
        "n": len(y_true),
        "pos_rate": y_true.mean(),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    # AUCs are undefined if a slice has only one class (e.g. a tiny subgroup).
    if len(np.unique(y_true)) == 2:
        out["pr_auc"] = average_precision_score(y_true, prob)
        out["roc_auc"] = roc_auc_score(y_true, prob)
    return out


def breakdown(df, prob_col="prob", by=("context_condition", "category"), threshold=0.5):
    """Metrics per slice (context type, category, ...). `df` needs `is_biased` and `prob_col`."""
    rows = []
    for col in by:
        for val, g in df.groupby(col):
            m = binary_metrics(g["is_biased"], g[prob_col], threshold)
            rows.append({"slice": f"{col}={val}", **m})
    return pd.DataFrame(rows)


def false_positive_rate_by_group(df, prob_col="prob", group_col="answer_group", threshold=0.5):
    """FPR per answer group: how often unbiased answers about each group get flagged.

    This checks whether the detector itself treats some groups unfairly
    (an equalized-odds check on the detector). `answer_group` tags are raw
    BBQ tags and are not cleaned here.
    """
    neg = df[df["is_biased"] == 0]
    pred = (neg[prob_col] >= threshold).astype(int)
    return (
        pred.groupby(neg[group_col]).agg(["mean", "size"])
        .rename(columns={"mean": "fpr", "size": "n_unbiased"})
        .sort_values("fpr", ascending=False)
    )


def signal_flag_rates(df, prob_col="prob", threshold=0.5):
    """Flag rates that separate the two halves of the task.

    - Signal A (is the answer unsupported by the context?): a good detector
      rarely flags correct answers or "unknown" answers.
    - Signal B (does the wrong answer follow the stereotype?): among wrong,
      non-"unknown" answers, a good detector flags biased ones much more
      often than anti-stereotype ones. If those two rates are close, the
      detector is flagging "wrong person" rather than "stereotyped person".
    """
    pred = (df[prob_col] >= threshold)
    wrong_person = (df["is_correct"] == 0) & (df["is_unknown"] == 0)
    return {
        "A: flag rate, correct answers": pred[df["is_correct"] == 1].mean(),
        "A: flag rate, unknown answers": pred[df["is_unknown"] == 1].mean(),
        "B: flag rate, wrong & biased": pred[wrong_person & (df["is_biased"] == 1)].mean(),
        "B: flag rate, wrong & anti-stereotype": pred[wrong_person & (df["is_biased"] == 0)].mean(),
    }


def reference_baselines(df, seed=0):
    """No-learning reference points the detector must beat.

    - random: flags each row with probability equal to the base rate (~0.25).
    - always_biased: flags everything (recall 1, precision = base rate).
    - oracle_signal_A: flags every wrong, non-"unknown" answer using the gold
      labels. This is a perfect "is the answer unsupported by the context?"
      signal with no stereotype knowledge, so precision is ~0.5 by BBQ's
      construction. It is a reference ceiling for context-only detectors,
      not a fair competitor (it reads gold labels).
    """
    rng = np.random.default_rng(seed)
    y = df["is_biased"].to_numpy()
    base = y.mean()
    oracle_a = ((df["is_correct"] == 0) & (df["is_unknown"] == 0)).astype(float).to_numpy()
    return {
        "random": binary_metrics(y, (rng.random(len(y)) < base).astype(float)),
        "always_biased": binary_metrics(y, np.ones(len(y))),
        "oracle_signal_A": binary_metrics(y, oracle_a),
    }
