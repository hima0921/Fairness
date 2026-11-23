"""
Compare bias detectors on the same rows: metrics, cluster-bootstrap CIs and McNemar tests.

Input: any number of per-row prediction files (`row_id`, `is_biased`, `prob`, plus the
schema columns written by train_deberta.py / train_classical.py). Rows are matched
by row_id; only rows present in every file are used.

Tests:
- Main: paired cluster bootstrap. Resample whole questions (all 3 answer rows) with
  replacement, B times; score every detector on the same resample. Gives a 95% CI
  for each PR-AUC and for each pairwise difference. Repeated with templates as the
  cluster as a sensitivity check (few templates, so wider and less exact CIs).
- Secondary: McNemar's test on row-level correctness at each detector's cutoff,
  Bonferroni-corrected over all pairs (as in the proposal). It assumes independent
  rows, which ours are not, so its p-values are too small; report it next to the
  bootstrap, not instead of it.

    .venv/bin/python -m src.detector.compare \
        --pred lr=outputs/classical/full_lr/val_preds.csv \
        --pred deberta=outputs/xsmall_6k/val_preds.csv --out outputs/compare_val
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from src.detector.metrics import binary_metrics, signal_flag_rates


def fast_ap(y_sorted, w_sorted, last_of_tie):
    """Weighted average precision for rows pre-sorted by score (descending).

    Same definition as sklearn's average_precision_score (step-wise, no interpolation):
    AP = sum_n (R_n - R_{n-1}) P_n over distinct score thresholds. `w` are bootstrap
    counts; `last_of_tie` marks the last row of each block of tied scores.
    Vectorized over the rows of a 2-D weight matrix (one bootstrap resample per row).
    """
    tp = np.cumsum(w_sorted * y_sorted, axis=-1)[..., last_of_tie]
    pp = np.cumsum(w_sorted, axis=-1)[..., last_of_tie]
    total_pos = tp[..., -1:]
    precision = np.divide(tp, pp, out=np.zeros_like(tp, dtype=float), where=pp > 0)
    recall = np.divide(tp, total_pos, out=np.zeros_like(tp, dtype=float), where=total_pos > 0)
    d_recall = np.diff(recall, axis=-1, prepend=0)
    return (d_recall * precision).sum(axis=-1)


class APBootstrapper:
    """Pre-sorts one detector's scores so each bootstrap AP is a cumsum, not a sort."""

    def __init__(self, y, prob):
        order = np.argsort(-prob, kind="mergesort")
        self.order = order
        self.y = y[order].astype(float)
        p = prob[order]
        self.last_of_tie = np.r_[p[1:] != p[:-1], True]

    def ap(self, row_weights):
        return fast_ap(self.y, row_weights[..., self.order], self.last_of_tie)


def cluster_weights(cluster_codes, n_clusters, B, rng, chunk):
    """Yield (chunk, n_rows) arrays of row weights = how often each row's cluster was drawn."""
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        draws = rng.integers(0, n_clusters, size=(b, n_clusters))
        counts = np.zeros((b, n_clusters))
        np.add.at(counts, (np.arange(b)[:, None], draws), 1)
        yield counts[:, cluster_codes]


def bootstrap(df, names, cluster_col, B, seed, thresholds, chunk=200):
    """Paired cluster bootstrap of PR-AUC and error rate for every detector and pair."""
    y = df["is_biased"].to_numpy()
    codes, uniq = pd.factorize(df[cluster_col])
    rng = np.random.default_rng(seed)
    boots = {n: APBootstrapper(y, df[f"prob_{n}"].to_numpy()) for n in names}
    wrong = {n: ((df[f"prob_{n}"].to_numpy() >= thresholds[n]).astype(int) != y).astype(float)
             for n in names}
    ap = {n: [] for n in names}
    err = {n: [] for n in names}
    for W in cluster_weights(codes, len(uniq), B, rng, chunk):
        tot = W.sum(1)
        for n in names:
            ap[n].append(boots[n].ap(W))
            err[n].append((W * wrong[n]).sum(1) / tot)
    ap = {n: np.concatenate(v) for n, v in ap.items()}
    err = {n: np.concatenate(v) for n, v in err.items()}

    def ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5])
        return [float(lo), float(hi)]

    single = {n: {"pr_auc_ci": ci(ap[n]), "error_ci": ci(err[n])} for n in names}
    pairs = {}
    for a, b in itertools.combinations(names, 2):
        d_ap, d_err = ap[a] - ap[b], err[a] - err[b]
        pairs[f"{a} - {b}"] = {
            "delta_pr_auc_ci": ci(d_ap), "delta_pr_auc_excludes_0": bool(np.sign(ci(d_ap)).prod() > 0),
            "delta_error_ci": ci(d_err), "delta_error_excludes_0": bool(np.sign(ci(d_err)).prod() > 0),
            # two-sided bootstrap p-value for the PR-AUC difference
            "p_boot_pr_auc": float(min(1.0, 2 * min((d_ap <= 0).mean(), (d_ap >= 0).mean()))),
        }
    return {"cluster": cluster_col, "n_clusters": int(len(uniq)), "B": B, "single": single, "pairs": pairs}


def mcnemar_tests(df, names, thresholds):
    y = df["is_biased"].to_numpy()
    rows = []
    for a, b in itertools.combinations(names, 2):
        ca = (df[f"prob_{a}"].to_numpy() >= thresholds[a]).astype(int) == y
        cb = (df[f"prob_{b}"].to_numpy() >= thresholds[b]).astype(int) == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())  # a right/b wrong, a wrong/b right
        table = [[int((ca & cb).sum()), n01], [n10, int((~ca & ~cb).sum())]]
        exact = (n01 + n10) < 25
        res = mcnemar(table, exact=exact, correction=True)
        rows.append({"pair": f"{a} vs {b}", "a_right_b_wrong": n01, "a_wrong_b_right": n10,
                     "test": "exact binomial" if exact else "chi2 (continuity corrected)",
                     "statistic": float(res.statistic), "p": float(res.pvalue)})
    out = pd.DataFrame(rows)
    if len(out):
        reject, p_adj, _, _ = multipletests(out["p"], alpha=0.05, method="bonferroni")
        out["p_bonferroni"] = p_adj
        out["significant_0.05"] = reject
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", action="append", required=True,
                   help="name=path/to/preds.csv, optionally name=path:threshold (default cutoff 0.5)")
    p.add_argument("--out", required=True)
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    names, thresholds, merged = [], {}, None
    for spec in args.pred:
        name, rest = spec.split("=", 1)
        path, _, thr = rest.partition(":")
        thresholds[name] = float(thr) if thr else 0.5
        d = pd.read_csv(path)
        names.append(name)
        if merged is None:
            merged = d.drop(columns=["prob"]).assign(**{f"prob_{name}": d["prob"]})
        else:
            merged = merged.merge(d[["row_id", "is_biased", "prob"]].rename(
                columns={"prob": f"prob_{name}", "is_biased": f"y_{name}"}), on="row_id", how="inner")
            assert (merged[f"y_{name}"] == merged["is_biased"]).all(), f"labels differ for {name}"
            merged = merged.drop(columns=[f"y_{name}"])
    print(f"{len(merged)} shared rows, {merged['question_id'].nunique()} questions, "
          f"{merged['template_id'].nunique()} templates")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame({n: {**binary_metrics(merged["is_biased"], merged[f"prob_{n}"], thresholds[n]),
                                "threshold": thresholds[n],
                                **signal_flag_rates(merged, f"prob_{n}", thresholds[n])} for n in names}).T
    boot_q = bootstrap(merged, names, "question_id", args.B, args.seed, thresholds)
    boot_t = bootstrap(merged, names, "template_id", args.B, args.seed, thresholds)
    mc = mcnemar_tests(merged, names, thresholds)

    metrics.to_csv(out / "metrics.csv")
    mc.to_csv(out / "mcnemar.csv", index=False)
    (out / "bootstrap.json").write_text(json.dumps({"question": boot_q, "template": boot_t}, indent=2))

    pd.set_option("display.width", 250)
    print(metrics[["precision", "recall", "f1", "pr_auc", "roc_auc", "threshold"]].round(3))
    for boot in (boot_q, boot_t):
        print(f"\nPR-AUC 95% CI, clusters = {boot['cluster']} (n={boot['n_clusters']}, B={boot['B']})")
        for n, s in boot["single"].items():
            print(f"  {n:20s} [{s['pr_auc_ci'][0]:.3f}, {s['pr_auc_ci'][1]:.3f}]")
        for k, s in boot["pairs"].items():
            print(f"  {k:30s} dPR-AUC [{s['delta_pr_auc_ci'][0]:+.3f}, {s['delta_pr_auc_ci'][1]:+.3f}]"
                  f"  dErr [{s['delta_error_ci'][0]:+.3f}, {s['delta_error_ci'][1]:+.3f}]")
    print("\nMcNemar (Bonferroni):")
    print(mc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
