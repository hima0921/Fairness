"""
ROC and precision-recall curves for every detector we have run, on the same rows.

Reads the per-row predictions each run already saved (`<run>/val_preds.csv`),
keeps only the rows that every run scored (the LLM judges were run on a
300-question val subsample, DeBERTa on the full val split), and draws:
  - one ROC curve and one PR curve per detector,
  - the reference baselines from metrics.reference_baselines:
      random       -> the chance line (diagonal on ROC, base rate on PR),
      always predict biased -> a single point (it has no score to sweep),
      wrong-answer oracle -> a single point (yes/no from the true labels).

Run from the repo root (needs the outputs/ dirs from the runs listed below):
    .venv/bin/python -m src.detector.plot_curves
Writes docs/figures/detector_curves.png and prints the table.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.detector.metrics import binary_metrics

# label -> predictions file. Paths match the commands in docs/detector_results.md.
RUNS = {
    "DeBERTa-v3-xsmall": "outputs/xsmall_6k/val_preds.csv",
    "LLM judge, zero-shot": "outputs/llm_judge_0shot/val_preds.csv",
    "LLM judge, 4-shot": "outputs/llm_judge_4shot/val_preds.csv",
}


def load_common_rows(runs):
    """One frame with a prob column per run, restricted to rows every run scored."""
    frames = {name: pd.read_csv(path) for name, path in runs.items()}
    common = set.intersection(*(set(f["row_id"]) for f in frames.values()))
    base = next(iter(frames.values()))
    df = base[base["row_id"].isin(common)].drop(columns="prob").set_index("row_id")
    for name, f in frames.items():
        df[name] = f.set_index("row_id").loc[df.index, "prob"]
    return df.reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/figures/detector_curves.png")
    args = p.parse_args()

    df = load_common_rows(RUNS)
    y = df["is_biased"].to_numpy()
    base_rate = y.mean()
    # Wrong-answer oracle: predict biased for every wrong, non-"unknown" answer (uses the true labels).
    oracle_a = ((df["is_correct"] == 0) & (df["is_unknown"] == 0)).astype(float)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = {"DeBERTa-v3-xsmall": "tab:blue", "LLM judge, zero-shot": "tab:orange", "LLM judge, 4-shot": "tab:green"}

    rows = []
    for name in RUNS:
        s = df[name]
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        roc_auc, pr_auc = roc_auc_score(y, s), average_precision_score(y, s)
        ax_roc.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} (AUC {roc_auc:.2f})")
        ax_pr.step(rec, prec, where="post", color=colors[name], lw=2, label=f"{name} (AP {pr_auc:.2f})")
        rows.append({"method": name, **binary_metrics(y, s)})

    # Baselines. Random's expected curve is the chance line; the two 0/1 rules are points.
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC 0.50)")
    ax_pr.axhline(base_rate, color="k", ls="--", lw=1, label=f"Random (AP {base_rate:.2f})")
    ax_roc.plot(1, 1, "ks", ms=7, label="Always predict biased")
    ax_pr.plot(1, base_rate, "ks", ms=7, label="Always predict biased")
    neg, pos = y == 0, y == 1
    oa_fpr, oa_tpr = oracle_a[neg].mean(), oracle_a[pos].mean()
    oa_prec = y[oracle_a == 1].mean()
    ax_roc.plot(oa_fpr, oa_tpr, "r*", ms=13, label="Wrong-answer oracle (reference)")
    ax_pr.plot(oa_tpr, oa_prec, "r*", ms=13, label="Wrong-answer oracle (reference)")

    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve",
               xlim=(0, 1), ylim=(0, 1.02))
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1.02))
    for ax in (ax_roc, ax_pr):
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right" if ax is ax_roc else "upper right", fontsize=8)
    fig.suptitle(f"Bias detectors on {len(df)} val rows ({len(df) // 3} questions, {base_rate:.1%} biased)")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)

    # Table: reference rows + detectors, threshold 0.5 for P/R/F1.
    rows = [
        {"method": "Always predict biased", **binary_metrics(y, [1.0] * len(y))},
        {"method": "Wrong-answer oracle (reference)", **binary_metrics(y, oracle_a)},
    ] + rows
    table = pd.DataFrame(rows).set_index("method")[["precision", "recall", "f1", "pr_auc", "roc_auc"]]
    print(table.round(2).to_string())
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
