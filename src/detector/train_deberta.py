"""
Fine-tune a DeBERTa-v3 cross-encoder as the bias detector.

Input:  (context, question + " " + answer) as a sentence pair.
Target: `is_biased` (see docs/schema.md).

Why this setup:
- We start from an NLI-pretrained cross-encoder (cross-encoder/nli-deberta-v3-base).
  Half of the task ("does the context support this answer?") is NLI, so the
  model starts with that skill; fine-tuning mainly has to add stereotype
  knowledge ("does the answer point the way the stereotype points?").
- The pair format mirrors NLI: context = premise, question + answer = hypothesis.
- The NLI head has 3 classes; we replace it with a fresh 2-class head.
- No hand-built features, and no metadata (category, context type, group)
  is given to the model. Metadata is only used to break down results.

Run from the repo root:
    .venv/bin/python -m src.detector.train_deberta --epochs 2
Outputs (gitignored) go to outputs/deberta/.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.data.bbq_loader import load_bbq
from src.detector.metrics import (
    binary_metrics,
    breakdown,
    false_positive_rate_by_group,
    reference_baselines,
    signal_flag_rates,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cross-encoder/nli-deberta-v3-base")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=256)  # BBQ inputs are short; this is not expected to truncate
    p.add_argument("--warmup-frac", type=float, default=0.06)
    p.add_argument("--max-train-questions", type=int, default=None,
                   help="Subsample training QUESTIONS (all 3 answer rows kept) for quick runs.")
    p.add_argument("--max-eval-questions", type=int, default=None,
                   help="Subsample eval QUESTIONS for quick runs. Numbers are then noisier than a full eval.")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="auto = cuda > mps > cpu. DeBERTa stalled on MPS on an M4 Mac in our test; use cpu there.")
    p.add_argument("--eval-split", default="val", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/deberta")
    p.add_argument("--save-model", action="store_true")
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(df, tok, max_len, batch_size, shuffle, seed):
    """Tokenize as (premise=context, hypothesis=question + answer) pairs, padded per batch."""
    hyp = (df["question"] + " " + df["answer_text"]).tolist()
    enc = tok(df["context"].tolist(), hyp, truncation=True, max_length=max_len)
    items = [
        {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
         "labels": int(df["is_biased"].iloc[i])}
        for i in range(len(df))
    ]

    def collate(batch):
        out = tok.pad([{k: b[k] for k in ("input_ids", "attention_mask")} for b in batch], return_tensors="pt")
        out["labels"] = torch.tensor([b["labels"] for b in batch])
        return out

    g = torch.Generator().manual_seed(seed)
    n_trunc = sum(len(x) >= max_len for x in enc["input_ids"])
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=g), n_trunc


@torch.no_grad()
def predict(model, loader, device):
    """Returns P(biased) per row, in loader order."""
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu())
    return torch.cat(probs).numpy()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_bbq()
    train = table[table["split"] == "train"]
    evald = table[table["split"] == args.eval_split]

    def subsample_questions(df, n):
        # Subsample whole questions so each keeps its 3 answer rows together.
        rng = np.random.default_rng(args.seed)
        qids = rng.choice(df["question_id"].unique(), size=n, replace=False)
        return df[df["question_id"].isin(qids)]

    if args.max_train_questions:
        train = subsample_questions(train, args.max_train_questions)
    if args.max_eval_questions:
        evald = subsample_questions(evald, args.max_eval_questions)
    train = train.reset_index(drop=True)
    evald = evald.reset_index(drop=True)
    print(f"device={device} train_rows={len(train)} {args.eval_split}_rows={len(evald)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    # ignore_mismatched_sizes: drop the 3-way NLI head, start a fresh 2-way head.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    train_loader, tr_trunc = make_loader(train, tok, args.max_len, args.batch_size, True, args.seed)
    eval_loader, ev_trunc = make_loader(evald, tok, args.max_len, args.batch_size * 2, False, args.seed)
    print(f"truncated inputs: train={tr_trunc} eval={ev_trunc}")

    # Plain cross-entropy, no class weights: we report threshold-free PR-AUC and
    # pick a threshold later on val if needed, rather than reweighting the loss.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(opt, math.ceil(args.warmup_frac * total), total)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item()
            if step % args.log_every == 0:
                rate = step / (time.time() - t0)
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running / args.log_every:.4f} "
                      f"({rate:.1f} it/s, ~{(len(train_loader) - step) / rate / 60:.1f} min left)", flush=True)
                running = 0.0

        prob = predict(model, eval_loader, device)
        m = binary_metrics(evald["is_biased"], prob)
        history.append({"epoch": epoch, **m})
        print(f"epoch {epoch} {args.eval_split}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items() if k != "n"), flush=True)

    # Final-epoch predictions are what we report (no checkpoint selection on the eval split).
    evald["prob"] = prob
    keep = ["row_id", "question_id", "template_id", "category", "context_condition", "question_polarity",
            "answer_group", "is_correct", "is_unknown", "anti_stereotype", "is_biased", "prob"]
    evald[keep].to_csv(out_dir / f"{args.eval_split}_preds.csv", index=False)

    report = {
        "args": vars(args),
        "history": history,
        "final": binary_metrics(evald["is_biased"], prob),
        "baselines": reference_baselines(evald, seed=args.seed),
    }
    (out_dir / f"{args.eval_split}_report.json").write_text(json.dumps(report, indent=2, default=float))

    pd.set_option("display.width", 200)
    print("\n=== Reference baselines vs. DeBERTa ===")
    print(pd.DataFrame({**report["baselines"], "deberta": report["final"]}).T.round(3))
    print("\n=== Breakdown ===")
    print(breakdown(evald).round(3).to_string(index=False))
    print("\n=== Signal A / B flag rates ===")
    for k, v in signal_flag_rates(evald).items():
        print(f"{k:40s} {v:.3f}")
    print("\n=== FPR by answer group (top 10) ===")
    print(false_positive_rate_by_group(evald).head(10).round(3))

    if args.save_model:
        model.save_pretrained(out_dir / "model")
        tok.save_pretrained(out_dir / "model")


if __name__ == "__main__":
    main()
