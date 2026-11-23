"""
Detector features: one 1158-number vector per (context, question, answer) row.

This is the script version of notebooks/03_features.ipynb. It runs on a laptop
(CPU or Mac GPU) as well as on Colab, and it can featurize new rows (for
example, a subject model's answers) through `FeatureExtractor.transform`.
See docs/features.md for why each part is there.

    | Columns   | What                                   |
    |-----------|----------------------------------------|
    | 0-383     | context embedding (all-MiniLM-L6-v2)   |
    | 384-767   | question embedding                     |
    | 768-1151  | answer embedding                       |
    | 1152-1154 | cos sim: context-question, context-answer, question-answer |
    | 1155-1157 | NLI probs: contradiction, entailment, neutral |

Run from the repo root to build features for the whole BBQ table:
    .venv/bin/python -m src.features.featurize --out outputs/features
Writes features.npy (69240 x 1158, float32) and features_meta.csv (row i of the
CSV goes with row i of the array). Both are gitignored.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
EMB_DIM = 384

# Column slices into the 1158-wide feature matrix.
CTX = slice(0, 384)
QUE = slice(384, 768)
ANS = slice(768, 1152)
SIMS = slice(1152, 1155)
NLI = slice(1155, 1158)
N_FEATURES = 1158
NLI_LABELS = ["contradiction", "entailment", "neutral"]  # output order we store

META_COLS = ["row_id", "split", "is_biased", "category", "context_condition", "statement"]


def pick_device(device="auto"):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_statement(question, answer, is_unknown):
    """Turn question + answer into one sentence for the NLI model.

    "Who was the secretary?" + "The woman" -> "The woman was the secretary."
    For the "unknown" option we always write "It is not known ...", because the
    raw option text gives nonsense ("Not enough information was the secretary.").
    `is_unknown` only says which option is the "can't tell" one, not whether it
    is correct, so it does not leak the label.
    """
    q = question.strip().rstrip("?")
    if is_unknown:
        return "It is not known " + q[0].lower() + q[1:] + "."
    if q.startswith("Who "):
        return answer + q[3:] + "."
    if q.startswith("Whose "):
        return answer + "'s" + q[5:] + "."
    if q.startswith("What group of people "):
        return "The group of people that" + q[len("What group of people"):] + " is " + answer + "."
    if q.startswith("Which parents "):
        return answer + q[len("Which parents"):] + "."
    return f"{q}? {answer}."


def find_unknown_option(options):
    """Index of the "can't tell" option among BBQ answer texts, or None.

    Used when featurizing rows that don't come from our loader (e.g. a subject
    model's pick), so `is_unknown` has to be worked out from the option text.
    """
    from src.rewriter.rewriter import find_unknown  # same word list as the rewriter
    return find_unknown(options)


class FeatureExtractor:
    """Loads the two pretrained models once; `transform(df)` returns an (n, 1158) float32 array.

    `df` needs columns: context, question, answer_text, is_unknown.
    """

    def __init__(self, device="auto", emb_batch=256, nli_batch=32, nli_device=None):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.device = pick_device(device)
        self.nli_device = pick_device(nli_device or device)
        self.emb_batch = emb_batch
        self.nli_batch = nli_batch
        self.emb_model = SentenceTransformer(EMB_MODEL, device=self.device)
        self.nli_model = CrossEncoder(NLI_MODEL, device=self.nli_device)
        # Map the model's own label order to ours (contradiction, entailment, neutral).
        id2label = {int(k): v.lower() for k, v in self.nli_model.model.config.id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        self.nli_order = [label2id[name] for name in NLI_LABELS]

    def embed(self, texts):
        """Embed each unique text once (contexts repeat across the 3 answer rows)."""
        texts = pd.Series(texts).astype(str)
        uniq = texts.unique()
        vecs = self.emb_model.encode(list(uniq), batch_size=self.emb_batch,
                                     normalize_embeddings=True, show_progress_bar=False,
                                     convert_to_numpy=True)
        pos = {t: i for i, t in enumerate(uniq)}
        return vecs[texts.map(pos).to_numpy()].astype(np.float32)

    def nli_probs(self, premises, hypotheses, log_every=0, cache_dir=None):
        """Softmax NLI probabilities in (contradiction, entailment, neutral) order.

        Pairs are sorted by length so each batch pads less; results come back in input order.
        With `cache_dir`, each chunk's logits are saved as it finishes, so a run that dies
        (e.g. out of memory on the longest pairs, which come last) resumes where it stopped.
        """
        pairs = list(zip(premises, hypotheses))
        order = np.argsort([len(p) + len(h) for p, h in pairs], kind="stable")
        logits = np.zeros((len(pairs), 3), dtype=np.float32)
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
        t0, done_new = time.time(), 0
        step = 1280
        for start in range(0, len(pairs), step):
            idx = order[start:start + step]
            chunk_file = cache_dir / f"nli_{start:06d}_{len(idx)}.npy" if cache_dir else None
            if chunk_file is not None and chunk_file.exists():
                logits[idx] = np.load(chunk_file)
                continue
            out = self.nli_model.predict([pairs[i] for i in idx], batch_size=self.nli_batch,
                                         show_progress_bar=False, convert_to_numpy=True,
                                         apply_softmax=False)
            logits[idx] = out
            if chunk_file is not None:
                np.save(chunk_file, out.astype(np.float32))
            if self.nli_device == "mps":
                torch.mps.empty_cache()
            done_new += len(idx)
            if log_every and (start // step) % log_every == 0:
                done = min(start + step, len(pairs))
                rate = done_new / (time.time() - t0)
                print(f"  NLI {done}/{len(pairs)} ({rate:.0f} pairs/s, "
                      f"~{(len(pairs) - done) / rate / 60:.1f} min left)", flush=True)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, self.nli_order].astype(np.float32)

    def transform(self, df, log_every=0, cache_dir=None):
        statements = [make_statement(q, a, u) for q, a, u in
                      zip(df["question"], df["answer_text"], df["is_unknown"])]
        c = self.embed(df["context"])
        q = self.embed(df["question"])
        a = self.embed(df["answer_text"])
        # Embeddings are unit length, so cosine similarity is the dot product.
        sims = np.stack([(c * q).sum(1), (c * a).sum(1), (q * a).sum(1)], axis=1)
        nli = self.nli_probs(df["context"].tolist(), statements, log_every=log_every, cache_dir=cache_dir)
        X = np.hstack([c, q, a, sims, nli]).astype(np.float32)
        assert X.shape[1] == N_FEATURES
        return X, statements


def load_features(folder="outputs/features"):
    """Returns (X, table): the feature matrix and the full BBQ table, row-aligned.

    `table` is load_bbq() reordered to match features_meta.csv, with `statement` added,
    so every schema column (question_id, template_id, ...) is available next to X.
    Works with the Colab files from the team Drive too (same two file names).
    """
    from src.data.bbq_loader import load_bbq

    folder = Path(folder)
    X = np.load(folder / "features.npy")
    meta = pd.read_csv(folder / "features_meta.csv")
    table = load_bbq().set_index("row_id")
    missing = set(meta["row_id"]) - set(table.index)
    if missing:
        raise ValueError(f"{len(missing)} feature rows are not in the BBQ table, e.g. {sorted(missing)[:3]}")
    table = table.loc[meta["row_id"]].reset_index()
    table["statement"] = meta["statement"].to_numpy()
    # The label in the meta file must match the current loader (it was regenerated after PR #6).
    if not (table["is_biased"].to_numpy() == meta["is_biased"].to_numpy()).all():
        raise ValueError("features_meta.csv is_biased disagrees with load_bbq(); regenerate the meta file")
    return X, table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/features")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--nli-device", default=None, choices=[None, "cuda", "mps", "cpu"],
                   help="Device for the NLI cross-encoder (defaults to --device).")
    p.add_argument("--nli-batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="Only the first N rows (smoke test).")
    args = p.parse_args()

    from src.data.bbq_loader import load_bbq

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = load_bbq()
    if args.limit:
        table = table.head(args.limit)
    print(f"rows={len(table)}")

    fx = FeatureExtractor(device=args.device, nli_batch=args.nli_batch, nli_device=args.nli_device)
    print(f"embeddings on {fx.device}, NLI on {fx.nli_device}; NLI column order from model: {fx.nli_order}")
    t0 = time.time()
    X, statements = fx.transform(table, log_every=5, cache_dir=out / "nli_cache")
    table["statement"] = statements
    print(f"done in {(time.time() - t0) / 60:.1f} min, X={X.shape}")

    np.save(out / "features.npy", X)
    table[META_COLS].to_csv(out / "features_meta.csv", index=False)
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
