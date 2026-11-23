"""
One interface for every trained bias detector, so the pipeline doesn't care which one it uses.

    from src.detector.detector import BiasDetector
    det = BiasDetector.load("outputs/classical/full_xgb")   # classical (model.joblib)
    det = BiasDetector.load("outputs/deberta/model")        # fine-tuned DeBERTa (HF folder)

    det.score_one(context, question, answer, options)      # -> P(biased), a float
    det.score(rows)                                        # -> np.array of P(biased)
    det.flag(rows)                                         # -> np.array of 0/1 at det.threshold

`rows` is a DataFrame (or list of dicts) in the docs/schema.md format. Only these
columns are needed: context, question, answer_text, and either is_unknown or options
(the 3 BBQ answer texts, used to tell which option is "unknown"). A subject model's
answer must first be mapped to one of the 3 options (see src/rewriter/rewriter.py
map_to_option), because every detector was trained on BBQ option texts.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _as_frame(rows):
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if "answer_text" not in df and "answer" in df:
        df["answer_text"] = df["answer"]
    for col in ("context", "question", "answer_text"):
        if col not in df:
            raise ValueError(f"rows need a '{col}' column")
    if "is_unknown" not in df:
        if "options" not in df:
            raise ValueError("rows need 'is_unknown' or 'options' (to find the unknown option)")
        from src.rewriter.rewriter import find_unknown, norm
        df["is_unknown"] = [
            int(find_unknown(list(opts)) is not None
                and norm(ans) == norm(list(opts)[find_unknown(list(opts))]))
            for ans, opts in zip(df["answer_text"], df["options"])
        ]
    return df


class BiasDetector:
    def __init__(self, kind, threshold=0.5):
        self.kind = kind
        self.threshold = threshold

    @staticmethod
    def load(path, threshold=None, device="auto"):
        """Picks the detector type from what is in the folder."""
        path = Path(path)
        if (path / "model.joblib").exists():
            det = _ClassicalDetector(path / "model.joblib", device)
        elif (path / "config.json").exists():
            det = _DebertaDetector(path, device)
        else:
            raise FileNotFoundError(f"{path} has neither model.joblib nor config.json")
        if threshold is not None:
            det.threshold = threshold
        return det

    def score(self, rows):
        raise NotImplementedError

    def flag(self, rows):
        return (self.score(rows) >= self.threshold).astype(int)

    def score_one(self, context, question, answer, options=None, is_unknown=None):
        row = {"context": context, "question": question, "answer_text": answer}
        if is_unknown is not None:
            row["is_unknown"] = int(is_unknown)
        else:
            row["options"] = options
        return float(self.score([row])[0])


class _ClassicalDetector(BiasDetector):
    """LR / RF / XGBoost on the 1158 features. Featurizes new rows with the same models."""

    def __init__(self, joblib_path, device):
        import joblib

        saved = joblib.load(joblib_path)
        super().__init__("classical", saved.get("threshold") or 0.5)
        self.model = saved["model"]
        self.feature_set = saved["feature_set"]
        self.device = device
        self._fx = None  # loaded on first use (two pretrained models)

    def score(self, rows):
        from src.detector.train_classical import build_features
        from src.features.featurize import FeatureExtractor

        df = _as_frame(rows)
        if self._fx is None:
            self._fx = FeatureExtractor(device=self.device)
        X, _ = self._fx.transform(df)
        return self.model.predict_proba(build_features(X, self.feature_set))[:, 1]


class _DebertaDetector(BiasDetector):
    """Fine-tuned cross-encoder from train_deberta.py --save-model. Input format must match
    training: (premise = context, hypothesis = question + " " + answer)."""

    def __init__(self, model_dir, device, batch_size=32, max_len=256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from src.features.featurize import pick_device

        super().__init__("deberta", 0.5)
        self.torch = torch
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.batch_size, self.max_len = batch_size, max_len

    def score(self, rows):
        df = _as_frame(rows)
        hyp = (df["question"] + " " + df["answer_text"]).tolist()
        ctx = df["context"].tolist()
        out = []
        with self.torch.no_grad():
            for i in range(0, len(df), self.batch_size):
                enc = self.tok(ctx[i:i + self.batch_size], hyp[i:i + self.batch_size], truncation=True,
                               max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                out.append(self.torch.softmax(logits, -1)[:, 1].float().cpu().numpy())
        return np.concatenate(out)
