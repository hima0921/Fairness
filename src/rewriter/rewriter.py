"""
T5 rewriter: turns a flagged answer into the correct answer plus a short reason.
Trained in notebooks/04_rewriter.ipynb (T5-base, 3 epochs). Weights are on the team Drive.

    from src.rewriter.rewriter import BiasRewriter
    rw = BiasRewriter("path/to/t5base_epoch3")
    out = rw.rewrite(context, question, options, answer, justification)
"""
import difflib
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MAX_IN, MAX_OUT = 384, 128
UNKNOWN = ["can't answer", "can't be determined", "cannot answer", "cannot be determined",
           "not answerable", "not enough info", "not enough information", "not known",
           "undetermined", "unknown"]


def norm(t):
    t = re.sub(r"[^\w\s']", " ", t.lower())
    t = re.sub(r"^\s*(the|a|an|my)\s+", "", t)
    return " ".join(t.split())


def find_unknown(options):
    for i, o in enumerate(options):
        if norm(o) in [norm(u) for u in UNKNOWN]:
            return i
    return None


def placeholder_just(answer, options):
    if options.index(answer) == find_unknown(options):
        return "There is not enough information to tell."
    return f"{answer} seems like the most likely answer."


def make_input(context, question, options, answer, justification=""):
    if not justification:
        justification = placeholder_just(answer, options)
    opts = " | ".join(options)
    return (f"fix bias: question: {question} options: {opts} "
            f"answer: {answer} justification: {justification} context: {context}")


def map_to_option(output, options):
    """Returns 0, 1 or 2, or -1 if the output matches no option."""
    ans = norm(output.split("Reason:")[0])
    opts = [norm(o) for o in options]
    if ans in opts:
        return opts.index(ans)
    unk = find_unknown(options)
    if unk is not None and any(norm(u) in ans for u in UNKNOWN):
        return unk
    scores = [difflib.SequenceMatcher(None, ans, o).ratio() for o in opts]
    best = max(range(len(options)), key=lambda i: scores[i])
    return best if scores[best] >= 0.6 else -1


class BiasRewriter:
    def __init__(self, model_dir, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device).eval()

    @torch.no_grad()
    def rewrite_batch(self, items, batch_size=32):
        """items: list of dicts with context, question, options, answer, justification."""
        texts = [make_input(it["context"], it["question"], it["options"],
                            it["answer"], it.get("justification", "")) for it in items]
        outs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], max_length=MAX_IN, truncation=True,
                           padding=True, return_tensors="pt").to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=MAX_OUT)
            outs += self.tok.batch_decode(gen, skip_special_tokens=True)
        results = []
        for it, out in zip(items, outs):
            idx = map_to_option(out, it["options"])
            results.append({
                "answer_idx": idx,
                "answer_text": it["options"][idx] if idx >= 0 else None,
                "justification": out.split("Reason:", 1)[1].strip() if "Reason:" in out else "",
                "raw_output": out,
            })
        return results

    def rewrite(self, context, question, options, answer, justification=""):
        return self.rewrite_batch([{"context": context, "question": question, "options": options,
                                    "answer": answer, "justification": justification}])[0]
