"""
Loads BBQ and turns it into our project table.
Column definitions: docs/schema.md
How the splits work: docs/splits.md
"""

import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/"
META_URL = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/analysis_scripts/additional_metadata.csv"
DEFAULT_CATEGORIES = ["Gender_identity", "Race_ethnicity", "Age", "SES"]
KEYS = ["category", "question_index", "example_id"]
SEED = 600


def _find_unknown(info):
    for i in range(3):
        if info[f"ans{i}"][1] == "unknown":
            return i
    return None


def _build_rows(m):
    rows = []
    for _, q in m.iterrows():
        unk = _find_unknown(q["answer_info"])
        target = q["target_loc"]
        non_target = [i for i in range(3) if i not in (unk, target)][0]
        stereo = target if q["question_polarity"] == "neg" else non_target
        for i in range(3):
            is_correct = int(i == q["label"])
            rows.append({
                "row_id": f"{q['category']}_{q['example_id']}_ans{i}",
                "example_id": q["example_id"],
                "template_id": f"{q['category']}_{q['question_index']}",
                "category": q["category"],
                "context_condition": q["context_condition"],
                "question_polarity": q["question_polarity"],
                "context": q["context"],
                "question": q["question"],
                "answer_text": q[f"ans{i}"],
                "answer_idx": i,
                "is_correct": is_correct,
                "is_biased": int(not is_correct and i == stereo),
                "anti_stereotype": int(not is_correct and i != stereo and i != unk),
                "is_stereotype_answer": int(i == stereo),
                "is_unknown": int(i == unk),
            })
    return pd.DataFrame(rows)


def _add_splits(table, categories, seed):
    split_map = {}
    for cat in categories:
        templates = sorted(table.loc[table["category"] == cat, "template_id"].unique())
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(templates)
        n = len(shuffled)
        n_train = round(0.70 * n)
        n_val = round(0.15 * n)
        for t in shuffled[:n_train]:
            split_map[t] = "train"
        for t in shuffled[n_train:n_train + n_val]:
            split_map[t] = "val"
        for t in shuffled[n_train + n_val:]:
            split_map[t] = "test"
    table["split"] = table["template_id"].map(split_map)
    return table


def load_bbq(categories=DEFAULT_CATEGORIES, seed=SEED):
    """Returns the full BBQ table with the split column. One row per (context, question, answer)."""
    meta = pd.read_csv(META_URL)
    meta_small = meta[KEYS + ["target_loc"]].drop_duplicates(subset=KEYS)

    tables = []
    for cat in categories:
        d = pd.read_json(BASE_URL + cat + ".jsonl", lines=True)
        m = d.merge(meta_small, on=KEYS, how="left")
        m = m.dropna(subset=["target_loc"])
        m["target_loc"] = m["target_loc"].astype(int)
        tables.append(_build_rows(m))

    table = pd.concat(tables, ignore_index=True)
    return _add_splits(table, categories, seed)
