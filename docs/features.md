# Features

This explains the features we give to the bias detector and why we picked them. The code is in [notebooks/03_features.ipynb](../notebooks/03_features.ipynb). It runs in Google Colab with a T4 GPU, and it reads `bbq_table.csv` from the team Google Drive and saves the results back there.

Every row of the table (one context + question + answer option) becomes one feature vector of 1158 numbers.

## What the features are

| Part | Size |
|---|---|
| Context embedding | 384 |
| Question embedding | 384 |
| Answer embedding | 384 |
| Similarities (context-question, context-answer, question-answer) | 3 |
| NLI probabilities (contradiction, entailment, neutral) | 3 |
| **Total** | **1158** |

## Why each part

- **Embeddings (`all-MiniLM-L6-v2`).** These turn each text into 384 numbers that describe its meaning. We chose this model because it's fast, widely used, and its output is small (only 384 numbers). The embeddings tell the detector who the answer is about and what the question is about, which it needs for the "stereotype" part of the label.
- **Similarities.** Cosine similarity between each pair of the three texts. This tells the detector directly how related the texts are, so it doesn't have to work this out from the raw embeddings. But a similarity can't tell which person the context supports, so these probably only help a little.
- **NLI (`cross-encoder/nli-deberta-v3-base`).** This checks whether the context supports the answer. The premise is the context and the hypothesis is a statement made from the question and answer (see below). This covers the "wrong" part of the label.
- **Why we need both.** In an ambiguous context, "The man was the secretary" and "The woman was the secretary" both get mostly neutral NLI scores, because the context doesn't support either one. So NLI alone can't tell which wrong answer is the biased one. The embeddings can, because they say who the answer is about.

## How statements are made

The NLI model needs one sentence to check against the context, so we turn each question + answer into a statement. There's one rule per question type:

| Question type | Example statement |
|---|---|
| Who | "Who was the secretary?" + "The woman" → "The woman was the secretary." |
| Whose | "The man's job is suffering due to family obligations." |
| What group of people | "The group of people that uses drugs is poor people." |
| Which parents | "The wealthy parents are uninvolved in their children's schooling." |
| Unknown answer (any question type) | "It is not known who was the secretary." |

For unknown answers, we first tried using the raw answer text, but that gave nonsense sentences like "Not enough information was the secretary." So if `is_unknown` is 1, we always make "It is not known ..." instead. Using `is_unknown` here doesn't leak the label. It only says which option is the "can't tell" option, not whether that answer is correct or biased.

How many rows have each question type:

| Question starts with | Rows |
|---|---|
| Who | 68256 |
| Whose | 624 |
| Which | 216 |
| What | 144 |

## NLI label order

For this model, column 0 = contradiction, 1 = entailment, 2 = neutral (we checked with `id2label` in the notebook). In `features.npy`, the last 3 columns are in this same order.

## NLI check results

Average NLI probabilities, grouped by context type, answer type and whether the answer is correct:

| Context | Answer | Correct? | Contradiction | Entailment | Neutral |
|---|---|---|---|---|---|
| ambig | person | wrong | 0.13 | 0.03 | 0.83 |
| ambig | unknown | correct | 0.42 | 0.01 | 0.57 |
| disambig | person | wrong | 0.51 | 0.07 | 0.41 |
| disambig | person | correct | 0.12 | 0.23 | 0.64 |
| disambig | unknown | wrong | 0.52 | 0.02 | 0.46 |

What this means: on average NLI separates correct from wrong person answers (for example, contradiction is 0.51 for wrong vs 0.12 for correct in disambiguated contexts), and person answers are the only ones that can be biased. It doesn't separate right from wrong unknown answers (0.42 vs 0.52 contradiction), but that matters little, because unknown answers are never labeled biased and are easy to recognize from the answer embedding.

## Limitations

- The NLI model is cautious with indirect evidence. In the secretary example, the disambiguated context says the man took notes, but the model rated "The man was the secretary" as neutral, not entailment.
- It handles the "It is not known ..." statements poorly.
- The NLI model learned from human-written data, so it could carry its own biases.

## Output files

These are saved to the team Google Drive, not the repo, because `features.npy` is about 320 MB.

`features.npy` doesn't depend on the labels, so it didn't need to be recomputed after the label fix in PR #6, but `features_meta.csv` was regenerated with the corrected `is_biased`.

- **`features.npy`**: 69240 x 1158, float32. Columns:

  | Columns | What |
  |---|---|
  | 0-383 | Context embedding |
  | 384-767 | Question embedding |
  | 768-1151 | Answer embedding |
  | 1152, 1153, 1154 | sim_cq, sim_ca, sim_qa |
  | 1155, 1156, 1157 | NLI contradiction, entailment, neutral |

- **`features_meta.csv`**: columns `row_id`, `split`, `is_biased`, `category`, `context_condition`, `statement`. Row i of the CSV goes with row i of `features.npy`.

## How to load it

```python
import numpy as np
import pandas as pd

folder = "/content/drive/MyDrive/CSE 575 - Fairness/"
X = np.load(folder + "features.npy")
meta = pd.read_csv(folder + "features_meta.csv")

train = (meta["split"] == "train").values
X_train = X[train]
y_train = meta.loc[train, "is_biased"].values

# same for val and test
val = (meta["split"] == "val").values
X_val, y_val = X[val], meta.loc[val, "is_biased"].values

test = (meta["split"] == "test").values
X_test, y_test = X[test], meta.loc[test, "is_biased"].values
```

## Idea for later

If there's time: train once with only the last 6 columns (similarities + NLI) and once with all 1158, to see how much the embeddings add.
