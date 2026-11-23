# Subject Models

This explains how we got real LLM answers to the BBQ test questions, and how biased those answers are before any fixing. The code is in [notebooks/05_subject_models.ipynb](../notebooks/05_subject_models.ipynb). It runs in Google Colab with a T4 GPU and takes about 20–30 minutes per model.

## Why we need this

The detector and the rewriter were only tested on BBQ's own answer options. To show that the pipeline (detector → rewriter) actually fixes bias in what a real model says, we need answers from a real model. These models are the "subject models": the models we test for bias.

## Models

- **Qwen2.5-3B-Instruct**
- **Qwen2.5-1.5B-Instruct**

We planned to use Llama-3.2-3B-Instruct, but it's gated on Hugging Face (you have to request access) and our request wasn't approved in time. So we used Qwen2.5, which the proposal listed as the second model, in two sizes.

## Setup

- **Test split only:** 3,944 questions. The detector and rewriter were trained on train, so using test keeps the pipeline evaluation fair.
- **Prompt:** the model sees the context, the question and the 3 options as A/B/C (in the same order as `ans0`/`ans1`/`ans2`). It's asked for the letter of the correct answer plus one short sentence explaining why, in the format `<letter>. <reason>`.
- **Greedy decoding** (`do_sample=False`), so the model gives the same answer every time we run it.
- **Max 60 new tokens**, which is enough for a letter and one sentence.
- Questions are sent in batches of 16, and answers are saved to Drive every 160 answers so a Colab disconnect doesn't lose the run.

## Reading the answers

`read_letter()` finds the letter in the reply and turns it into option 0, 1 or 2 (A, B, C), or -1 if there's no letter.

**The "<C>" issue:** the prompt says "Format: <letter>. <reason>", and Qwen-3B sometimes copied the angle brackets literally and wrote things like "<C>. The man mentioned ...". The first version of `read_letter` didn't accept that, so 201 of the 3B answers weren't read at first. We fixed `read_letter` to also accept `<C>`, `(C)` and `[C]` style answers, then re-read the letters from the saved replies and saved the file again. The 1.5B run used the fixed version from the start.

After the fix, **0 answers were unreadable for both models.**

## Output files

Saved in the team Google Drive (`CSE 575 - Fairness`), not the repo:

- `qwen_test_answers.csv` (Qwen-3B)
- `qwen15_test_answers.csv` (Qwen-1.5B)

Both have 3,944 rows and these columns:

| Column | What |
|---|---|
| `question_id` | which test question |
| `chosen_idx` | the option the model picked (0, 1 or 2) |
| `reply` | the model's full reply (letter + reason) |

**How to join them to the table:** match `question_id` + `chosen_idx` to `question_id` + `answer_idx` in `bbq_table.csv` (or `load_bbq()`). That gives each answer its labels: `is_correct`, `is_biased`, `is_stereotype_answer`, `is_unknown`.

```python
ans = pd.read_csv("qwen15_test_answers.csv")
picked = ans.merge(table, left_on=["question_id", "chosen_idx"], right_on=["question_id", "answer_idx"])
```

## Baseline results (no intervention)

| | Qwen-3B ambig | Qwen-3B disambig | Qwen-1.5B ambig | Qwen-1.5B disambig |
|---|---|---|---|---|
| Accuracy | 0.843 | 0.852 | 0.309 | 0.854 |
| Share "unknown" | 0.843 | 0.111 | 0.309 | 0.005 |
| Share biased | 0.106 | 0.009 | 0.408 | 0.082 |
| BBQ bias score | 0.055 | -0.014 | 0.125 | 0.044 |

Letter choice (1.5B): A 0.243, B 0.385, C 0.372. So it doesn't just pick one letter all the time.

**How the BBQ bias score works:** it's 2 × (share of stereotype answers among the answers that aren't "unknown") − 1, computed separately for ambiguous (s_AMB) and disambiguated (s_DIS) contexts. It's 0 if the model is balanced, +1 if it always picks the stereotype when it names someone, and −1 if it always picks the other person. For ambiguous contexts it's multiplied by (1 − accuracy), because there the right answer is always "unknown", so a model that's almost always right shouldn't get a big score from its few mistakes.

## What it means

- **Qwen-3B is mildly biased.** In ambiguous contexts it mostly says "unknown" (84%), but when it does name a person, it picks the stereotype about twice as often as the other person (0.106 vs about 0.05 of rows). In disambiguated contexts it has basically no bias. Its main mistake there is saying "unknown" when the answer is in the text (11%), which is over-caution, not bias.
- **Qwen-1.5B is clearly more biased.** In ambiguous contexts it names a person about 69% of the time and leans toward the stereotype (bias score 0.125). In disambiguated contexts it reads the evidence as well as the 3B (85% correct), but when it's wrong it leans slightly toward the stereotype.
- **Why we use both:** the 1.5B tests whether the pipeline actually reduces bias. The 3B tests whether the pipeline leaves good answers alone (no over-correction). Both have about 85% accuracy on disambiguated contexts, and the pipeline should not lower that.

## Limitations

- BBQ is a well-known benchmark, so newer models may have been tuned toward saying "unknown" on it. This test may understate how biased the models really are.
- It's multiple-choice only. The papers in our proposal found more bias in open-ended answers.
- We used one prompt wording. Results could change with a different prompt.
