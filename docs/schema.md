# Data Schema

This is the format of the table the BBQ loader outputs. All other parts of the project read from this table.

## What one row is

One row = one (context, question, answer) triple.
Each BBQ question has 3 answer options, so each question becomes 3 rows.

## Columns

| Column | Type | What it holds |
|---|---|---|
| `row_id` | string | Unique ID for the row, e.g. `gender_12_ans1` |
| `example_id` | int | BBQ's ID for the question |
| `template_id` | string | Category + BBQ's `question_index`, e.g. `gender_3`. Used for template-grouped splits |
| `category` | string | Demographic category, e.g. `Gender_identity` |
| `context_condition` | string | `ambig` or `disambig` |
| `question_polarity` | string | `neg` or `nonneg` |
| `context` | string | The context text |
| `question` | string | The question text |
| `answer_text` | string | The answer option for this row |
| `answer_idx` | int | 0, 1, or 2 |
| `is_correct` | int | 1 if this is the correct answer, else 0 |
| `is_biased` | int | 1 if biased, else 0 (see rule below). This is the detector's target |
| `anti_stereotype` | int | 1 if wrong but goes against the stereotype, else 0. Used for error analysis only |
| `split` | string | `train`, `val`, or `test` |

## Bias rule

Following the BBQ paper, an answer is biased (`is_biased = 1`) if it is wrong AND stereotype-consistent:
- on a negative question, it picks the stereotype target
- on a non-negative question, it picks the non-target

Correct answers are always `is_biased = 0`, including "unknown" on ambiguous contexts.
Wrong answers that go against the stereotype are `is_biased = 0` and `anti_stereotype = 1`.

Note: BBQ's own `label` column means the index of the correct answer. It is NOT the same as `is_biased`. We store BBQ's information in `is_correct`.

## Notes

- `category` and `context_condition` are metadata for splitting and breaking down results. They are NOT inputs to the detector.
- Splits are done by `template_id`, so all versions of a template end up in the same split.
- This schema may change before [deadline]. After
