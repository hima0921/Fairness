# Data Exploration (BBQ)

This is a short write-up of what we checked in the BBQ data and what we found. The code is in `notebooks/01_bbq_exploration.ipynb`. For what each column in our final table means, see [schema.md](schema.md).

## Files we used

All from the BBQ repo: https://github.com/nyu-mll/BBQ

- The 4 category files from the `data/` folder: `Gender_identity.jsonl`, `Race_ethnicity.jsonl`, `Age.jsonl`, `SES.jsonl`
- `analysis_scripts/additional_metadata.csv`, for the `target_loc` column

## Why we use target_loc from the metadata file

To label an answer as biased we need to know which answer option is the stereotype target. The data files do have a `stereotyped_groups` field, but it uses short labels like "F", while `answer_info` describes the answer options with words like "woman". Matching those two directly is messy and easy to get wrong. The metadata file has `target_loc`, which just gives the index (0, 1 or 2) of the stereotype-consistent answer, so we join that in instead.

**Careful: `target_loc` already accounts for question polarity.** On a negative question it points to the stereotyped group (e.g. "the grandfather" for "Who was not comfortable using the phone?"). On a non-negative question it points to the *other* person, because that is the stereotype-consistent answer there (e.g. "the grandson" for "Who was giving the other the opportunity to learn how to use the phone?"). We checked this on the data: for negative questions `target_loc` points to a group in `stereotyped_groups` ~100% of the time, for non-negative questions 0% of the time. BBQ's own scorer (`analysis_scripts/BBQ_calculate_bias_score.R`) uses it the same way, counting `neg_Target + nonneg_Target` as biased. So the loader uses `target_loc` directly as the stereotype-consistent answer and does not flip it by polarity. (An earlier version of the loader did flip it, which inverted `is_biased`, `anti_stereotype` and `is_stereotype_answer` on every non-negative question.)

## What we found

- **All 4 versions are there.** Every template in all 4 categories has all 4 versions (ambig/disambig × neg/nonneg).
- **Template counts:** Gender_identity 50, Race_ethnicity 50, Age 25, SES 25.
- **Duplicates in the metadata file.** It has 64 duplicate Gender_identity rows (the same question listed twice, with the same `target_loc` both times). We removed them before the join. Without that, the join creates extra rows (5736 instead of 5672 for Gender_identity).
- **Missing target_loc.** 16 Gender_identity questions have no `target_loc`. They're all template 13, the "Who is really emotional?" / "Who is really logical?" question, which is 4 full sets of the 4 versions. Without a target we can't label bias, so we dropped them.
- **"unknown" is always there.** We found the "unknown" answer option in every question.
- **Final row counts** (3 rows per question):

  | Category | Rows |
  |---|---|
  | Gender_identity | 16968 |
  | Race_ethnicity | 20640 |
  | Age | 11040 |
  | SES | 20592 |
  | **Total** | **69240** |

- **Share of biased rows is about 0.25 in every category** (exactly 0.25 for Age, Gender_identity and Race_ethnicity; 0.248 for SES, where 51% rather than 50% of disambiguated questions have the stereotype-consistent answer as the correct one). This is what we expected. In ambiguous contexts the correct answer is "unknown" and one of the other two options follows the stereotype, so 1/3 of rows are biased. In disambiguated contexts, if the correct answer is the stereotyped one there's no biased option, otherwise there's one, so it's 1/6 on average. Half ambiguous and half disambiguated gives (1/3 + 1/6) / 2 = 0.25.

## Note on Age and SES

Age and SES only have 25 templates each. Since we split by template, their test sets will be small (about 4 templates each), so results on those two categories will be noisy. That's one reason we also report grouped 5-fold cross-validation (grouped by template).
