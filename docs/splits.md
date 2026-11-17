# Train / Validation / Test Splits

This explains how we split the BBQ table into train, val and test, and why. The code is in [notebooks/02_splits.ipynb](../notebooks/02_splits.ipynb). It rebuilds the same table as notebook 01 and then adds a `split` column (`train`, `val` or `test`).

## How we split

- **By template, not by row.** BBQ makes many rows from one template by swapping in different names, so rows from the same template are basically near-copies of each other. If some of them went to train and others went to test, the detector would be tested on sentences it has pretty much already seen, and the scores would look better than they really are (data leakage). So every row from a template goes to the same split.
- **Each category separately.** We split the templates of each category on their own. That way every category shows up in train, val and test, and if we add or drop a category later, the other categories' splits stay the same.
- **70/15/15 of the templates in each category.** We shuffle the templates with a fixed seed, then take 70% for train, 15% for val, and the rest for test.
- **Context types stay balanced on their own.** Every template has both ambiguous and disambiguated versions, so each split gets both kinds without us having to do anything extra.

## The seed

We use `SEED = 600`. We first tried 575, but that gave about 75/14/11 by rows, which was uneven. We switched to 600 before training any model, only to get a more even row split. The seed is now fixed and should not be changed, especially not after seeing results, since picking a seed based on results would be a form of cheating.

## Results

Templates per category:

| Category | Train | Val | Test | Total |
|---|---|---|---|---|
| Gender_identity | 35 | 8 | 7 | 50 |
| Race_ethnicity | 35 | 8 | 7 | 50 |
| Age | 18 | 4 | 3 | 25 |
| SES | 18 | 4 | 3 | 25 |

Rows per split:

| Split | Rows | Share |
|---|---|---|
| Train | 45984 | ~66% |
| Val | 11424 | ~16.5% |
| Test | 11832 | ~17% |
| **Total** | **69240** | |

The row split isn't exactly 70/15/15 because we split templates, not rows, and templates don't all have the same number of rows (some have more name swaps than others).

The share of biased rows is **0.25 in every split**, the same as in the full table.

We also checked that no template is in more than one split and that every row has a split (both are 0).

## Limitations

- **Small test sets for Age and SES.** These two categories have only 25 templates, so their test sets have only 3 templates each. Their test scores can change a lot depending on which templates happened to land in test. That's why we also report grouped 5-fold cross-validation (grouped by template).
- **Same groups in train and test.** Train and test have the same 4 categories, so our results show that the detector works on new questions about groups it has already seen, not on completely new groups. The optional held-out intersectional test set would check that.
- **Only BBQ data.** All of our data is BBQ, which is short, made-up multiple-choice questions. Good results here don't mean the detector would work on real chatbot answers.
- **Possible shortcuts.** The detector might learn shortcuts like "answers about this group tend to be biased" instead of actually checking the context. Splitting by template makes this less likely, but doesn't get rid of it. The per-category false-positive check in evaluation is there to catch this.
