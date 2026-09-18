# DAColor-Hierarchical final result

The frozen best version uses joint high-order-prior and DAColor reranking for query sizes 1 and 2, and hierarchical high-order backoff for query size 3. Every numerical weight is selected on validation data inside each fold.

| Query size | Acc@1 | Acc@5 | Acc@10 | MRR |
|---:|---:|---:|---:|---:|
| 1 | 0.23399 | 0.47737 | 0.57887 | 0.35707 |
| 2 | 0.20568 | 0.42834 | 0.54036 | 0.31989 |
| 3 | 0.17046 | 0.38338 | 0.50486 | 0.28141 |
| Overall | 0.19706 | 0.42016 | 0.53385 | 0.31184 |

The exact fold metrics and validation-selected weights are recorded in `results/dacolor_final_cv_summary.json`.

Because this final micro-track was selected after exploratory work, a publication-level confirmatory claim should be checked on untouched paper-grouped folds or additional pre-declared random seeds.
