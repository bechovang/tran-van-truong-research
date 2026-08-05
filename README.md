# tran-van-truong-research

Adaptation experiments for the **JOUR-2026 / sheet "Implement" (rows 11–15)** task: applying the core idea of 5 papers onto the team's unified setup and running on Kaggle to collect metrics.

> This is **adaptation, not full reproduction** — each paper's method is ported onto the shared protocol below.

## Unified protocol

| Item | Standard value |
|---|---|
| Base model | **Qwen2.5-VL-3B** (4-bit for inference-only) |
| Dataset | **Visual CoT–GQA** (GQA subset of Visual CoT) |
| Eval subset | **n = 200** |
| Training (when applicable) | train **256**, eval **200**, **QLoRA r=8**, ~32 steps |
| Metrics | Acc, Prec, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR, Train Time, Params, Steps |

## The 5 papers (rows 11–15)

| # | Paper | Approach |
|---|---|---|
| 11 | **COCO-Tree** (EMNLP'25) | concept-tree + beam-search augmentation, inference-only |
| 12 | **LLaVA-CoT** (ICCV'25) | 4-stage reasoning SFT (Summary/Caption/Reasoning/Conclusion) — best protocol fit |
| 13 | **Pix2Graph / PGSG** (CVPR'24) | image→scene-graph, then graph-augmented VQA |
| 14 | **MR-MKG** (ACL'24) | MMKG adapter + cross-modal alignment |
| 15 | **LLM4SGG** (CVPR'24) | LLM CoT few-shot triplet extraction → graph-augmented VQA |

See [`PLAN_Phuc_5papers.md`](./PLAN_Phuc_5papers.md) for the full plan.

## Repository layout

```
code/           Python sources for the Kaggle kernels (edit these, not the .ipynb)
kernels/        Generated Kaggle notebooks (built from code/*.py)
summaries/      Verified paper summaries
Paper-Jour2026/ Source papers (PDF) + reference notebooks
test-dataset-visual-cot.ipynb   Template: load Visual CoT–GQA + run Qwen2.5-VL-3B
```

> **Notebook build pipeline:** `kernels/*.ipynb` are generated from `code/*.py`. Edit the `.py`, never the `.ipynb`.

## Kaggle

Published under the **bechovang** account.
