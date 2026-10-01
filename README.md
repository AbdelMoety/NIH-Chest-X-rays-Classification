# NIH Chest X-Ray Multi-Label Classification

A DenseNet-121 based deep learning model that detects 14 thoracic abnormalities from frontal-view chest X-rays, trained on the NIH ChestX-ray14 dataset. The pipeline handles patient-level data splitting to prevent leakage, two-phase transfer learning, class-imbalance-aware loss, multi-GPU training, and a full evaluation suite (AUROC, AUPRC, Precision, Recall, F1).

> **Disclaimer:** This is a research/educational project. It is **not** a clinical diagnostic tool and has not been validated for medical use.

## Dataset

[NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) — 112,120 frontal-view X-ray images from 30,805 unique patients, multi-labeled across 14 findings:

`Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax`

**Splitting strategy:**
- The official NIH train/val vs. test split (`train_val_list.txt` / `test_list.txt`) is used so the test set stays completely untouched during model development.
- The train/val portion is further split 75/25 into training and validation **by patient ID** (not by image), so no single patient's images leak across splits — important since patients can have multiple X-rays in the dataset.

## Model

- **Backbone:** DenseNet-121, ImageNet-pretrained (`torchvision.models.densenet121`)
- **Head:** Original 1000-class classifier replaced with a single linear layer with 14 outputs, trained as independent per-class sigmoid outputs (multi-label, not multi-class)

## Training Pipeline

**1. Warm-up phase** — backbone frozen, only the new classifier head is trained for 2 epochs (`lr=1e-3`) to avoid destroying pretrained features with large early gradients.

**2. Fine-tuning phase** — full network unfrozen and trained with **discriminative learning rates**:
- Backbone: `lr=1e-5` (small, gentle updates to pretrained features)
- Classifier head: `lr=1e-4` (faster, since it's randomly initialized)
- Optimizer: AdamW, `weight_decay=1e-4`
- Early stopping on validation Macro AUROC (patience = 5 epochs)

**3. Loss function** — a custom multi-label **focal loss** combining two complementary imbalance-handling mechanisms:
- `pos_weight` per class (from the class's negative:positive ratio) — up-weights rare positive findings
- Focal modulation `(1 - p_t)^γ`, `γ=2.0` — down-weights easy, already-well-classified examples so gradient signal focuses on hard cases

**4. Mixed precision training** (`torch.amp.autocast` + `GradScaler`) for speed and memory efficiency.

**5. Multi-GPU training** via `nn.DataParallel` (developed/tested on Kaggle's 2× T4 GPU instances).

**6. Checkpointing & resume** — the best model (by validation Macro AUROC) is saved every epoch it improves, including model, optimizer, and AMP scaler state, so training can resume from a Kaggle session cut off by GPU-hour limits without starting over.

**Data augmentation** (training only): resize to 224×224, random rotation (±10°), random translation (±5%), subtle brightness/contrast jitter, ImageNet mean/std normalization. Validation/test use resize + normalization only.

## Evaluation

The final checkpoint is evaluated once on the held-out NIH test set (never used in training or model selection). Two families of metrics are reported:

- **Threshold-independent** (AUROC, AUPRC): computed directly from predicted probabilities.
- **Threshold-dependent** (Precision, Recall, F1): reported at two operating points —
  - **Tuned per class**: threshold chosen to maximize F1 on the *validation* set, then applied unchanged to the test set (no test-set tuning).
  - **Fixed 0.5**: included as a naive baseline for comparison.

### Results — tuned per-class thresholds

| Class | AUROC | Precision | Recall | F1 | AUPRC | Threshold | Prevalence |
|---|---|---|---|---|---|---|---|
| Atelectasis | 0.7357 | 0.2527 | 0.5965 | 0.3550 | 0.2936 | 0.5602 | 0.1281 |
| Cardiomegaly | 0.8626 | 0.3303 | 0.4060 | 0.3642 | 0.2950 | 0.6937 | 0.0418 |
| Consolidation | 0.7048 | 0.1314 | 0.5383 | 0.2113 | 0.1340 | 0.5617 | 0.0709 |
| Edema | 0.8277 | 0.1540 | 0.4151 | 0.2246 | 0.1445 | 0.7199 | 0.0361 |
| Effusion | 0.7949 | 0.3706 | 0.7005 | 0.4848 | 0.4469 | 0.5400 | 0.1820 |
| Emphysema | 0.8604 | 0.2974 | 0.4465 | 0.3570 | 0.2877 | 0.6794 | 0.0427 |
| Fibrosis | 0.7779 | 0.0906 | 0.1149 | 0.1013 | 0.0636 | 0.7081 | 0.0170 |
| Hernia | 0.8907 | 0.7500 | 0.1744 | 0.2830 | 0.2950 | 0.9192 | 0.0034 |
| Infiltration | 0.6795 | 0.3044 | 0.8084 | 0.4423 | 0.3736 | 0.4923 | 0.2388 |
| Mass | 0.7469 | 0.2223 | 0.3484 | 0.2715 | 0.2069 | 0.6008 | 0.0683 |
| Nodule | 0.7081 | 0.2188 | 0.2502 | 0.2334 | 0.1813 | 0.5614 | 0.0634 |
| Pleural_Thickening | 0.7336 | 0.1207 | 0.3963 | 0.1850 | 0.1105 | 0.5804 | 0.0447 |
| Pneumonia | 0.6703 | 0.0603 | 0.0937 | 0.0734 | 0.0416 | 0.7155 | 0.0217 |
| Pneumothorax | 0.8321 | 0.4035 | 0.4473 | 0.4243 | 0.3663 | 0.6040 | 0.1041 |
| **Macro** | **0.7732** | **0.2648** | **0.4098** | **0.2865** | **0.2315** | — | — |

**Summary:** Macro AUROC **0.7732** · Macro AUPRC **0.2315** · Macro F1 (tuned) **0.2865** · Macro F1 (0.5 threshold) **0.2490**

### Notes on interpreting the results

- **AUROC vs. AUPRC:** AUROC stays relatively stable (0.67–0.89) across classes regardless of rarity, but AUPRC drops sharply for rare findings (e.g. Hernia: AUROC 0.89 but AUPRC 0.30) because AUPRC's baseline for a random classifier equals class prevalence — Hernia is positive in only 0.34% of test images, so its AUPRC is impressive relative to that baseline even though the absolute number looks unimpressive next to AUROC.
- **Tuned vs. fixed threshold:** because the loss function up-weights positive/rare classes, predicted probabilities skew upward, so a flat 0.5 cutoff over-predicts positives (high recall, poor precision — see Edema: 0.5 threshold gives 88.8% recall but only 7.9% precision). Per-class F1-tuned thresholds (chosen on validation data, applied blind to test data) rebalance this and raise Macro F1 from 0.2490 → 0.2865.
- **Weakest classes** (Pneumonia, Fibrosis, Consolidation) have both low prevalence and known low inter-radiologist agreement in the original NIH labels, which caps achievable performance regardless of model choice.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── nih-x-ray.ipynb       # data prep, training (warm-up + fine-tune), evaluation, checkpointing
└── model/                # local only — gitignored; checkpoint hosted on Hugging Face (see below)
```

## Model Weights

The final trained checkpoint is hosted on the Hugging Face Hub (MIT licensed, same as this repo):

**[Download the model checkpoint](https://huggingface.co/AbdelMoety1/nih-chest-xray-densenet121/resolve/main/NIH%20Chest%20X-Ray%20model.pt)**

```python
from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download(
    repo_id="AbdelMoety1/nih-chest-xray-densenet121",
    filename="NIH Chest X-Ray model.pt"
)
checkpoint = torch.load(checkpoint_path, map_location="cpu")
```

The checkpoint dict contains:
```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scaler_state_dict": ...,
    "best_macro_auroc": float
}
```

## Setup & Usage

**Environment:** Python 3.10+, PyTorch with CUDA, developed on Kaggle notebooks (2× T4 GPU).

```bash
pip install -r requirements.txt
```

`requirements.txt` (core dependencies):
```
torch
torchvision
torchmetrics
torchinfo
scikit-learn
pandas
numpy
seaborn
matplotlib
Pillow
tqdm
```

**To reproduce training:**
1. Download the [NIH ChestX-ray14 dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data).
2. Update the dataset paths at the top of `nih-x-ray.ipynb` to match your environment.
3. Run cells top to bottom. To resume an interrupted run, point `CHECKPOINT_PATH` at your last saved checkpoint before running the training cells.

**To reproduce evaluation only:**
1. Run the notebook cells from *Imports* through the `base_model` setup cell (skip the checkpoint/warm-up/training cells).
2. Point `FINAL_MODEL_PATH` at your trained checkpoint.
3. Run the evaluation cells — this generates the per-class and macro metrics tables shown above.

## Limitations & Future Work

- `DataParallel` (not `DistributedDataParallel`) was used for simplicity on Kaggle; DDP would scale more efficiently to more GPUs.
- Classification thresholds are tuned post-hoc by maximizing F1 on the validation set; no probability recalibration (e.g. Platt scaling, isotonic regression) was applied.
- No test-time augmentation or model ensembling.
- Trained at 224×224 resolution; some fine-grained findings (e.g. small nodules) may benefit from higher resolution.
- NIH ChestX-ray14 labels were extracted via NLP from radiology reports rather than expert-adjudicated per image, which caps the achievable ceiling for any model trained on them — a known limitation of this dataset, not specific to this implementation.

## Citation

If you use this dataset, please cite the original NIH release:

```
Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest
X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of
Common Thorax Diseases. CVPR 2017.
```

## License

This project's code and the trained model weights are released under the **MIT License** — see [`LICENSE`](LICENSE).

Note that the MIT license covers this code and the resulting model weights only, not the NIH ChestX-ray14 dataset itself, which has its own usage terms — see the [NIH data use page](https://nihcc.app.box.com/v/ChestXray-NIHCC) before redistributing images or data derived from them.