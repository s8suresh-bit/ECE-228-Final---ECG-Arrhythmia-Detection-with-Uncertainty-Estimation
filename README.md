# ECG Arrhythmia Detection with Uncertainty Estimation

ECE 228 final project — Team 40.

A CNN-BiLSTM-Attention network for AAMI 5-class heartbeat classification (N, S, V, F, Q) on the MIT-BIH Arrhythmia Database, equipped with Monte Carlo (MC) Dropout for per-beat uncertainty estimation. The goal is not just an accurate classifier, but one that "knows when it doesn't know" — enabling a reject-or-classify workflow where the most uncertain beats are deferred to a clinician.

The full write-up, including methodology, figures, and discussion, is in `ECE228_Final_Report_Team40.pdf`.

## Repository Contents

- `ECG_Arrhythmia_Uncertainty.ipynb` — end-to-end Colab notebook covering data download/preprocessing, model definition, training, threshold tuning, evaluation, and MC Dropout uncertainty analysis. Sections are ordered so the notebook can be run top to bottom.
- `ECE228_Final_Report_Team40.pdf` — the final project report.
- `requirements.txt` — Python dependencies.

## Setup

```bash
git clone https://github.com/s8suresh-bit/ECE-228-Final---ECG-Arrhythmia-Detection-with-Uncertainty-Estimation.git
cd ECE-228-Final---ECG-Arrhythmia-Detection-with-Uncertainty-Estimation
pip install -r requirements.txt
```

The notebook downloads the MIT-BIH Arrhythmia Database directly from PhysioNet via the `wfdb` package, so no manual data download is required. If running in Google Colab, the notebook also includes a Google Drive mount cell for caching the downloaded records and saving checkpoints.

## How to Run

Open `ECG_Arrhythmia_Uncertainty.ipynb` (in Jupyter or Colab) and run all cells in order:

1. **Data preparation** — downloads MIT-BIH records, applies a 4th-order Butterworth bandpass filter (0.5–40 Hz), segments beats around each annotated R-peak (200 samples before, 140 after → 340-sample windows), applies median/IQR normalization, computes RR-interval features, and builds the patient-independent train/validation/test (DS1/DS2) split.
2. **Model definition** — builds the CNN-BiLSTM-Attention model (85,814 parameters) with configurable MC Dropout rates.
3. **Training** — trains with a class-weighted, label-smoothed cross-entropy loss, a weighted random sampler, gradient clipping, `ReduceLROnPlateau`, and early stopping on validation macro-F1.
4. **Threshold tuning & evaluation** — grid-searches per-class decision thresholds for the rare S and F classes on the validation set, then evaluates on the held-out DS2 test set (confusion matrix, per-class precision/recall/F1, AUROC).
5. **Uncertainty quantification** — runs T = 50 MC Dropout forward passes to compute predictive entropy and mutual information, calibration (reliability diagram, ECE), and the rejection-accuracy trade-off curve.

All randomness (Python, NumPy, PyTorch, CUDA) is seeded with `42` for reproducibility.

## Key Hyperparameters

| Setting | Value |
|---|---|
| Input window | 340 samples (200 pre- / 140 post-R-peak) |
| Bandpass filter | 4th-order Butterworth, 0.5–40 Hz |
| CNN channels | 1 → 16 → 32 → 64 (kernels 7/5/3) |
| BiLSTM hidden size | 64 |
| Dropout (conv / LSTM / head) | 0.10 / 0.25 / 0.30 |
| Optimizer | AdamW (lr = 5e-4, weight decay = 1e-4) |
| Batch size | 256 |
| Gradient clip | 2.0 |
| LR schedule | ReduceLROnPlateau (factor 0.5, patience 4) |
| Early stopping | patience 12 (on validation macro-F1) |
| Label smoothing | ε = 0.01 |
| MC Dropout passes (T) | 50 |
| ECE bins | 15 |

## Results (DS2 test set)

| Metric | Value |
|---|---|
| Overall accuracy | 88.7% |
| Weighted-F1 | 90.3% |
| Macro-F1 | 40.8% |
| Macro AUROC (one-vs-rest) | 88.3% |
| Expected Calibration Error (ECE) | 0.027 |
| Trainable parameters | 85,814 |

The macro-F1 gap reflects the extreme rarity of the F and Q classes in MIT-BIH. Per-class AUROC remains strong even for these classes, and rejecting the most uncertain beats (via MC Dropout predictive entropy) substantially increases accuracy on the retained set — see the report for full figures and discussion.

## Reference

Moody, G. B., & Mark, R. G. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology Magazine, 20(3), 45–50, 2001.
