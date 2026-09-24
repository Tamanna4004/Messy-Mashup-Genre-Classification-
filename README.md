# 🎵 Messy Mashup — Music Genre Classification

A deep learning project that classifies music genres from **noisy, multi-stem audio mashups**. The challenge: each test audio file is a blend of stems (drums, bass, vocals, others) from multiple songs of the same genre, layered with real-world environmental noise from the ESC-50 dataset.

---

## 📁 Project Structure

```
messy_mashup/
├── genres_stems/          # Training data — 10 genres × 100 songs × 4 stems each
│   ├── blues/
│   ├── classical/
│   └── ...
├── ESC-50-master/
│   └── audio/             # Environmental noise files (.wav)
├── mashups/               # Test audio files (noisy mashups to classify)
└── test.csv               # Test file index with filenames and IDs
```

---

## 🧩 Problem Setup

| Property | Detail |
|---|---|
| Task | 10-class audio genre classification |
| Input | 5-second audio clips → 128-band Mel spectrograms |
| Training data | 4,000 stems (on-the-fly synthetic mashups) |
| Test data | ~3,020 real noisy mashups |
| Evaluation metric | Macro F1 Score |
| Target | F1 ≥ 0.80 |

**Genres:** Blues · Classical · Country · Disco · Hip-Hop · Jazz · Metal · Pop · Reggae · Rock

---

## ⚙️ Data Pipeline — `MessyDataset`

A custom PyTorch `Dataset` that synthesises training examples on the fly:

1. **Pick** a random genre (the label).
2. **Sample** 4 different song folders from that genre.
3. **Load** one stem (`drums`, `vocals`, `bass`, `others`) from each folder.
4. **Mix** the 4 stems by averaging them (preserves volume, avoids clipping).
5. **Add ESC-50 noise** at a random SNR between 5–20 % (the "messy" part).
6. **Convert** to a 128-band Mel spectrogram, normalise to `[0, 1]`.

This mirrors exactly how the test mashups were constructed, making the model robust to real noise.

---

## 🤖 Models

### Model 1 — XGBoost (Classical ML Baseline)
Hand-crafted features (MFCCs, Spectral Centroid, Chroma, Spectral Rolloff) are extracted and summarised as mean/std vectors, then fed to an XGBoost classifier.

- **Strength:** Fast to train, interpretable.
- **Weakness:** Can't capture temporal structure or complex patterns.
- **Result:** Macro F1 ≈ 0.41

---

### Model 2 — Simple CNN
A 3-block convolutional network (`Conv2d → BatchNorm → ReLU → MaxPool`) that treats the Mel spectrogram as an image.

- **Strength:** Learns spatial texture patterns in the spectrogram.
- **Weakness:** No temporal memory — misses rhythm and how music evolves over time.
- **Result:** Macro F1 ≈ 0.26

---

### Model 3 — CRNN (CNN + Bidirectional LSTM)
Combines convolutional feature extraction with a 2-layer bidirectional LSTM to model temporal sequences.

- **Architecture:** 3 × `(Conv2d → BN → ReLU → MaxPool)` → Bi-LSTM (128 hidden, 2 layers) → FC head.
- **Strength:** Captures both texture (CNN) and rhythm (LSTM).
- **Weakness:** Still struggles with acoustically similar genre pairs (Rock/Metal, Jazz/Classical).
- **Result:** Macro F1 ≈ 0.54

---

### Model 4 — Pretrained AST Transformer ⭐ Best Model
Fine-tuned Audio Spectrogram Transformer pre-trained on AudioSet (`MIT/ast-finetuned-audioset-10-10-0.4593`).

- **Fine-tuning details:** AdamW (lr = 2e-5, weight_decay = 0.01), CosineAnnealingLR over 6 epochs.
- **Input padding:** Mel spectrograms are zero-padded from 216 → 1024 time steps to match AST's expected input shape.
- **Strength:** Leverages massive pre-training; understands complex audio patterns out of the box.
- **Result:** Macro F1 ≈ 0.95

---

### Model 5 — Ensemble (AST + CRNN)
Blends the softmax probability outputs of the AST and CRNN models:

```
final_prob = 0.9 × P(AST) + 0.1 × P(CRNN)
```

AST gets higher weight due to its superior individual performance. The ensemble resolves edge cases where AST is uncertain by consulting the CRNN's rhythm-based representation.

---

## 📊 Results Summary

| Model | Macro F1 |
|---|---|
| XGBoost (Baseline) | 0.414 |
| Simple CNN | 0.259 |
| CRNN | 0.538 |
| AST Transformer | 0.946 |
| **Ensemble (AST + CRNN)** | **Best** |

---

## 🚀 Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU:** For PyTorch with CUDA support, follow the official guide at https://pytorch.org/get-started/locally/

### 2. Authenticate with Hugging Face

The AST model is downloaded from the Hugging Face Hub. You'll need a token:

```python
# Option A — environment variable (recommended for local use)
import os
os.environ["HF_TOKEN"] = "your_token_here"

# Option B — huggingface_hub login
from huggingface_hub import login
login(token="your_token_here")
```

> **On Kaggle:** The notebook uses `kaggle_secrets` to fetch the token securely. This is a Kaggle-internal package and is not pip-installable outside of Kaggle.

### 3. Update data paths

At the top of the notebook, update these variables to point to your local data:

```python
data_path   = '/path/to/messy_mashup'
genre_path  = '/path/to/messy_mashup/genres_stems'
noise_path  = '/path/to/messy_mashup/ESC-50-master/audio'
test_path   = '/path/to/messy_mashup/mashups'
```

### 4. Run the notebook

Execute cells top-to-bottom. Each model section is self-contained and saves its best checkpoint:

- `model_scratch.pth` — Simple CNN weights
- `best_crnn_model.pth` — Best CRNN weights (by validation F1)
- `best_ast_model.pth` — Best AST weights (by validation F1)

### 5. Generate submission

The final cell runs `run_final_submission()`, which loads the best AST and CRNN weights, runs the ensemble over `test.csv`, and writes `submission.csv`.

---

## 🛠️ Key Design Decisions

- **On-the-fly data synthesis** — The `MessyDataset` generates 3,000 unique training mashups per epoch, providing massive effective data augmentation without storing files.
- **Mixed-precision training** — `torch.cuda.amp.autocast` and `GradScaler` halve GPU memory usage and speed up training on compatible hardware.
- **Noise SNR randomisation** — Mixing ESC-50 noise at a random 5–20 % intensity ensures the model isn't tuned to a single noise level.
- **Graceful fallback** — Corrupted or missing audio files return silence (`np.zeros`) so training never crashes mid-epoch.

---

## 📦 Dependencies

See `requirements.txt` for the full list. Key packages:

| Package | Purpose |
|---|---|
| `torch`, `torchaudio` | Model training and GPU acceleration |
| `librosa` | Audio loading and Mel spectrogram extraction |
| `transformers` | Pre-trained AST model |
| `xgboost` | Classical ML baseline |
| `scikit-learn` | Metrics, preprocessing, train/val split |
| `wandb` | Experiment tracking (optional — cells are commented out) |
| `huggingface_hub` | Downloading pre-trained weights |

---

## 📝 Notes

- W&B logging cells are present in the notebook but **commented out**. To enable experiment tracking, uncomment those blocks and log in with `wandb.login()`.
- The notebook was originally developed on **Kaggle** with a P100 GPU. Training times will vary on other hardware.
- The `kaggle_secrets` import block is Kaggle-specific — replace it with `os.environ` when running locally (see Setup step 2).
