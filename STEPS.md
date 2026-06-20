# ✅ Run-it-yourself guide

Everything is built. Here's exactly what **you** do — about 20–25 minutes, most of it Colab training while you wait.

---

## Part 1 — Train the model on Colab (free GPU)

1. **Get a Kaggle API token.** Go to [kaggle.com](https://www.kaggle.com) → your avatar → **Settings** → **API** → **Create New API Token**. This downloads `kaggle.json`.
2. **Open the notebook in Colab** — click the **“Train in Colab”** badge in the README (or open `notebooks/Pneumonia_Detection_Colab.ipynb` and use *Open in Colab*).
3. In Colab: **Runtime → Change runtime type → GPU (T4)**.
4. **Run all cells** top to bottom (`Runtime → Run all`). When prompted, **upload `kaggle.json`**. The notebook will:
   - download the dataset from Kaggle,
   - train (feature-extraction → fine-tuning),
   - evaluate (accuracy, ROC-AUC, confusion matrix),
   - generate **Grad-CAM** heatmaps,
   - and at the end **download `pneumonia_outputs.zip`**.

> ⏱️ Training is ~10–20 min on the free T4 GPU.

---

## Part 2 — Put the results into the repo

5. **Unzip `pneumonia_outputs.zip`.** Copy:
   - `pneumonia_efficientnet.keras` → **`app/`**
   - everything in `figures/` (`training_history.png`, `confusion_matrix.png`, `roc_curve.png`, `gradcam_examples.png`, `sample_xrays.png`) → **`assets/`**
6. Open `outputs/metrics.txt` and **fill the Results table** in `README.md` (accuracy, ROC-AUC, recall, precision, F1).

---

## Part 3 — Try the demo locally (optional but worth a screenshot)

7. Install deps and launch:
   ```bash
   pip install -r requirements.txt
   streamlit run app/app.py
   ```
8. Upload any chest X-ray → see the prediction + Grad-CAM heatmap. **Screenshot it → `assets/demo.png`** (the README shows it).

---

## Part 4 — Publish

9. Commit & push:
   ```bash
   git add -A
   git commit -m "Add trained model, metrics, and result figures"
   git push
   ```

**Easiest option:** just send me **`pneumonia_outputs.zip`** (+ a demo screenshot if you have one) and I'll drop everything in, fill the metrics, and push the finished repo for you.

---

*Reminder: educational project — not for clinical use.*
