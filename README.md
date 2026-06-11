---
title: NewsTruth Classifier
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# NewsTruth v3

Lightweight fake-news detection demo:
- Local classifier (Word2Vec + sklearn) + heuristics
- Gemini single-word cross-check (Real/Fake/Uncertain) — optional (requires GEMINI_API_KEY)
- Flag mismatches to Firestore (or local JSON fallback) for manual labeling & retraining
- Per-model confusion matrices, classification reports, and automatic best-model selection

## Setup (macOS / Linux)

1. Open terminal in `newstruth_v3/backend`
2. Run:
   ```bash
   ./run_demo.sh
   source venv/bin/activate

3. Set env vars (optional):
export GEMINI_API_KEY="your_gemini_key"
export NEWS_API_KEY="your_newsapi_key"  # optional
export FIREBASE_CRED_JSON="/path/to/serviceAccountKey.json"  # optional

4. Place news_dataset_cleaned.csv into backend/data/.

5. Train models:  Artifacts will appear under backend/models/ (metrics, confusion matrices, best model).
python model_training.py

6. Run the app:
python app.py

Visit: http://127.0.0.1:5500

Demo flow

Submit a headline + optional URL + author.

App shows local model verdict and Gemini verdict.

Mismatches are saved to Firestore or backend/data/flagged_local.json.

