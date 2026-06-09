#!/usr/bin/env bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python - <<'PY'
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from textblob import download_corpora
download_corpora.download_all()
print("Downloaded NLTK & TextBlob corpora.")
PY
echo "Setup finished. Activate venv with: source venv/bin/activate"
echo "To train: python model_training.py"
echo "To run app: python app.py"
