#!/usr/bin/env bash
set -euo pipefail
echo "=== NewsTruth_v3 dependency installer ==="
echo "Working dir: $(pwd)"
echo "Ensure you activated your venv before running: source venv/bin/activate"
echo

# sanity check
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "ERROR: no venv detected. Activate venv first: source venv/bin/activate"
  exit 1
fi

python --version
pip --version
echo

echo "1) upgrade pip, setuptools, wheel, build tools, Cython"
pip install --upgrade pip setuptools wheel build Cython

echo
echo "2) install critical compiled packages first (numpy, scipy, scikit-learn, gensim, imbalanced-learn, lxml)"
# pinned versions chosen for macOS arm64 + Python 3.11 compatibility
pip install --upgrade numpy
pip install --upgrade scipy
pip install --upgrade scikit-learn==1.3.2
pip install gensim==4.3.1 || pip install --no-binary gensim gensim==4.3.1
# imbalanced-learn fallback chain
pip install imbalanced-learn==0.10.1 || pip install imbalanced-learn==0.9.1 || true
# lxml - may require brew libxml2/libxslt if fails
pip install lxml || echo "lxml install failed - please install libxml2/libxslt via brew and re-run (see README)."

echo
echo "3) install everything from requirements.txt except google-generativeai (we'll handle that separately)"
if [ ! -f requirements.txt ]; then
  echo "ERROR: requirements.txt not found in current directory"
  exit 1
fi
grep -v '^google-generativeai' requirements.txt > /tmp/req_no_gai.txt
pip install -r /tmp/req_no_gai.txt || (echo "pip install -r failed; inspect output above" && exit 1)
rm -f /tmp/req_no_gai.txt

echo
echo "4) Attempt to install google-generativeai (normal install)"
set +e
pip install --upgrade google-generativeai
GOK=$?
set -e

if [ $GOK -eq 0 ]; then
  echo "google-generativeai installed successfully (normal)."
else
  echo "Normal pip install google-generativeai failed (exit $GOK). Trying pre-release..."
  set +e
  pip install --pre google-generativeai
  GOK2=$?
  set -e
  if [ $GOK2 -eq 0 ]; then
    echo "google-generativeai installed (pre-release)."
  else
    echo "Pre-release install failed (exit $GOK2). Trying install from GitHub source..."
    set +e
    pip install git+https://github.com/google/generative-ai-python.git
    GOK3=$?
    set -e
    if [ $GOK3 -eq 0 ]; then
      echo "google-generativeai installed from GitHub source."
    else
      echo "All attempts to install google-generativeai failed."
      echo "You can skip it for now; the app will still work with heuristic fallback."
      echo "To debug further, copy the failing pip command output and paste it to me."
    fi
  fi
fi

echo
echo "5) (Optional) If you want tensorflow on Apple Silicon, install these (may be large):"
echo "   pip install tensorflow-macos==2.14.0 tensorflow-metal==1.1.0"
echo "   (only run if you want ANN/GPU support; else skip)"

echo
echo "6) Download NLTK/TextBlob corpora (recommended now):"
python - <<'PY'
import nltk
try:
    nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')
except Exception as e:
    print("NLTK download error:", e)
try:
    from textblob import download_corpora
    download_corpora.download_all()
except Exception as e:
    print("TextBlob corpora download error:", e)
print("Corpus step finished.")
PY

echo
echo "Installation script finished."
echo "Next steps:"
echo "  - Run `python model_training.py` to train models (creates backend/models/)."
echo "  - Run `python app.py` to start the web app (http://127.0.0.1:5500)."
echo
