# backend/app.py
import os
import json
import logging
import traceback
import numbers
import warnings
from datetime import datetime
from urllib.parse import urlparse

from flask import Flask, request, jsonify, render_template

import numpy as np
import gensim
import pickle

# for Firestore
import firebase_admin
from firebase_admin import credentials, firestore

# optional: google generative ai (gemini)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------------------------
# CONFIG / paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
WORD2VEC_PATH = os.path.join(MODELS_DIR, "word2vec.model")
SCALER_PATH = os.path.join(MODELS_DIR, "heuristic_scaler.pkl")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_combined.pkl")

# Update these if your model uses a different embedding size
WORD2VEC_VECTOR_SIZE = 100
# number of heuristic features (must match training order & count)
N_HEURISTICS = 9

# env keys
FIREBASE_CRED_ENV = "FIREBASE_CRED_JSON"  # path to service account json
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"     # Gemini key (if using)

# flask app
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "..", "frontend"), static_folder=os.path.join(BASE_DIR, "..", "frontend"), static_url_path="")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("newstruth_app")


# ---------------------------
# helpers
# ---------------------------
def to_native(x):
    """Convert numpy/pandas/other types to native Python types (recursively)."""
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (list, tuple, set)):
        return [to_native(i) for i in x]
    if isinstance(x, dict):
        return {str(k): to_native(v) for k, v in x.items()}
    try:
        return json.loads(json.dumps(x, default=str))
    except Exception:
        return str(x)

def simple_tokenize(text):
    if not text:
        return []
    return [w.lower().strip(".,!?;:\"'()[]") for w in text.split() if w.strip()]

def avg_wordvec(tokens, w2v_model, dim=WORD2VEC_VECTOR_SIZE):
    """Compute mean vector for list of tokens using gensim Word2Vec model."""
    vec = np.zeros(dim, dtype=float)
    if tokens is None or len(tokens) == 0 or w2v_model is None:
        return vec
    count = 0
    for w in tokens:
        if w in w2v_model.wv:
            vec += w2v_model.wv[w]
            count += 1
    if count > 0:
        vec /= count
    return vec

def map_label_to_text(raw_label, prob=None, heur_dict=None):
    """
    Normalize a raw model label (could be int, np.int64, str) to "Real" or "Fake".
    If raw_label is not clearly mappable, use prob (if available) then heuristics fallback.
    """
    try:
        if hasattr(raw_label, "item"):
            raw_label = raw_label.item()
    except Exception:
        pass

    if raw_label is None:
        raw_label = ""

    s = str(raw_label).strip().lower()
    if s in ("1", "real", "r", "true"):
        return "Real"
    if s in ("0", "fake", "f", "false"):
        return "Fake"

    # probability fallback
    if prob is not None:
        try:
            prob = float(prob)
            return "Real" if prob >= 0.5 else "Fake"
        except Exception:
            pass

    # heuristics fallback
    if heur_dict:
        if heur_dict.get("domain_reputation", 0) > 0.5 or heur_dict.get("supporting_links", 0) > 0.5:
            return "Real"
        return "Fake"

    return "Fake"

class DummyModel:
    """
    Very small deterministic fallback classifier that mimics sklearn API
    (predict, predict_proba, classes_). Uses heuristic features (last N_HEURISTICS cols)
    to return a stable prediction rather than None.
    """
    def __init__(self):
        # map indices to same convention you use (0 -> Fake, 1 -> Real)
        self.classes_ = np.array([0, 1])

    def predict(self, X):
        # If shape works, use heuristics (last N_HEURISTICS features)
        if X is None or getattr(X, "shape", None) is None or X.shape[0] == 0:
            return np.array([0])
        out = []
        for i in range(X.shape[0]):
            row = X[i, :]
            # assume heuristics are the last N_HEURISTICS features
            heur = row[-N_HEURISTICS:] if row.shape[0] >= N_HEURISTICS else row
            # simple rule: if sum of heuristics > 0 -> Real else Fake
            score = float(np.nansum(np.nan_to_num(heur)))
            out.append(1 if score > 0.0 else 0)
        return np.array(out)

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = []
        for p in preds:
            if p == 1:
                proba.append([0.1, 0.9])
            else:
                proba.append([0.9, 0.1])
        return np.array(proba)

# ---------------------------
# load persisted models
# ---------------------------
w2v = None
scaler = None
best_model = None

def load_models():
    global w2v, scaler, best_model
    # Word2Vec
    if os.path.exists(WORD2VEC_PATH):
        try:
            logger.info("Loading Word2Vec from %s", WORD2VEC_PATH)
            w2v = gensim.models.Word2Vec.load(WORD2VEC_PATH)
        except Exception:
            logger.exception("Failed to load Word2Vec")
            w2v = None
    else:
        logger.warning("Word2Vec model not found at %s", WORD2VEC_PATH)
        w2v = None

    # scaler
    if os.path.exists(SCALER_PATH):
        try:
            logger.info("Loading heuristic scaler from %s", SCALER_PATH)
            with open(SCALER_PATH, "rb") as fh:
                scaler = pickle.load(fh)
        except Exception:
            logger.exception("Failed to load scaler")
            scaler = None
    else:
        logger.warning("Scaler not found at %s", SCALER_PATH)
        scaler = None

    # Try sklearn .pkl first, then TensorFlow .h5
    loaded = False
    if os.path.exists(BEST_MODEL_PATH):
        try:
            logger.info("Loading best model (sklearn/pickle) from %s", BEST_MODEL_PATH)
            with open(BEST_MODEL_PATH, "rb") as fh:
                best_model = pickle.load(fh)
            loaded = True
            logger.info("Loaded sklearn model from pickle.")
        except Exception:
            logger.exception("Failed to load sklearn model from pickle.")

    # Try ANN h5 if pkl missing or failed
    ANN_PATH = os.path.join(MODELS_DIR, "best_model_combined.h5")
    if not loaded and os.path.exists(ANN_PATH) and _HAS_TF:
        try:
            logger.info("Loading ANN model from %s", ANN_PATH)
            best_model = load_model(ANN_PATH)
            # wrap a keras model to present predict/predict_proba-like interface:
            class KerasWrapper:
                def __init__(self, model):
                    self._m = model
                    # pretend classes_
                    self.classes_ = np.array([0, 1])
                def predict(self, X):
                    probs = self._m.predict(X)
                    # handle binary output shape
                    if probs.ndim == 2 and probs.shape[1] == 1:
                        return (probs > 0.5).astype("int32").flatten()
                    if probs.ndim == 2 and probs.shape[1] == 2:
                        return np.argmax(probs, axis=1)
                    return probs
                def predict_proba(self, X):
                    probs = self._m.predict(X)
                    if probs.ndim == 1:
                        # treat as probability of class 1
                        p1 = probs
                        p0 = 1.0 - p1
                        return np.vstack([p0, p1]).T
                    if probs.ndim == 2 and probs.shape[1] == 1:
                        p1 = probs.flatten()
                        p0 = 1.0 - p1
                        return np.vstack([p0, p1]).T
                    return probs
            best_model = KerasWrapper(best_model)
            loaded = True
            logger.info("Loaded ANN and wrapped for sklearn-like API.")
        except Exception:
            logger.exception("Failed to load ANN model.")

    if not loaded:
        logger.warning("No persisted classifier found; using DummyModel fallback.")
        best_model = DummyModel()

# run loader
load_models()

# ---------------------------
# Firestore init
# ---------------------------
db = None
def init_firestore():
    global db
    try:
        cred_path = os.environ.get(FIREBASE_CRED_ENV)
        if cred_path and os.path.exists(cred_path):
            logger.info("Initializing Firestore using credentials at %s", cred_path)
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            return
        candidate = os.path.join(BASE_DIR, "ServiceAccountKeyv3.json")
        if os.path.exists(candidate):
            logger.info("Initializing Firestore using %s", candidate)
            cred = credentials.Certificate(candidate)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            return
        logger.info("Trying Application Default Credentials for Firestore")
        firebase_admin.initialize_app()
        db = firestore.client()
        return
    except Exception as e:
        logger.warning("Failed to initialize Firestore: %s", e)
        db = None

init_firestore()

# ---------------------------
# Heuristics (must be same order used in training)
# ---------------------------
def compute_heuristics(title, content, url, author):
    heur = {}
    words_title = simple_tokenize(title)
    words_content = simple_tokenize(content)
    total_words = max(1, len(words_title) + len(words_content))
    heur["all_caps_words"] = sum(1 for w in words_title + words_content if w.isupper()) / total_words
    heur["author_credibility"] = 1.0 if author and len(author.strip()) > 2 else 0.0
    heur["content_subjectivity"] = 0.0
    heur["domain_reputation"] = 1.0 if url and urlparse(url).netloc.endswith("thehindu.com") else 0.0
    heur["headline_body_sentiment_diff"] = 0.0
    heur["is_old_news"] = 0.0
    heur["is_satire"] = 0.0
    heur["supporting_links"] = 1.0 if content and "http" in content else 0.0
    heur["title_subjectivity"] = 0.0

    arr = np.array([
        heur["all_caps_words"],
        heur["author_credibility"],
        heur["content_subjectivity"],
        heur["domain_reputation"],
        heur["headline_body_sentiment_diff"],
        heur["is_old_news"],
        heur["is_satire"],
        heur["supporting_links"],
        heur["title_subjectivity"]
    ], dtype=float)
    return arr.reshape(1, -1), heur

# ---------------------------
# build feature vector (ensures 2*vec_size + N_HEURISTICS)
# ---------------------------
def build_feature_vector(title, content, url, author):
    title_tokens = simple_tokenize(title)
    content_tokens = simple_tokenize(content)
    title_vec = avg_wordvec(title_tokens, w2v, dim=WORD2VEC_VECTOR_SIZE) if w2v is not None else np.zeros(WORD2VEC_VECTOR_SIZE)
    content_vec = avg_wordvec(content_tokens, w2v, dim=WORD2VEC_VECTOR_SIZE) if w2v is not None else np.zeros(WORD2VEC_VECTOR_SIZE)
    heur_vec, heur_dict = compute_heuristics(title, content, url, author)

    # If scaler exists, scale heuristics
    heur_for_concat = heur_vec
    if scaler is not None:
        try:
            heur_for_concat = scaler.transform(heur_vec)
        except Exception:
            logger.warning("Scaler transform failed in build_feature_vector; using raw heuristics")

    feature = np.concatenate([title_vec, content_vec, heur_for_concat.reshape(-1)], axis=0)
    expected_len = 2 * WORD2VEC_VECTOR_SIZE + N_HEURISTICS
    if feature.shape[0] != expected_len:
        logger.warning("Feature length mismatch (got %d, expected %d). Padding/truncating.", feature.shape[0], expected_len)
        if feature.shape[0] < expected_len:
            pad = np.zeros(expected_len - feature.shape[0], dtype=float)
            feature = np.concatenate([feature, pad], axis=0)
        else:
            feature = feature[:expected_len]
    return feature.reshape(1, -1), heur_dict

# ---------------------------
# Predict from feature vector (local model)
# ---------------------------
def predict_local_from_features(feature_vector, heur_dict):
    global best_model
    raw_label = None
    prob = None

    if best_model is None:
        logger.warning("No best_model loaded; using heuristics fallback only.")
        final_label = map_label_to_text(None, prob=None, heur_dict=heur_dict)
        return {"label": final_label, "prob": 0.0, "heuristics": heur_dict}

    try:
        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(feature_vector)
            # If predict_proba returned 1D (sometimes), coerce to 2d
            if isinstance(proba, (list, tuple, np.ndarray)) and np.array(proba).ndim == 1:
                proba = np.array(proba).reshape(1, -1)
            proba = np.array(proba)
            # choose class index (works for binary and multiclass)
            idx = int(np.argmax(proba, axis=1)[0])
            prob = float(np.max(proba, axis=1)[0])
            try:
                classes = list(getattr(best_model, "classes_", []))
                raw_label = classes[idx] if classes else idx
            except Exception:
                raw_label = idx
        else:
            raw_label = best_model.predict(feature_vector)[0]
            prob = None
    except Exception as e:
        logger.exception("Local model predict failed: %s", e)
        raw_label = None
        prob = None

    final_label = map_label_to_text(raw_label, prob=prob, heur_dict=heur_dict)
    if final_label is None:
        # safety net
        final_label = "Uncertain"
    if prob is None:
        prob = 0.0
    return {"label": final_label, "prob": float(prob), "heuristics": heur_dict}


# ---------------------------
# Gemini / generative model wrapper (robust)
# ---------------------------
def call_gemini(prompt, model_preference=None):
    """Try a few different call patterns for google.generativeai to handle versions."""
    if genai is None:
        return {"status": "disabled", "verdict": "undefined", "model": None, "raw": ""}

    # configure API key if present
    api_key = os.environ.get(GOOGLE_API_KEY_ENV)
    if api_key:
        try:
            genai.configure(api_key=api_key)
        except Exception:
            pass

    model_used = model_preference or "models/gemini-2.5-flash"
    try:
        if hasattr(genai, "generate_text"):
            r = genai.generate_text(model=model_used, text=prompt, max_output_tokens=300)
            content = getattr(r, "text", None) or str(r)
            import re
            m = re.search(r"```json(.*?)```", content, flags=re.S)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                    verdict = parsed.get("verdict")
                except Exception:
                    verdict = None
            else:
                verdict = None
            return {"status": "ok", "verdict": verdict, "model": model_used, "raw": content}
    # on exceptions, return verdict None
    except Exception as e:
        logger.warning("Gemini call failed: %s", e)
        return {"status": "error", "verdict": None, "model": model_used, "raw": str(e)}

# ---------------------------
# Persist flagged result to Firestore (if available)
# ---------------------------
def persist_flagged(payload):
    if db is None:
        logger.info("Firestore not initialized; skipping persist")
        return {"stored": "none"}
    try:
        col = db.collection("flagged_articles")
        doc_ref = col.document()
        doc_ref.set(payload)
        return {"stored": "firestore", "doc_id": doc_ref.id}
    except Exception:
        logger.exception("Failed to persist to Firestore")
        return {"stored": "error"}

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json() or {}
        title = data.get("title", "")
        content = data.get("content", "")
        url = data.get("url", "")
        author = data.get("author", "")

        # build features (ensures expected dims) and heuristics
        feature_vec, heur = build_feature_vector(title, content, url, author)

        # predict locally using feature vector + heuristics
        local = predict_local_from_features(feature_vec, heur)

        # prepare prompt for Gemini
        prompt = f"Title: {title}\nContent: {content}\nPlease classify this article as 'Real' or 'Fake' and provide brief justification in JSON: {{\"verdict\": <Real|Fake|Uncertain>, \"justification\": <text>}}"

        gemini_out = call_gemini(prompt)

        # compute mismatch logic: compare local label & gemini_out verdict if available
        gem_verdict = gemini_out.get("verdict") if gemini_out else None
        mismatch = False
        try:
            if gem_verdict and local.get("label") is not None:
                if str(local.get("label")).lower() != str(gem_verdict).lower():
                    mismatch = True
        except Exception:
            mismatch = False

        persist_info = {"stored": "none"}
        if mismatch:
            full_payload = {
                "title": title,
                "content": content,
                "url": url,
                "author": author,
                "local": to_native(local),
                "gemini": to_native(gemini_out),
                "heuristics": to_native(heur),
                "mismatch": True,
                "ts": datetime.utcnow().isoformat()
            }
            persist_info = persist_flagged(full_payload)

        resp = {
            "status": "ok",
            "input_title": title,
            "input_content": content,
            "input_url": url,
            "input_author": author,
            "heuristics": to_native(heur),
            "internal_model": {"label": to_native(local.get("label")), "prob": to_native(local.get("prob"))},
            "gemini_analysis": to_native(gemini_out),
            "mismatch_flagged": mismatch,
            "persist_info": to_native(persist_info),
            "final_verdict": to_native(local.get("label"))
        }
        return jsonify(to_native(resp))
    except Exception:
        logger.exception("Exception during /predict")
        return jsonify({"status": "error", "message": str(traceback.format_exc())}), 500

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    logger.info("Starting NewsTruth v3 backend")
    port = int(os.environ.get("PORT", 5500))
    app.run(host="0.0.0.0", port=port, debug=True)

