# backend/app.py
import os
import json
import logging
import traceback
import threading
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

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("newstruth_app")

# Manually load environment variables from .env if present
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
root_env_path = os.path.join(BASE_DIR, "..", ".env")

for path in (env_path, root_env_path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip().strip('"').strip("'")
            logger.info("Manually loaded environment variables from %s", path)
        except Exception as e:
            pass

# ---------------------------
# CONFIG / paths
# ---------------------------
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

    # best model (sklearn)
    if os.path.exists(BEST_MODEL_PATH):
        try:
            logger.info("Loading best model from %s", BEST_MODEL_PATH)
            with open(BEST_MODEL_PATH, "rb") as fh:
                best_model = pickle.load(fh)
        except Exception:
            logger.exception("Failed to load best model")
            best_model = None
    else:
        logger.warning("Best model not found at %s", BEST_MODEL_PATH)
        best_model = None

load_models()

# ---------------------------
# Firestore init
# ---------------------------
db = None
def init_firestore():
    global db
    try:
        # 1. Try environment variables for Firebase credentials first
        project_id = os.environ.get("FIREBASE_PROJECT_ID")
        client_email = os.environ.get("FIREBASE_CLIENT_EMAIL")
        private_key = os.environ.get("FIREBASE_PRIVATE_KEY")
        
        if project_id and client_email and private_key:
            logger.info("Initializing Firestore using environment variables")
            # Replace escaped newlines if present
            formatted_key = private_key.replace("\\n", "\n")
            cred_dict = {
                "type": "service_account",
                "project_id": project_id,
                "private_key": formatted_key,
                "client_email": client_email,
                "token_uri": "https://oauth2.googleapis.com/token",
            }
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            return

        # 2. Fallback to credentials path env
        cred_path = os.environ.get(FIREBASE_CRED_ENV)
        if cred_path and os.path.exists(cred_path):
            logger.info("Initializing Firestore using credentials at %s", cred_path)
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            return

        # 3. Fallback to local file candidate
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
    """
    feature_vector: numpy array shaped (1, D)
    heur_dict: dict of heuristics
    returns {label, prob, heuristics}
    """
    global best_model
    raw_label = None
    prob = None

    if best_model is None:
        logger.warning("No best_model loaded; using heuristics fallback only.")
        final_label = map_label_to_text(None, prob=None, heur_dict=heur_dict)
        return {"label": final_label, "prob": None, "heuristics": heur_dict}

    try:
        # if classifier supports predict_proba
        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(feature_vector)[0]  # shape (n_classes,)
            # best_class index
            idx = int(np.argmax(proba))
            prob = float(np.max(proba))
            # try to get class label
            try:
                classes = list(best_model.classes_)
                raw_label = classes[idx]
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
    return {"label": final_label, "prob": (None if prob is None else float(prob)), "heuristics": heur_dict}

# ---------------------------
# Gemini / generative model wrapper (robust)
# ---------------------------
def call_gemini(prompt, model_preference=None):
    """Call Google Generative AI (Gemini API) using GenerativeModel."""
    if genai is None:
        return {"status": "disabled", "verdict": "Uncertain", "model": None, "raw": ""}

    api_key = os.environ.get(GOOGLE_API_KEY_ENV)
    if not api_key:
        return {
            "status": "error",
            "verdict": "Uncertain",
            "model": None,
            "raw": "GOOGLE_API_KEY environment variable not set."
        }

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        logger.warning("Failed to configure Gemini API: %s", e)
        return {"status": "error", "verdict": "Uncertain", "model": None, "raw": str(e)}

    # default to gemini-1.5-flash which is part of Gemini API free tier
    model_name = model_preference or "gemini-1.5-flash"
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        content = response.text
        
        # Clean up response to find JSON content
        clean_content = content.strip()
        if clean_content.startswith("```"):
            import re
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_content, flags=re.S)
            if m:
                clean_content = m.group(1).strip()
        
        try:
            parsed = json.loads(clean_content)
            verdict = parsed.get("verdict", "Uncertain")
            justification = parsed.get("justification", "")
        except Exception:
            # Simple regex fallback if JSON parsing fails
            import re
            verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', clean_content, re.I)
            verdict = verdict_match.group(1) if verdict_match else "Uncertain"
            justification_match = re.search(r'"justification"\s*:\s*"([^"]+)"', clean_content, re.I)
            justification = justification_match.group(1) if justification_match else ""

        # Normalize verdict
        v_lower = str(verdict).lower()
        if "real" in v_lower or "true" in v_lower or v_lower == "1":
            normalized_verdict = "Real"
        elif "fake" in v_lower or "false" in v_lower or v_lower == "0":
            normalized_verdict = "Fake"
        else:
            normalized_verdict = "Uncertain"

        return {
            "status": "ok",
            "verdict": normalized_verdict,
            "justification": justification or content,
            "model": model_name,
            "raw": content
        }
    except Exception as e:
        logger.warning("Gemini generate_content failed: %s", e)
        return {"status": "error", "verdict": "Uncertain", "model": model_name, "raw": str(e)}


# ---------------------------
# Persist flagged result to Firestore (if available)
# ---------------------------
import json
import os
from datetime import datetime

LOCAL_FLAGGED_PATH = os.path.join(BASE_DIR, "models", "flagged_local.jsonl")

def persist_flagged(payload):
    """
    Persist flagged payload to Firestore if available.
    If Firestore isn't initialized or write fails, append to a local JSONL file
    so the UI gets a successful response and you can inspect/ingest later.
    Returns a dict describing where it was stored.
    """
    # first try Firestore if db is set
    if db is not None:
        try:
            col = db.collection("flagged_articles")
            doc_ref = col.document()
            doc_ref.set(payload)
            return {"stored": "firestore", "doc_id": doc_ref.id}
        except Exception as e:
            logger.exception("Failed to persist to Firestore: %s", e)
            # fallthrough to local file

    # fallback: write to local file (append JSON lines)
    try:
        os.makedirs(os.path.dirname(LOCAL_FLAGGED_PATH), exist_ok=True)
        with open(LOCAL_FLAGGED_PATH, "a", encoding="utf-8") as fh:
            # add timestamp if missing
            payload_copy = dict(payload)
            if "ts" not in payload_copy:
                payload_copy["ts"] = datetime.utcnow().isoformat()
            fh.write(json.dumps(payload_copy, default=str, ensure_ascii=False) + "\n")
        logger.info("Persisted flagged payload to local file: %s", LOCAL_FLAGGED_PATH)
        return {"stored": "local_file", "path": LOCAL_FLAGGED_PATH}
    except Exception as e:
        logger.exception("Failed to persist flagged payload to local file: %s", e)
        return {"stored": "error", "error": str(e)}


# ---------------------------
# Background Retraining and Dataset Helpers
# ---------------------------
training_lock = threading.Lock()
is_training = False

def run_background_retraining():
    global is_training
    with training_lock:
        if is_training:
            logger.warning("Retraining already in progress. Skipping.")
            return
        is_training = True
    
    try:
        logger.info("Background retraining started...")
        from model_training import train_models
        train_models()
        logger.info("Background retraining complete. Reloading models...")
        load_models()
    except Exception as e:
        logger.exception("Error in background retraining thread: %s", e)
    finally:
        with training_lock:
            is_training = False
        logger.info("Background retraining thread finalized.")

def append_to_dataset(title, content, label_str, url, author):
    csv_path = os.path.join(BASE_DIR, "data", "news_dataset_cleaned.csv")
    
    # Map 'Real' -> 0, 'Fake' -> 1
    label_num = 0 if label_str == "Real" else 1
    
    # Clean up fields
    title = title.replace("\n", " ").replace("\r", "").strip()
    content = content.replace("\n", " ").replace("\r", "").strip()
    url = url.strip()
    author = author.strip()
    
    import csv
    row = [title, content, label_num, url, author]
    
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["title", "content", "label", "url", "author"])
            writer.writerow(row)
        logger.info("Successfully appended flagged article to CSV dataset: %s", title)
        return True
    except Exception as e:
        logger.exception("Failed to append to CSV dataset: %s", e)
        return False

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

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
        prompt = (
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"URL: {url}\n"
            f"Author: {author}\n\n"
            f"Please verify this article claim and categorize it as Real or Fake. "
            f"Respond strictly in the following JSON format:\n"
            f'{{\n  "verdict": "Real" | "Fake" | "Uncertain",\n  "justification": "<explanation>"\n}}'
        )

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

        # Standardize prediction properties for the UI
        model_pred = local.get("label")
        model_prob = local.get("prob")
        # Format confidence as percentage
        if model_prob is not None:
            model_conf = round(float(model_prob) * 100, 1)
        else:
            model_conf = 0.0

        gem_verdict = gemini_out.get("verdict", "Uncertain")
        gem_justification = gemini_out.get("justification", "")
        gem_source = gemini_out.get("model", "None")

        resp = {
            "status": "ok",
            "input_title": title,
            "input_content": content,
            "input_url": url,
            "input_author": author,
            "heuristics": to_native(heur),
            "internal_model": {"label": to_native(model_pred), "prob": to_native(model_prob)},
            "model_prediction": to_native(model_pred),
            "model_confidence": model_conf,
            "gemini_analysis": to_native(gemini_out),
            "gemini_prediction": to_native(gem_verdict),
            "gemini_justification": to_native(gem_justification),
            "gemini_source": to_native(gem_source),
            "mismatch_flagged": mismatch,
            "persist_info": to_native(persist_info),
            "final_verdict": to_native(model_pred)
        }
        return jsonify(to_native(resp))
    except Exception:
        logger.exception("Exception during /predict")
        return jsonify({"status": "error", "message": str(traceback.format_exc())}), 500

@app.route("/flag", methods=["POST"])
def flag_route():
    try:
        data = request.get_json() or {}
        title = data.get("title", "")
        content = data.get("content", "")
        url = data.get("url", "")
        author = data.get("author", "")
        user_label = data.get("label", "Fake") # "Real" or "Fake"
        
        if not title:
            return jsonify({"status": "error", "message": "Headline/Title is required."}), 400

        # Ask Gemini to verify the article
        prompt = (
            f"Please cross-verify this news article and verify if it is Real or Fake.\n"
            f"User correction label: {user_label}\n"
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"URL: {url}\n"
            f"Author: {author}\n\n"
            f"Respond strictly in JSON format:\n"
            f'{{\n  "verdict": "Real" | "Fake" | "Uncertain",\n  "justification": "<explanation>"\n}}'
        )

        gemini_out = call_gemini(prompt)
        gem_verdict = gemini_out.get("verdict", "Uncertain")
        gem_justification = gemini_out.get("justification", "")
        
        # Decide verified label
        verified_label = gem_verdict if gem_verdict in ("Real", "Fake") else user_label
        
        # Append to CSV
        success = append_to_dataset(title, content, verified_label, url, author)
        if not success:
            return jsonify({"status": "error", "message": "Failed to save article to CSV dataset."}), 500
            
        # Trigger background retraining
        global is_training
        retraining_triggered = False
        with training_lock:
            if not is_training:
                t = threading.Thread(target=run_background_retraining)
                t.daemon = True
                t.start()
                retraining_triggered = True
                
        # Also persist flagged entry to Firestore (or local JSONL fallback)
        persist_payload = {
            "title": title,
            "content": content,
            "url": url,
            "author": author,
            "user_label": user_label,
            "verified_label": verified_label,
            "gemini_verdict": gem_verdict,
            "gemini_justification": gem_justification,
            "retrained": retraining_triggered,
            "ts": datetime.utcnow().isoformat()
        }
        persist_info = persist_flagged(persist_payload)
        
        msg = f"News article cross-verified with Gemini API (Verdict: {gem_verdict}). "
        if retraining_triggered:
            msg += "Successfully added to dataset and background retraining triggered."
        else:
            msg += "Successfully added to dataset. Retraining is already running in background."
            
        return jsonify({
            "status": "ok",
            "message": msg,
            "verified_label": verified_label,
            "gemini_verdict": gem_verdict,
            "gemini_justification": gem_justification,
            "retraining_triggered": retraining_triggered,
            "persist_info": persist_info
        })
    except Exception as e:
        logger.exception("Error in /flag route")
        return jsonify({"status": "error", "message": str(traceback.format_exc())}), 500

@app.route("/training-status", methods=["GET"])
def training_status_route():
    global is_training
    return jsonify({
        "status": "ok",
        "is_training": is_training
    })

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    logger.info("Starting NewsTruth v3 backend")
    port = int(os.environ.get("PORT", 5500))
    app.run(host="0.0.0.0", port=port, debug=True)
