# model_training.py
import os
import re
import json
import math
import pickle
import logging
import traceback
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# ---- Ensure Smart_open doesn't crash when boto3 missing ----
# If boto3 is not present, disable S3 transport in smart_open to avoid import errors.
try:
    import boto3  # type: ignore
    _HAS_BOTO3 = True
except Exception:
    _HAS_BOTO3 = False
    os.environ["SMART_OPEN_DISABLE_S3"] = "1"
    os.environ["SMART_OPEN_DISABLE_SES"] = "1"

# NLP libs (lazy downloads)
import ssl
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob, download_corpora as textblob_download_corpora

# gensim
from gensim.models import Word2Vec

# sklearn / imblearn / tensorflow (optional)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# imblearn (SMOTE) may be available
try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False

# TensorFlow is optional — only used for ANN if available
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    _HAS_TF = True
except Exception:
    _HAS_TF = False

# Matplotlib for plotting confusion matrices
import matplotlib
# Do not force a GUI backend; use Agg (file output)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Firebase optional imports (training may fetch flagged records)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    _HAS_FIREBASE = True
except Exception:
    firebase_admin = None
    firestore = None
    _HAS_FIREBASE = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("model_training")

# Configuration - adjust paths if needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "news_dataset_cleaned.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "test_reports", "reports"))
CONF_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "test_reports", "confusion_matrices"))
W2V_MODEL_PATH = os.path.join(MODELS_DIR, "word2vec.model")
SCALER_PATH = os.path.join(MODELS_DIR, "heuristic_scaler.pkl")
CLASSIFIER_PKL_PATH = os.path.join(MODELS_DIR, "best_model_combined.pkl")
CLASSIFIER_H5_PATH = os.path.join(MODELS_DIR, "best_model_combined.h5")

SERVICE_ACCOUNT_KEY = os.environ.get("FIREBASE_CRED_JSON", os.path.join(BASE_DIR, "serviceAccountKey.json"))
FIREBASE_COLLECTION = "flagged_articles"

# Heuristic feature names (order matters)
HEURISTIC_FEATURE_NAMES = [
    "domain_reputation",
    "is_satire",
    "author_credibility",
    "supporting_links",
    "is_old_news",
    "title_subjectivity",
    "content_subjectivity",
    "headline_body_sentiment_diff",
    "all_caps_words",
]

# Internal globals
_lemmatizer = None
_english_stopwords = None
_nltk_ensured = False

# --------------------- Utilities & preprocessing ---------------------


def ensure_nltk_textblob():
    """
    Ensure NLTK and TextBlob corpora exist. Use an SSL-safe fallback for macOS where needed.
    """
    global _nltk_ensured
    if _nltk_ensured:
        return

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        logger.info("Downloading NLTK stopwords...")
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        logger.info("Downloading NLTK wordnet...")
        nltk.download("wordnet")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        logger.info("Downloading NLTK punkt...")
        nltk.download("punkt")
    try:
        textblob_download_corpora()
    except Exception as e:
        logger.warning("textblob downloader failed: %s", e)
    _nltk_ensured = True


def get_lemmatizer_and_stopwords():
    global _lemmatizer, _english_stopwords
    if _lemmatizer is None or _english_stopwords is None:
        ensure_nltk_textblob()
        _lemmatizer = WordNetLemmatizer()
        try:
            _english_stopwords = set(stopwords.words("english"))
        except Exception:
            logger.info("Downloading stopwords (fallback)...")
            nltk.download("stopwords")
            _english_stopwords = set(stopwords.words("english"))
    return _lemmatizer, _english_stopwords


def preprocess_text(text: str) -> List[str]:
    lemmatizer, english_stopwords = get_lemmatizer_and_stopwords()
    if pd.isna(text) or text is None:
        return []
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    text = text.lower()
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in english_stopwords and len(word) > 1
    ]
    return tokens


def extract_heuristic_features(title: str, content: str, url: str, author: str) -> List[float]:
    # normalise NA values
    if pd.isna(url):
        url = ""
    if pd.isna(author):
        author = ""
    if pd.isna(title):
        title = ""
    if pd.isna(content):
        content = ""

    parsed = {"domain_reputation": 0, "is_satire": 0, "author_credibility": 0,
              "supporting_links": 0, "is_old_news": 0, "title_subjectivity": 0.0,
              "content_subjectivity": 0.0, "headline_body_sentiment_diff": 0.0,
              "all_caps_words": 0}

    try:
        # domain reputation heuristic
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        netloc = parsed_url.netloc or ""
        parsed["domain_reputation"] = 1 if netloc.endswith((".gov", ".edu", ".org")) else 0

        parsed["is_satire"] = 1 if any(s in url.lower() for s in ["theonion", "babylonbee"]) else 0
        parsed["author_credibility"] = 1 if author and len(author.split()) >= 2 else 0
        parsed["supporting_links"] = str(content).count("http")
        parsed["is_old_news"] = 1 if any(y in str(content) for y in ["2019", "2020", "2021"]) else 0

        title_blob = TextBlob(title or "")
        content_blob = TextBlob(content or "")
        parsed["title_subjectivity"] = title_blob.sentiment.subjectivity
        parsed["content_subjectivity"] = content_blob.sentiment.subjectivity
        parsed["headline_body_sentiment_diff"] = abs(title_blob.sentiment.polarity - content_blob.sentiment.polarity)
    except Exception:
        # keep defaults if TextBlob fails
        pass

    parsed["all_caps_words"] = len(re.findall(r"\b[A-Z]{3,}\b", title or ""))

    # Ensure the order matches HEURISTIC_FEATURE_NAMES
    return [float(parsed[name]) for name in HEURISTIC_FEATURE_NAMES]


def create_doc_vectors(tokens_list: List[List[str]], w2v_model: Word2Vec) -> np.ndarray:
    if w2v_model is None:
        raise ValueError("Word2Vec model is None.")
    vector_size = w2v_model.vector_size
    doc_vectors = []
    for doc_tokens in tokens_list:
        vec = np.zeros(vector_size, dtype=float)
        count = 0
        for w in doc_tokens:
            if w in w2v_model.wv:
                vec += w2v_model.wv[w]
                count += 1
        if count > 0:
            vec = vec / count
        doc_vectors.append(vec)
    return np.vstack(doc_vectors) if doc_vectors else np.zeros((0, vector_size))


# --------------------- Firebase helpers (optional) ---------------------
def initialize_firebase():
    if not _HAS_FIREBASE:
        logger.info("Firebase admin SDK not available; skipping Firebase initialization.")
        return None
    try:
        firebase_admin.get_app()
    except Exception:
        try:
            # 1. Try environment variables first
            project_id = os.environ.get("FIREBASE_PROJECT_ID")
            client_email = os.environ.get("FIREBASE_CLIENT_EMAIL")
            private_key = os.environ.get("FIREBASE_PRIVATE_KEY")
            
            if project_id and client_email and private_key:
                logger.info("Initializing Firebase SDK using environment variables")
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
            else:
                logger.info("Initializing Firebase SDK using credentials file: %s", SERVICE_ACCOUNT_KEY)
                cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
                firebase_admin.initialize_app(cred)
        except Exception as e:
            logger.error("Failed to init Firebase: %s", e)
            return None
    try:
        return firestore.client()
    except Exception as e:
        logger.error("Failed to get Firestore client: %s", e)
        return None



def fetch_flagged_data(db_client) -> pd.DataFrame:
    if db_client is None:
        return pd.DataFrame()
    try:
        docs = db_client.collection(FIREBASE_COLLECTION).stream()
        recs = []
        for doc in docs:
            d = doc.to_dict()
            if all(k in d for k in ["title", "content", "label", "url", "author"]):
                recs.append(d)
        return pd.DataFrame(recs)
    except Exception as e:
        logger.error("Error fetching flagged data: %s", e)
        return pd.DataFrame()


# --------------------- Data loading & merging ---------------------
def load_and_merge_data(db_client=None) -> pd.DataFrame:
    try:
        df_csv = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        logger.error("Data file not found at %s", DATA_FILE)
        return pd.DataFrame()

    required_cols = ["title", "content", "label", "url", "author"]
    if not all(col in df_csv.columns for col in required_cols):
        logger.error("CSV missing required columns. Found: %s", df_csv.columns.tolist())
        return pd.DataFrame()

    # Drop rows with missing required fields
    df_csv = df_csv.dropna(subset=required_cols).copy()

    # Fetch flagged data from firestore if available
    if db_client is not None:
        df_fb = fetch_flagged_data(db_client)
        if not df_fb.empty:
            df_fb = df_fb[required_cols]
            df_csv = pd.concat([df_csv[required_cols], df_fb], ignore_index=True)

    # Drop duplicates
    df_csv = df_csv.drop_duplicates(subset=["url", "title"], keep="last").reset_index(drop=True)

    logger.info("Preprocessing: computing tokens for title & content ...")
    df_csv["title_tokens"] = df_csv["title"].apply(preprocess_text)
    df_csv["content_tokens"] = df_csv["content"].apply(preprocess_text)

    logger.info("Total unique articles for training: %d", len(df_csv))
    return df_csv


# --------------------- Model training and helpers ---------------------
def ensure_dirs():
    for d in (MODELS_DIR, REPORTS_DIR, CONF_DIR):
        os.makedirs(d, exist_ok=True)


def save_classification_report(report: str, model_name: str):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path_txt = os.path.join(REPORTS_DIR, f"{model_name}_report_{ts}.txt")
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Saved classification report to %s", path_txt)


def plot_and_save_confusion_matrix(y_true, y_pred, model_name: str, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(cm, cmap="Blues")
    plt.colorbar(cax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    if labels is None:
        # try to infer unique labels
        labels = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    plt.tight_layout()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outp = os.path.join(CONF_DIR, f"{model_name}_confmat_{ts}.png")
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", outp)


def train_and_eval(name: str, model, X_tr, y_tr, X_te, y_te, label_map=None):
    logger.info("=== Training %s ===", name)
    try:
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        if getattr(preds, "ndim", 1) > 1:
            preds = (preds > 0.5).astype("int32").flatten()
        acc = float(accuracy_score(y_te, preds))
        cr = classification_report(y_te, preds, zero_division=0)
        logger.info("%s accuracy: %.4f", name, acc)
        logger.info("\n%s", cr)
        save_classification_report(cr, name)
        # Try to pretty-print labels in plots
        labels = None
        if label_map:
            inv_map = {v: k for k, v in label_map.items()}
            labels = [inv_map[i] for i in sorted(inv_map.keys())]
        plot_and_save_confusion_matrix(y_te, preds, name, labels=labels)
        return model, acc
    except Exception as e:
        logger.error("Failed training/eval for %s: %s\n%s", name, e, traceback.format_exc())
        return None, 0.0


def train_models():
    ensure_nltk_textblob()
    ensure_dirs()

    db = initialize_firebase() if _HAS_FIREBASE else None
    if db is None and _HAS_FIREBASE:
        logger.info("Firebase initialization returned None; continuing with CSV only.")

    df = load_and_merge_data(db)
    if df.empty:
        logger.error("No data for training. Exiting.")
        return

    # Map labels to binary ints if needed
    y_raw = df["label"].values
    if y_raw.dtype == object or isinstance(y_raw[0], str):
        # common string labels like 'Fake'/'Real'
        unique_labels = sorted(pd.Series(y_raw).unique().tolist())
        # prefer mapping Real->0 Fake->1 if present
        if "Real" in unique_labels and "Fake" in unique_labels:
            label_map = {"Real": 0, "Fake": 1}
        else:
            label_map = {lab: i for i, lab in enumerate(unique_labels)}
        y = np.array([label_map[x] for x in y_raw], dtype=int)
    else:
        # assume numeric already 0/1
        y = np.array(y_raw, dtype=int)
        label_map = None

    logger.info("Label map: %s", label_map)

    # Train Word2Vec on combined tokens
    all_tokens = list(df["title_tokens"]) + list(df["content_tokens"])
    # Guard against empty tokens
    if not any(all_tokens):
        logger.error("No tokens extracted from dataset. Check preprocessing.")
        return

    w2v_vector_size = 100
    logger.info("Training Word2Vec on %d documents (title+content)...", len(all_tokens))
    w2v_model = Word2Vec(sentences=all_tokens, vector_size=w2v_vector_size, window=5, min_count=2, workers=4, epochs=5)
    w2v_model.save(W2V_MODEL_PATH)
    logger.info("Saved Word2Vec model to %s", W2V_MODEL_PATH)

    # Create document vectors
    X_title = create_doc_vectors(df["title_tokens"].tolist(), w2v_model)
    X_content = create_doc_vectors(df["content_tokens"].tolist(), w2v_model)

    # Heuristic features
    heuristics = [extract_heuristic_features(r["title"], r["content"], r["url"], r["author"]) for _, r in df.iterrows()]
    X_heur = np.array(heuristics, dtype=float)

    # Scale heuristics
    scaler = StandardScaler()
    # handle any NaN or infinite values
    X_heur = np.nan_to_num(X_heur, nan=0.0, posinf=0.0, neginf=0.0)
    X_heur_scaled = scaler.fit_transform(X_heur)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved heuristic scaler to %s", SCALER_PATH)

    # Final features
    # ensure dims align
    try:
        X_final = np.hstack((X_title, X_content, X_heur_scaled))
    except Exception as e:
        logger.error("Failed to stack feature matrices: %s", e)
        # attempt to align by length
        min_rows = min(X_title.shape[0], X_content.shape[0], X_heur_scaled.shape[0])
        X_final = np.hstack((X_title[:min_rows], X_content[:min_rows], X_heur_scaled[:min_rows]))
        y = y[:min_rows]
        logger.info("Aligned to min_rows=%d", min_rows)

    logger.info("Final feature shape: %s", X_final.shape)

    # ---------------- Safe train/test split ----------------
    # Defensive logic to choose a safe test_size and whether to stratify.
    n_samples = X_final.shape[0]
    if n_samples != len(y):
        logger.warning("Feature/label mismatch: features=%d labels=%d. Aligning to min.", n_samples, len(y))
        n_min = min(n_samples, len(y))
        X_final = X_final[:n_min]
        y = y[:n_min]
        n_samples = n_min

    logger.info("Total samples after alignment: %d", n_samples)
    try:
        label_counts = pd.Series(y).value_counts()
        logger.info("Label distribution:\n%s", label_counts.to_string())
    except Exception:
        label_counts = None

    DEFAULT_TEST_SIZE = 0.2

    def safe_test_fraction(n, desired=DEFAULT_TEST_SIZE):
        if n < 5:
            return 1.0 / float(n)
        test_count = max(1, int(round(desired * n)))
        if n - test_count < 1:
            test_count = n - 1
        return test_count / float(n)

    test_frac = safe_test_fraction(n_samples)
    logger.info("Using test fraction: %.4f", test_frac)

    use_stratify = False
    if label_counts is not None:
        if label_counts.min() >= 2:
            use_stratify = True
        else:
            logger.warning("Not enough samples per class for stratify; falling back to non-stratified split.")

    try:
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=test_frac, random_state=42, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=test_frac, random_state=42, shuffle=True)
    except ValueError as e:
        logger.warning("train_test_split failed: %s; retrying with minimal test size", e)
        if n_samples >= 2:
            test_frac = 1.0 / float(n_samples)
            X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=test_frac, random_state=42, shuffle=True)
        else:
            raise RuntimeError("Not enough data to split into train & test.") from e

    logger.info("Train/test shapes: X_train=%s X_test=%s y_train=%d y_test=%d", X_train.shape, X_test.shape, len(y_train), len(y_test))

    # ---------------- Optional SMOTE ----------------
    if _HAS_SMOTE:
        try:
            sm = SMOTE(random_state=42)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            logger.info("Applied SMOTE. New train distribution: %s", pd.Series(y_train_res).value_counts().to_dict())
        except Exception as e:
            logger.warning("SMOTE failed: %s. Proceeding without SMOTE.", e)
            X_train_res, y_train_res = X_train, y_train
    else:
        logger.info("imblearn.SMOTE not available; continuing without over-sampling.")
        X_train_res, y_train_res = X_train, y_train

    # ---------------- Train classical models ----------------
    results = {}

    # Passive Aggressive
    pa = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    train_and_eval("PassiveAggressive", pa, X_train_res, y_train_res, X_test, y_test)
    results["PassiveAggressive"] = (pa, None)

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000, random_state=42)
    train_and_eval("LogisticRegression", lr, X_train_res, y_train_res, X_test, y_test)
    results["LogisticRegression"] = (lr, None)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    train_and_eval("RandomForest", rf, X_train_res, y_train_res, X_test, y_test)
    results["RandomForest"] = (rf, None)

    # GaussianNB
    gnb = GaussianNB()
    train_and_eval("GaussianNB", gnb, X_train_res, y_train_res, X_test, y_test)
    results["GaussianNB"] = (gnb, None)

    # ---------------- Optional ANN (TensorFlow) ----------------
    ann_acc = 0.0
    ann_model = None
    if _HAS_TF:
        try:
            input_dim = X_train_res.shape[1]
            ann = Sequential([
                Dense(128, activation="relu", input_shape=(input_dim,)),
                Dropout(0.5),
                Dense(64, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid")
            ])
            ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            logger.info("Training ANN (TensorFlow) for 20 epochs (may take time)...")
            ann.fit(X_train_res, y_train_res, epochs=20, batch_size=64, validation_split=0.1, verbose=1)
            probs = ann.predict(X_test)
            preds_ann = (probs > 0.5).astype("int32").flatten()
            ann_acc = float(accuracy_score(y_test, preds_ann))
            logger.info("ANN accuracy: %.4f", ann_acc)
            cr_ann = classification_report(y_test, preds_ann, zero_division=0)
            save_classification_report(cr_ann, "ANN")
            plot_and_save_confusion_matrix(y_test, preds_ann, "ANN")
            ann_model = ann
            results["ANN"] = (ann, ann_acc)
        except Exception as e:
            logger.warning("ANN training failed or skipped: %s", e)

    # ---------------- Choose best model ----------------
    # Evaluate classical models on X_test to compute accuracy scores
    eval_scores = {}
    for name, (m, _) in results.items():
        try:
            preds = m.predict(X_test)
            if getattr(preds, "ndim", 1) > 1:
                preds = (preds > 0.5).astype("int32").flatten()
            acc = float(accuracy_score(y_test, preds))
            eval_scores[name] = acc
        except Exception:
            eval_scores[name] = 0.0

    if ann_model is not None:
        eval_scores["ANN"] = ann_acc

    # pick best by accuracy
    if eval_scores:
        best_name = max(eval_scores, key=lambda k: eval_scores[k])
        best_score = eval_scores[best_name]
        logger.info("Best model: %s with accuracy %.4f", best_name, best_score)
    else:
        logger.error("No model evaluation scores found; nothing to save.")
        return

    # Save best model
    if best_name == "ANN" and ann_model is not None:
        ann_model.save(CLASSIFIER_H5_PATH)
        logger.info("Saved ANN model to %s", CLASSIFIER_H5_PATH)
    else:
        best_model_obj = results.get(best_name, (None,))[0]
        if best_model_obj is not None:
            with open(CLASSIFIER_PKL_PATH, "wb") as f:
                pickle.dump(best_model_obj, f)
            logger.info("Saved best sklearn model (%s) to %s", best_name, CLASSIFIER_PKL_PATH)

    logger.info("Training complete. Models & reports saved under %s", MODELS_DIR)


if __name__ == "__main__":
    try:
        train_models()
    except Exception as exc:
        logger.error("Fatal error during training: %s\n%s", exc, traceback.format_exc())
        raise
