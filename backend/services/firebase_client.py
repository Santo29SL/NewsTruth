# services/firebase_client.py
import os
import logging
import json
from datetime import datetime

logger = logging.getLogger("firebase_client")

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except Exception:
    firebase_admin = None

FIREBASE_CRED_PATH = os.environ.get("FIREBASE_CRED_JSON", "serviceAccountKey.json")
FIREBASE_ENABLED = False
db = None

LOCAL_FLAG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "flagged_local.json")

def initialize_firestone():
    global FIREBASE_ENABLED, db
    if firebase_admin is None:
        logger.warning("firebase_admin not installed; Firestore disabled.")
        FIREBASE_ENABLED = False
        return
    try:
        if os.path.exists(FIREBASE_CRED_PATH):
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            FIREBASE_ENABLED = True
            logger.info("Firestore initialized (service account).")
        else:
            firebase_admin.initialize_app()
            db = firestore.client()
            FIREBASE_ENABLED = True
            logger.info("Firestore initialized (application default).")
    except Exception as e:
        logger.warning(f"Could not initialize Firestore: {e}")
        FIREBASE_ENABLED = False

def flag_record(doc: dict):
    doc_with_ts = dict(doc)
    doc_with_ts["flagged_at"] = datetime.utcnow().isoformat()
    try:
        if FIREBASE_ENABLED and db:
            coll = db.collection("flagged_articles")
            coll.add(doc_with_ts)
            logger.info("Flagged record saved to Firestore.")
            return True
        os.makedirs(os.path.dirname(LOCAL_FLAG_PATH), exist_ok=True)
        if os.path.exists(LOCAL_FLAG_PATH):
            with open(LOCAL_FLAG_PATH, "r", encoding="utf-8") as f:
                arr = json.load(f)
        else:
            arr = []
        arr.append(doc_with_ts)
        with open(LOCAL_FLAG_PATH, "w", encoding="utf-8") as f:
            json.dump(arr, f, indent=2, ensure_ascii=False)
        logger.info("Flagged record saved locally.")
        return True
    except Exception as e:
        logger.error(f"Failed to save flagged record: {e}")
        return False

def fetch_flagged_records(limit=500):
    try:
        if FIREBASE_ENABLED and db:
            docs = db.collection("flagged_articles").limit(limit).stream()
            return [d.to_dict() for d in docs]
        if os.path.exists(LOCAL_FLAG_PATH):
            with open(LOCAL_FLAG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Failed to fetch flagged records: {e}")
        return []
