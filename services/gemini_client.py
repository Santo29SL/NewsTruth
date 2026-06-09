# services/gemini_client.py
import os
import google.generativeai as genai
import logging

logger = logging.getLogger("gemini_client")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        logger.info("Gemini client configured.")
    except Exception as e:
        logger.warning(f"Failed to configure Gemini client: {e}")
        GEMINI_KEY = ""

def gemini_verdict_single_word(title, url, author, snippet, heur_dict, model_name="gemini-1.5-mini"):
    """
    Uses your Gemini key (if provided) to request EXACTLY one token: Real|Fake|Uncertain.
    Returns token string or None if call fails/missing key.
    """
    prompt = f"""You are an expert fact-checker. Given the inputs below, RETURN EXACTLY ONE WORD (Real OR Fake OR Uncertain) and NOTHING ELSE.

TITLE: "{title}"
URL: "{url}"
AUTHOR: "{author}"
SNIPPET: "{snippet}"

HEURISTICS:
- domain_reputation: {heur_dict.get('domain_reputation')}
- is_satire: {heur_dict.get('is_satire')}
- author_credibility: {heur_dict.get('author_credibility')}
- supporting_links: {heur_dict.get('supporting_links')}
- is_old_news: {heur_dict.get('is_old_news')}
- title_subjectivity: {heur_dict.get('title_subjectivity')}
- content_subjectivity: {heur_dict.get('content_subjectivity')}
- headline_body_sentiment_diff: {heur_dict.get('headline_body_sentiment_diff')}
- all_caps_words: {heur_dict.get('all_caps_words')}
- cross_ref_count: {heur_dict.get('cross_ref_count')}
- major_source_match: {heur_dict.get('major_source_match')}

Return ONE of: Real OR Fake OR Uncertain
"""

    if not GEMINI_KEY:
        return None

    try:
        resp = genai.generate_text(model=model_name, input=prompt, max_output_tokens=3, temperature=0.0)
        text = (resp.text or "").strip()
        token = text.split()[0] if text else ""
        if token in {"Real", "Fake", "Uncertain"}:
            return token
        t_up = token.capitalize()
        if t_up in {"Real", "Fake", "Uncertain"}:
            return t_up
        return None
    except Exception as e:
        logger.error(f"Gemini call error: {e}")
        return None
