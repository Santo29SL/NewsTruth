# backend/list_models.py
import os
import sys
import traceback

print("Python:", sys.version.splitlines()[0])

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY env var not set.")
    print("Run this before executing the script:")
    print('  export GOOGLE_API_KEY="your_gemini_api_key_here"')
    sys.exit(1)

try:
    import google.generativeai as genai
except Exception as e:
    print("❌ Could not import google.generativeai:", e)
    sys.exit(1)

# Configure Gemini
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print("❌ Failed to configure Gemini:", e)
    sys.exit(1)

# Try listing models
try:
    models = genai.list_models()
    print("\n✅ SUCCESS! Gemini API connected.\nAvailable models:")
    for m in models:
        # handle different response object shapes
        name = getattr(m, "name", str(m))
        desc = getattr(m, "display_name", "")
        print(f" - {name} {f'({desc})' if desc else ''}")
except Exception as e:
    print("❌ Error listing models:")
    traceback.print_exc()
    print("\n⚠️ If you see a permissions or 404 error, you may not have access to some Gemini models yet.")
