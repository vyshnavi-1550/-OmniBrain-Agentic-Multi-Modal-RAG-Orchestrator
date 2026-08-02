"""
OmniBrain configuration.
All heavy/paid dependencies (OpenAI GPT-4o) are OPTIONAL. If OPENAI_API_KEY is not
set, the system automatically falls back to lightweight local alternatives
(TF-IDF embeddings instead of OpenAI embeddings, keyword-based SQL templating
instead of LLM-based NL-to-SQL, and a plain-text image caption stub instead of
a real vision-language call) so you can develop and test the *architecture*
end-to-end without spending money on API calls. Swap in real credentials any
time -- the interfaces don't change.
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
SQL_DB_PATH = DATA_DIR / "stock_data.db"

for d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
USE_OPENAI = bool(OPENAI_API_KEY)

# Chunking
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between chunks

# Retrieval
TOP_K = 5
