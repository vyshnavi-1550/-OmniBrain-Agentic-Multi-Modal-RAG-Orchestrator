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
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

# Load variables from a .env file at the project root (if present)
load_dotenv(ROOT_DIR / ".env")

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

# Langfuse (observability / tracing) — optional; tracing is skipped gracefully if unset
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
USE_LANGFUSE = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# Gemini (vision-language model) — optional; used for real chart/table
# extraction on ingested images when OPENAI_API_KEY isn't set.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
USE_GEMINI = bool(GOOGLE_API_KEY)