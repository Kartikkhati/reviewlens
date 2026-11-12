# config.py - every setting the project shares lives here, in one place.
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # copies LLM_* values from the .env file into environment variables

# --- LLM settings (read from .env locally, from Secrets on Streamlit Cloud) ---
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL") or "openai/gpt-oss-120b"  # default = model the data was labelled with
REASONING_EFFORT = "low"  # gpt-oss is a reasoning model; "low" = less thinking, fewer tokens

# --- Groq free-tier limits for openai/gpt-oss-120b ---
TOKENS_PER_MINUTE = 8000
TOKENS_PER_DAY = 200_000  # refills gradually (~139 tokens/min), not all at midnight
TOKEN_RESERVE = 10_000    # classify.py leaves this much of the daily limit unused (headroom for test calls)
EST_TOKENS_PER_BATCH = 3000  # a classify batch uses ~2,200-3,100 tokens

RANDOM_SEED = 42  # same seed = same "random" sample every run (reproducible)

# --- Step 2: theme discovery ---
THEME_SAMPLE_NEG = 70  # 1-2 star reviews sent to the LLM (split evenly across apps)
THEME_SAMPLE_POS = 30  # 4-5 star reviews
THEME_MAX_CHARS = 200  # cut each review to this length to keep the request small

# --- Step 3: classification ---
BATCH_SIZE = 25
CLASSIFY_MAX_CHARS = 300
# Only classify the date window BOTH apps cover, so the comparison is fair
WINDOW_START = "2026-10-27"
WINDOW_END = "2026-11-12"  # inclusive
DISPLAY_END = "2026-11-11"  # dashboard stops here: 12 Nov was only partly scraped

# --- Apps to compare: display name -> Google Play package ID ---
APPS = {
    "Swiggy": "in.swiggy.android",
    "Zomato": "com.application.zomato",
}

# --- Scraping settings ---
REVIEWS_PER_APP = 10000  # newest reviews to fetch per app (~60% get dropped as too short)
PAGE_SIZE = 200         # reviews per request (Google Play's maximum per page)
MIN_WORDS = 4           # shorter reviews ("good", "nice app") carry no useful detail
LANG, COUNTRY = "en", "in"

# --- File paths (Path makes them work on Mac, Windows and Linux) ---
DATA_DIR = Path(__file__).parent / "data"
REVIEWS_CSV = DATA_DIR / "reviews.csv"
SCRAPE_SUMMARY_CSV = DATA_DIR / "scrape_summary.csv"
THEMES_JSON = DATA_DIR / "themes.json"
CLASSIFIED_CSV = DATA_DIR / "classified.csv"
LABEL_ME_CSV = DATA_DIR / "label_me.csv"
TOKEN_LOG_CSV = DATA_DIR / "token_log.csv"  # one row per LLM call, to track daily usage
