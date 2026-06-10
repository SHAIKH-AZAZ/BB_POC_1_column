import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14")
OPENAI_IMAGE_DETAIL = os.getenv("OPENAI_IMAGE_DETAIL", "high")

# ==============================
# PATTERN OVERRIDES
# ==============================
# Pin a specific PDF (by file name WITHOUT the .pdf extension) to a known
# pattern number. When a PDF name matches a key here, auto_runner skips the
# (paid, non-deterministic) vision classifier and runs that pattern directly.
# Matching is case-insensitive. Leave empty to always auto-detect.
PATTERN_OVERRIDES = {
    "Column X-section": 15,
}

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")
