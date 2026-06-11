import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")
OPENAI_IMAGE_DETAIL = os.getenv("OPENAI_IMAGE_DETAIL", "high")

# Sampling temperature. The gpt-5 family ONLY accepts the default temperature (1)
# and rejects temperature=0, while gpt-4.x supports 0 for deterministic output.
# OPENAI_TEMPERATURE=None means "omit the parameter" (use the model default).
# Override via env: a number, or "none"/"default"/"" to omit it.
_temp_env = os.getenv("OPENAI_TEMPERATURE")
if _temp_env is not None:
    _temp_env = _temp_env.strip().lower()
    OPENAI_TEMPERATURE = None if _temp_env in {"none", "default", ""} else float(_temp_env)
elif "gpt-5" in OPENAI_MODEL.lower():
    OPENAI_TEMPERATURE = None
else:
    OPENAI_TEMPERATURE = 0


def max_output_tokens_kwargs(n):
    """Return the correct output-token-limit kwarg for the active model.

    gpt-5 renamed `max_tokens` -> `max_completion_tokens` and rejects the old name.
    Pass the result with ** into client.chat.completions.create(...). n=None omits it.
    """
    if n is None:
        return {}
    key = "max_completion_tokens" if "gpt-5" in OPENAI_MODEL.lower() else "max_tokens"
    return {key: n}

# ==============================
# PATTERN OVERRIDES
# ==============================
# Optional escape hatch: pin a specific PDF (by file name WITHOUT the .pdf
# extension) to a known pattern number. Intentionally EMPTY — every PDF is
# classified from its own content, so the pipeline generalises to other
# drawings of the same pattern that have different IDs / values / structure.
PATTERN_OVERRIDES = {}

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")
