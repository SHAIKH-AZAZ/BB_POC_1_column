import os

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ==============================================================================
# OPENAI MODEL CONFIG
# ==============================================================================
# Two model families are supported and they need DIFFERENT request parameters.
# Everything that depends on which family is active is decided here, once, and
# exposed through a single helper `openai_request_kwargs()` so the rest of the
# codebase never has to special-case the model.
#
#   gpt-4.x  (e.g. gpt-4.1-mini)  — classic model
#   gpt-5.x  (e.g. gpt-5-mini)    — REASONING model
#
# Three concrete differences the gpt-5 family imposes:
#   1. temperature : gpt-5 accepts ONLY the default (1); gpt-4.x allows 0 (deterministic).
#   2. token param : gpt-5 calls it `max_completion_tokens`; gpt-4.x calls it `max_tokens`.
#   3. reasoning   : gpt-5 spends hidden "reasoning" tokens out of the SAME budget, so a
#                    small limit (e.g. 5 tokens for a YES/NO) is fully consumed by reasoning
#                    and the visible answer comes back EMPTY. For budget-limited calls we
#                    therefore raise the ceiling (GPT5_TOKEN_FLOOR) and ask for minimal
#                    reasoning so short label / yes-no reads stay correct, cheap and fast.
# ==============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")
OPENAI_IMAGE_DETAIL = os.getenv("OPENAI_IMAGE_DETAIL", "high")

# Is the active model a gpt-5-family reasoning model? Decided once, used everywhere.
IS_REASONING_MODEL = "gpt-5" in OPENAI_MODEL.lower()

# --- temperature -------------------------------------------------------------
# None  -> omit the parameter entirely (use the model default).
# Env override OPENAI_TEMPERATURE: a number, or "none"/"default"/"" to omit.
_temp_env = os.getenv("OPENAI_TEMPERATURE")
if _temp_env is not None:
    _temp_env = _temp_env.strip().lower()
    OPENAI_TEMPERATURE = None if _temp_env in {"none", "default", ""} else float(_temp_env)
elif IS_REASONING_MODEL:
    OPENAI_TEMPERATURE = None      # gpt-5: must use the default
else:
    OPENAI_TEMPERATURE = 0         # gpt-4.x: deterministic

# --- reasoning-model tuning (gpt-5 only) -------------------------------------
GPT5_TOKEN_FLOOR = int(os.getenv("OPENAI_GPT5_TOKEN_FLOOR", "2048"))
GPT5_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "minimal").strip().lower()


def openai_request_kwargs(max_output_tokens=None):
    """Build the model-specific kwargs to spread into chat.completions.create().

    One place handles every gpt-5-vs-gpt-4.x difference, so call sites read simply:

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=...,
            **openai_request_kwargs(max_output_tokens=200),   # or () for no limit
        )

    `max_output_tokens` = how long the ANSWER may be (e.g. 5 for a YES/NO); None = no
    explicit limit. Reasoning effort is only constrained when a limit is set — the
    unlimited tool-protocol calls keep the model's full reasoning.
    """
    kwargs = {}

    # 1) temperature (omitted when None)
    if OPENAI_TEMPERATURE is not None:
        kwargs["temperature"] = OPENAI_TEMPERATURE

    if max_output_tokens is not None:
        if IS_REASONING_MODEL:
            # 2) gpt-5 token param + reasoning headroom so the answer is never empty
            kwargs["max_completion_tokens"] = max(max_output_tokens, GPT5_TOKEN_FLOOR)
            # 3) keep these short, budget-limited reads cheap by minimising reasoning
            if GPT5_REASONING_EFFORT not in {"", "none", "default"}:
                kwargs["reasoning_effort"] = GPT5_REASONING_EFFORT
        else:
            kwargs["max_tokens"] = max_output_tokens

    return kwargs

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
