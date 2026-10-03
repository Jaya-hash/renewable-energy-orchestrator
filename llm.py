import hashlib
import json
import os
import re
import time
from dotenv import load_dotenv

load_dotenv()                     # reads your secret keys from the .env file

PROVIDER = "gemini"               # "anthropic", "openai" or "gemini"
MODELS = {                        # model names change over time: copy the exact name from your provider
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.8-flash",
}
MIN_GAP_SECONDS = {"gemini": 4.0}           # free Gemini keys only allow a few requests per minute
RETRY_CODES = (429, 500, 502, 503, 504)     # "too many requests" and "server busy" are worth retrying
RETRY_WORDS = ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded", "rate limit")
_last_call = [0.0]
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_cache.json")


def _call_once(system, user):
    """One attempt. Raises an error if anything fails."""
    model = MODELS[PROVIDER]

    if PROVIDER == "anthropic":
        from anthropic import Anthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY was not found in .env")
        reply = Anthropic(api_key=key).messages.create(
            model=model, max_tokens=1500, system=system,
            messages=[{"role": "user", "content": user}])
        return reply.content[0].text

    if PROVIDER == "openai":
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY was not found in .env")
        reply = OpenAI(api_key=key).chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return reply.choices[0].message.content

    if PROVIDER == "gemini":
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY was not found in .env")
        client = genai.Client(api_key=key)      # keep the client in a variable so it is not closed too early
        reply = client.models.generate_content(
            model=model, contents=f"{system}\n\n{user}")
        if not reply.text:
            raise RuntimeError("Gemini returned an empty answer")
        return reply.text

    raise ValueError(f"Unknown PROVIDER: {PROVIDER}")


def _should_retry(err):
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    if isinstance(code, int):
        return code in RETRY_CODES
    text = str(err)
    return any(word.lower() in text.lower() for word in RETRY_WORDS)


def _asked_wait(err):
    """The number of seconds the server asks us to wait, if it says so."""
    text = str(err)
    m = (re.search(r"retry in ([\d.]+)\s*s", text, re.I)
         or re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", text))
    return float(m.group(1)) if m else None


def _is_daily_limit(err):
    """A daily quota cannot be fixed by waiting a few seconds, so we should not keep retrying."""
    asked = _asked_wait(err)
    return "PerDay" in str(err) or (asked is not None and asked > 120)


def _wait_time(err, attempt):
    """Use the wait time the server asks for. Otherwise wait 5, 10, 20... seconds."""
    asked = _asked_wait(err)
    if asked is not None:
        return min(asked + 1, 60)
    return min(5 * 2 ** attempt, 60)


def _load_cache():
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1)
    except Exception:
        pass                                    # the cache is only a convenience


def call_llm(system, user, tries=4):
    """Ask the LLM. Answers are saved in llm_cache.json, so asking the same question again is free.
    Spaces out requests and retries 'too many requests' and 'server busy' errors."""
    cache_key = hashlib.sha256(f"{PROVIDER}|{MODELS[PROVIDER]}|{system}|{user}".encode()).hexdigest()
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]

    for attempt in range(tries):
        gap = MIN_GAP_SECONDS.get(PROVIDER, 0) - (time.time() - _last_call[0])
        if gap > 0:
            time.sleep(gap)
        try:
            _last_call[0] = time.time()
            text = _call_once(system, user)
            cache = _load_cache()
            cache[cache_key] = text
            _save_cache(cache)
            return text
        except Exception as err:
            print(f"[LLM] {type(err).__name__}: {str(err)[:1500]}")      # shows the real reason in the terminal
            if attempt == tries - 1 or not _should_retry(err) or _is_daily_limit(err):
                raise
            time.sleep(_wait_time(err, attempt))