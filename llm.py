# llm.py - small helpers for talking to the LLM: one call function, retries, JSON parsing.
import csv
import json
import re
import time
from datetime import datetime, timedelta

import openai
from openai import OpenAI

import config

# max_retries=0: we handle retries ourselves so we can treat the daily limit differently
client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, max_retries=0)

DAILY_STATE_JSON = config.DATA_DIR / "groq_daily.json"  # what Groq told us the last time we hit the limit
REFILL_PER_MIN = config.TOKENS_PER_DAY / (24 * 60)      # ~139 tokens come back every minute


class DailyLimitReached(Exception):
    """Raised when the provider says the daily quota is used up. Retrying won't help."""


def chat(prompt, max_tokens=4000, retries=4, label="call"):
    """Send one prompt and return (reply_text, usage). Retries short-term failures."""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,  # same input -> (nearly) same output; good for classification
                extra_body={"reasoning_effort": config.REASONING_EFFORT},
            )
            log_tokens(label, response.usage)
            return response.choices[0].message.content or "", response.usage

        except openai.RateLimitError as e:  # HTTP 429
            if "per day" in str(e).lower():  # e.g. "tokens per day (TPD)" -> stop, don't retry
                save_daily_state(str(e))
                raise DailyLimitReached(str(e)) from e
            wait = 20 * (attempt + 1)  # per-minute limit: wait and try again
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
            wait = 5 * (attempt + 1)  # network blip or server error: short wait

        if attempt == retries:
            raise RuntimeError(f"LLM call failed after {retries} retries")
        print(f"  ...temporary error, retrying in {wait}s")
        time.sleep(wait)


def extract_json(text):
    """Pull the first JSON object/array out of a reply, even if wrapped in ```json fences."""
    text = re.sub(r"```(?:json)?", "", text)
    start = min((i for i in (text.find("["), text.find("{")) if i != -1), default=-1)
    if start == -1:
        raise ValueError("No JSON found in LLM reply")
    data, _ = json.JSONDecoder().raw_decode(text[start:])  # ignores any text after the JSON
    return data


def log_tokens(label, usage):
    """Append one row per call to data/token_log.csv."""
    new_file = not config.TOKEN_LOG_CSV.exists()
    with open(config.TOKEN_LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "label", "prompt_tokens", "completion_tokens", "total_tokens"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), label,
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens])


def tokens_used_since(since):
    """Total tokens logged after a given datetime."""
    if not config.TOKEN_LOG_CSV.exists():
        return 0
    with open(config.TOKEN_LOG_CSV) as f:
        return sum(int(r["total_tokens"]) for r in csv.DictReader(f)
                   if datetime.fromisoformat(r["time"]) > since)


def save_daily_state(message):
    """Groq's 429 says e.g. 'Limit 200000, Used 199100' - remember how much was left, and when."""
    m = re.search(r"Limit (\d+), Used (\d+)", message)
    remaining = max(0, int(m[1]) - int(m[2])) if m else 0
    DAILY_STATE_JSON.write_text(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                                            "remaining": remaining}))


def estimated_remaining():
    """Estimate of Groq's daily tokens left = (left at last limit hit) + refill since - used since.
    Slightly pessimistic: Groq doesn't count some cached prompt tokens, but we count all of them."""
    if not DAILY_STATE_JSON.exists():  # never hit the limit: assume the last 24h of usage still counts
        return config.TOKENS_PER_DAY - tokens_used_since(datetime.now() - timedelta(days=1))
    state = json.loads(DAILY_STATE_JSON.read_text())
    since = datetime.fromisoformat(state["time"])
    refill = (datetime.now() - since).total_seconds() / 60 * REFILL_PER_MIN
    return int(min(config.TOKENS_PER_DAY, state["remaining"] + refill - tokens_used_since(since)))
