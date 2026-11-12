# classify.py - Step 3: label every review with one theme + a sentiment.
# Output: data/classified.csv, appended after every batch so the script can resume.
import json
import re
import sys
import time
from datetime import datetime

import pandas as pd

import config
import llm

SENTIMENT = {"neg": "negative", "neu": "neutral", "pos": "positive"}

# Kept separate so eval.py can copy the SAME rules into the human labelling guide
RULES = """- Several problems: pick the one the user is most upset about; if unclear, the one mentioned first.
- User asking where their money is -> "Refund issues", even if they also complain about support.
- Money deducted but order failed, or charged twice -> "Payment problems".
- Food arrived cold BECAUSE it was late -> "Late delivery".
- Rider being slow -> "Late delivery". Rider rude, asking for money, or faking delivery -> "Delivery partner behaviour".
- Vague negative review with no specific reason ("worst app", "very bad service") -> "Complaint: general".
- "Other" is ONLY for reviews unrelated to the app experience or impossible to judge (e.g. "don't know yet").
- Judge sentiment from the text, not the star rating."""

PROMPT = """Classify Google Play reviews of Indian food-delivery apps.
Reviews may be in English, Hindi or Hinglish.

Themes (answer with the number):
{themes}

For EVERY review choose exactly one theme (its main point) and a sentiment: "neg", "neu" or "pos".

Tie-break rules:
""" + RULES + """

Reply with a JSON array only, one object per review, no extra text:
[{{"i":0,"t":3,"s":"neg"}},{{"i":1,"t":0,"s":"pos"}}]

Reviews:
{reviews}"""


def load_themes():
    themes = json.loads(config.THEMES_JSON.read_text())
    if "Other" not in [t["name"] for t in themes]:
        themes.append({"name": "Other", "description": "Anything that does not fit the themes above."})
    return themes


def reviews_to_do():
    """Reviews inside the date window, in a fixed random order, minus ones already classified."""
    df = pd.read_csv(config.REVIEWS_CSV)
    day = pd.to_datetime(df["at"]).dt.normalize()  # drop the time, keep the date
    df = df[(day >= config.WINDOW_START) & (day <= config.WINDOW_END)]
    print(f"Reviews in window {config.WINDOW_START} to {config.WINDOW_END}: "
          + ", ".join(f"{app} {n}" for app, n in df["app"].value_counts().items()))
    df = df.sample(frac=1, random_state=config.RANDOM_SEED)  # shuffle: partial results = random sample

    # Reviews you are hand-labelling go first, so eval.py can run before the full job finishes
    if config.LABEL_ME_CSV.exists():
        is_eval = df["reviewId"].isin(pd.read_csv(config.LABEL_ME_CSV, encoding="utf-8-sig")["reviewId"])
        df = pd.concat([df[is_eval], df[~is_eval]])

    if config.CLASSIFIED_CSV.exists():
        done = set(pd.read_csv(config.CLASSIFIED_CSV)["reviewId"])
        df = df[~df["reviewId"].isin(done)]
    return df


def parse_labels(results, names):
    """Turn [{"i":0,"t":3,"s":"neg"}, ...] into {0: ("Theme name", "negative")}. Bad values are fixed."""
    labels, fixed = {}, 0
    for r in results if isinstance(results, list) else []:
        try:
            i, t = int(r["i"]), int(r.get("t", -1))
        except (KeyError, TypeError, ValueError):
            continue  # unusable row: this review stays unclassified and is retried next run
        if not 0 <= t < len(names):
            t, fixed = names.index("Other"), fixed + 1  # unknown theme -> "Other"
        sentiment = SENTIMENT.get(str(r.get("s", "")).lower()[:3], "neutral")
        labels[i] = (names[t], sentiment)
    return labels, fixed


def wait_for_budget():
    """Pause until a batch fits without eating into the token reserve kept for teardowns."""
    need = config.TOKEN_RESERVE + config.EST_TOKENS_PER_BATCH
    while (left := llm.estimated_remaining()) < need:
        minutes = (need - left) / llm.REFILL_PER_MIN
        print(f"  ~{max(left, 0):,} daily tokens left, keeping {config.TOKEN_RESERVE:,} in reserve. "
              f"Waiting ~{minutes:.0f} min for Groq to refill...")
        time.sleep(min(minutes, 15) * 60 + 5)  # re-check at least every 15 minutes


def main():
    themes = load_themes()
    names = [t["name"] for t in themes]
    theme_list = "\n".join(f"{n}. {t['name']}: {t['description']}" for n, t in enumerate(themes))

    todo = reviews_to_do()
    if "--limit" in sys.argv:  # test run, e.g. "python classify.py --limit 50"
        todo = todo.head(int(sys.argv[sys.argv.index("--limit") + 1]))
    total_batches = -(-len(todo) // config.BATCH_SIZE)  # ceiling division
    print(f"{len(todo)} reviews to classify in {total_batches} batches of {config.BATCH_SIZE}.")

    for b, start in enumerate(range(0, len(todo), config.BATCH_SIZE), 1):
        batch = todo.iloc[start : start + config.BATCH_SIZE].reset_index(drop=True)
        texts = batch["content"].str.replace(r"\s+", " ", regex=True).str[: config.CLASSIFY_MAX_CHARS]
        reviews = "\n".join(f"{i}. {t}" for i, t in enumerate(texts))

        wait_for_budget()
        try:
            reply, usage = llm.chat(PROMPT.format(themes=theme_list, reviews=reviews),
                                    max_tokens=3000, label="classify")
            labels, fixed = parse_labels(llm.extract_json(reply), names)
        except llm.DailyLimitReached as e:
            retry = re.search(r"try again in ([\w.]+)", str(e))
            sys.exit(f"\nDaily token limit reached. Progress is saved in {config.CLASSIFIED_CSV}.\n"
                     f"Run 'python classify.py' again later"
                     f"{' (Groq says: try again in ' + retry.group(1) + ')' if retry else ''} to continue.")
        except (ValueError, RuntimeError) as e:  # bad JSON or repeated network errors
            print(f"Batch {b}: skipped ({e}). These reviews will be retried on the next run.")
            continue

        # Save this batch straight away (append), so a crash never loses finished work
        out = batch.loc[list(labels)].copy()
        out["theme"] = [labels[i][0] for i in out.index]
        out["sentiment"] = [labels[i][1] for i in out.index]
        out.to_csv(config.CLASSIFIED_CSV, mode="a", header=not config.CLASSIFIED_CSV.exists(), index=False)

        print(f"{datetime.now():%H:%M} Batch {b}/{total_batches}: {len(out)}/{len(batch)} labelled"
              f"{f', {fixed} unknown theme(s) -> Other' if fixed else ''} | "
              f"tokens {usage.total_tokens} (in {usage.prompt_tokens}, out {usage.completion_tokens}) | "
              f"est. daily left ~{llm.estimated_remaining():,}")

        # Pace ourselves: a batch of N tokens "uses up" N/8000 of a minute (+10% safety margin)
        time.sleep(max(2.0, usage.total_tokens / config.TOKENS_PER_MINUTE * 60 * 1.1))

    print(f"\nDone. Results in {config.CLASSIFIED_CSV}")


if __name__ == "__main__":
    main()
