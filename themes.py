# themes.py - Step 2: ask the LLM to propose ONE shared list of review themes.
# Output: data/themes.json  (you can edit this file by hand afterwards)
import json
import sys

import pandas as pd

import config
import llm


def take_sample(df):
    """Balanced sample: same number of negative/positive reviews from each app."""
    parts = []
    n_apps = len(config.APPS)
    for app in config.APPS:
        a = df[df["app"] == app]
        parts.append(a[a["score"] <= 2].sample(config.THEME_SAMPLE_NEG // n_apps, random_state=config.RANDOM_SEED))
        parts.append(a[a["score"] >= 4].sample(config.THEME_SAMPLE_POS // n_apps, random_state=config.RANDOM_SEED))
    sample = pd.concat(parts).sample(frac=1, random_state=config.RANDOM_SEED)  # mix the order
    return sample.reset_index(drop=True)


PROMPT = """You are a product analyst studying Google Play reviews of Indian food-delivery apps.
Reviews may be in English, Hindi or Hinglish.

Propose 8-12 themes that together cover these reviews. Rules:
- Specific and actionable (e.g. "Late delivery", "Refund delays"), not vague ("Bad service").
- Include complaint themes AND 1-3 praise themes (e.g. "Fast delivery praise").
- Themes must apply to any food-delivery app; don't name a specific app.
- Do NOT include "Other" - it is added automatically.
- For each theme give "examples": up to 2 review numbers from the list that fit it best.

Reply with JSON only:
{{"themes": [{{"name": "...", "description": "one sentence on what belongs here", "examples": [3, 17]}}]}}

Reviews:
{reviews}"""


def main():
    # Protect hand edits: don't overwrite an existing themes.json unless asked
    if config.THEMES_JSON.exists() and "--force" not in sys.argv:
        sys.exit(f"{config.THEMES_JSON} already exists. Run 'python themes.py --force' to overwrite it.")

    df = pd.read_csv(config.REVIEWS_CSV)
    sample = take_sample(df)
    texts = sample["content"].str.replace(r"\s+", " ", regex=True).str[: config.THEME_MAX_CHARS]
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    print(f"Sending {len(sample)} reviews ({(sample.score <= 2).sum()} negative, {(sample.score >= 4).sum()} positive)...")

    reply, usage = llm.chat(PROMPT.format(reviews=numbered), label="themes")
    themes = llm.extract_json(reply)["themes"]

    # Save only name + description; add the catch-all "Other" theme ourselves
    clean = [{"name": t["name"].strip(), "description": t["description"].strip()} for t in themes]
    clean.append({"name": "Other", "description": "Anything that does not fit the themes above."})
    config.THEMES_JSON.write_text(json.dumps(clean, indent=2, ensure_ascii=False))

    # Show each theme with up to 2 real reviews the model picked as examples
    print(f"\n{len(clean)} themes saved to {config.THEMES_JSON}\n")
    for n, t in enumerate(themes, 1):
        print(f"{n}. {t['name']} - {t['description']}")
        for i in t.get("examples", [])[:2]:
            if isinstance(i, int) and 0 <= i < len(sample):
                r = sample.iloc[i]
                print(f"     [{r['app']}, {r['score']}★] {r['content'][:150]}")
    print(f"{len(clean)}. Other - Anything that does not fit the themes above.")

    print(f"\nTokens used: prompt {usage.prompt_tokens}, completion {usage.completion_tokens}, total {usage.total_tokens}")


if __name__ == "__main__":
    main()
