# eval.py - Step 5: measure how often the AI's labels agree with a human (you).
#   python eval.py sample   -> writes data/label_me.csv (150 reviews, NO AI labels) + data/theme_guide.txt
#   python eval.py          -> compares your labels with the AI's and prints accuracy + mistakes
import sys
from datetime import datetime

import pandas as pd

import classify
import config

SAMPLE_SIZE = 150
EVAL_SEED = 7  # different seed from classify.py, so the sample isn't just "the first reviews classified"
EVAL_RESULTS_CSV = config.DATA_DIR / "eval_results.csv"
GUIDE_TXT = config.DATA_DIR / "theme_guide.txt"
SENTIMENT_ALIASES = {"neg": "negative", "neu": "neutral", "pos": "positive"}


def write_sample():
    # Never overwrite a file you may already have spent an hour labelling
    if config.LABEL_ME_CSV.exists() and "--force" not in sys.argv:
        sys.exit(f"{config.LABEL_ME_CSV} already exists. Use 'python eval.py sample --force' to overwrite.")

    df = pd.read_csv(config.REVIEWS_CSV)
    day = pd.to_datetime(df["at"]).dt.normalize()
    df = df[(day >= config.WINDOW_START) & (day <= config.WINDOW_END)]

    # Balanced: the same number of reviews from each app
    per_app = SAMPLE_SIZE // len(config.APPS)
    sample = pd.concat(df[df["app"] == app].sample(per_app, random_state=EVAL_SEED) for app in config.APPS)
    sample = sample.sample(frac=1, random_state=EVAL_SEED)  # mix apps so you can't guess a pattern

    out = sample[["reviewId", "app", "content"]].copy()
    out["my_theme"] = ""
    out["my_sentiment"] = ""
    # utf-8-sig lets Excel show Hindi text and emojis correctly
    out.to_csv(config.LABEL_ME_CSV, index=False, encoding="utf-8-sig")

    write_guide()
    print(f"Wrote {len(out)} reviews ({out['app'].value_counts().to_dict()}) to {config.LABEL_ME_CSV}")
    print(f"Labelling guide: {GUIDE_TXT}")


def write_guide():
    """A plain-text cheat sheet with the exact themes and rules the AI was given."""
    themes = classify.load_themes()
    lines = ["THEMES (copy the name exactly into my_theme)", ""]
    lines += [f"{t['name']}\n    {t['description']}" for t in themes]
    lines += ["", "TIE-BREAK RULES", classify.RULES, "",
              "SENTIMENT (my_sentiment): negative, neutral or positive - judged from the text, not the stars."]
    GUIDE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare():
    mine = pd.read_csv(config.LABEL_ME_CSV, encoding="utf-8-sig")
    mine = mine[mine["my_theme"].notna() & (mine["my_theme"].astype(str).str.strip() != "")]
    if mine.empty:
        sys.exit("No labels yet - fill in the my_theme column in data/label_me.csv first.")

    # Tidy your labels: ignore case/extra spaces, accept neg/neu/pos shortcuts
    names = {t["name"].lower(): t["name"] for t in classify.load_themes()}
    mine["my_theme"] = mine["my_theme"].astype(str).str.strip().str.lower().map(names).fillna("??? " + mine["my_theme"].astype(str))
    sent = mine["my_sentiment"].fillna("").astype(str).str.strip().str.lower()
    mine["my_sentiment"] = sent.map(lambda s: SENTIMENT_ALIASES.get(s[:3], s))

    typos = mine[mine["my_theme"].str.startswith("???")]
    if len(typos):
        print("WARNING - these labels don't match any theme name (check spelling):")
        print(typos[["reviewId", "my_theme"]].to_string(index=False), "\n")

    ai = pd.read_csv(config.CLASSIFIED_CSV)[["reviewId", "theme", "sentiment"]]
    both = mine.merge(ai, on="reviewId", how="inner")
    if len(both) < len(mine):
        print(f"Note: {len(mine) - len(both)} labelled reviews are not classified by the AI yet - skipped.\n")
    if both.empty:
        sys.exit("None of your labelled reviews have AI labels yet.")

    theme_ok = both["my_theme"] == both["theme"]
    has_sent = both["my_sentiment"] != ""
    sent_ok = both.loc[has_sent, "my_sentiment"] == both.loc[has_sent, "sentiment"]

    print(f"Reviews compared: {len(both)}")
    print(f"Theme accuracy:     {theme_ok.mean():.1%}  ({theme_ok.sum()}/{len(both)})")
    if has_sent.any():
        print(f"Sentiment accuracy: {sent_ok.mean():.1%}  ({sent_ok.sum()}/{has_sent.sum()})")

    print("\nConfusion table (rows = your label, columns = AI label):")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.crosstab(both["my_theme"], both["theme"], margins=True, margins_name="Total"))

    print("\nPer-theme accuracy (how often the AI agreed when YOU chose this theme):")
    per_theme = both.assign(ok=theme_ok).groupby("my_theme")["ok"].agg(n="size", accuracy="mean")
    print(per_theme.sort_values("accuracy").to_string(formatters={"accuracy": "{:.0%}".format}))

    mistakes = both[~theme_ok]
    print(f"\nMistakes ({len(mistakes)}):")
    for r in mistakes.itertuples():
        print(f"- [{r.app}] {str(r.content)[:140]}\n    you: {r.my_theme}  |  AI: {r.theme}")

    # Keep a history so the README can show accuracy before/after prompt changes
    row = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"), "n": len(both),
                         "theme_accuracy": round(theme_ok.mean(), 3),
                         "sentiment_accuracy": round(sent_ok.mean(), 3) if has_sent.any() else None}])
    row.to_csv(EVAL_RESULTS_CSV, mode="a", header=not EVAL_RESULTS_CSV.exists(), index=False)
    print(f"\nSaved this result to {EVAL_RESULTS_CSV}")


if __name__ == "__main__":
    write_sample() if len(sys.argv) > 1 and sys.argv[1] == "sample" else compare()
