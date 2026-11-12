# analysis.py - every number behind the dashboard. Plain pandas, no Streamlit, so it is easy to test.
import json
import math

import pandas as pd

import config

START = pd.Timestamp(config.WINDOW_START)
END = pd.Timestamp(config.DISPLAY_END)  # dashboard uses 27 Oct - 11 Nov; 12 Nov was only partly scraped
PERIOD = f"{START:%d %b} – {END:%d %b %Y}"
NOT_ACTIONABLE = {"Complaint: general", "Other"}  # real complaints, but nothing specific to fix
FESTIVAL = (pd.Timestamp("2026-11-06"), pd.Timestamp("2026-11-11"))  # Dhanteras to Bhai Dooj; Diwali = 8 Nov
INSTAMART_PATTERN = r"instamart|grocer(?:y|ies)"


def load():
    """Reviews in the display window, joined with their AI labels (classified reviews only)."""
    reviews = pd.read_csv(config.REVIEWS_CSV, parse_dates=["at"])
    day = reviews["at"].dt.normalize()
    reviews = reviews[(day >= START) & (day <= END)]

    names = [t["name"] for t in json.loads(config.THEMES_JSON.read_text())]
    if config.CLASSIFIED_CSV.exists():
        # on_bad_lines="skip": classify.py may be writing a line at this very moment
        labels = pd.read_csv(config.CLASSIFIED_CSV, usecols=["reviewId", "theme", "sentiment"], on_bad_lines="skip")
        labels = labels[labels["theme"].isin(names)].drop_duplicates("reviewId")
    else:
        labels = pd.DataFrame(columns=["reviewId", "theme", "sentiment"])

    df = reviews.merge(labels, on="reviewId")  # inner join = only reviews with AI labels
    return reviews, df, names, pd.read_csv(config.SCRAPE_SUMMARY_CSV)


def is_complaint(df, theme):
    return (df["theme"] == theme) & (df["sentiment"] == "negative")


# ---------------- Significance ----------------
def two_proportion_test(x1, n1, x2, n2):
    """Is x1/n1 really different from x2/n2, or could the gap be luck?
    Returns the gap and its 95% confidence interval (percentage points) and the two-sided p-value."""
    p1, p2 = x1 / n1, x2 / n2
    diff = p1 - p2
    pooled = (x1 + x2) / (n1 + n2)  # if both apps had one shared rate, it would be this
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    p_value = math.erfc(abs(diff) / se_pooled / math.sqrt(2)) if se_pooled > 0 else 1.0
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return diff * 100, (diff - 1.96 * se) * 100, (diff + 1.96 * se) * 100, p_value


def competitor_tests(df, complaint_themes, a, b):
    """Per complaint theme: % of each app's reviews complaining about it, gap, 95% CI, p-value."""
    n1, n2 = (df["app"] == a).sum(), (df["app"] == b).sum()
    rows = []
    for theme in complaint_themes:
        hit = is_complaint(df, theme)
        x1, x2 = (hit & (df["app"] == a)).sum(), (hit & (df["app"] == b)).sum()
        gap, low, high, p = two_proportion_test(x1, n1, x2, n2)
        rows.append({"theme": theme, a: x1 / n1 * 100, b: x2 / n2 * 100, "gap": gap, "ci_low": low,
                     "ci_high": high, "p": p})
    return pd.DataFrame(rows).set_index("theme")


def key_finding(df, complaint_themes):
    """The largest gap that survives a Bonferroni-corrected test (stricter, because we test many themes)."""
    a, b = list(config.APPS)[:2]
    tests = competitor_tests(df, complaint_themes, a, b)
    alpha = 0.05 / len(tests)
    tests["significant"] = tests["p"] < alpha
    result = {"tests": tests, "alpha": alpha, "a": a, "b": b, "theme": None,
              "n_sig": int(tests["significant"].sum())}
    sig = tests[tests["significant"]]
    if sig.empty:
        return result
    theme = sig["gap"].abs().idxmax()
    row = sig.loc[theme]
    worse, better = (a, b) if row["gap"] > 0 else (b, a)
    return result | {"theme": theme, "worse": worse, "better": better,
                     "hi": row[worse], "lo": row[better], "p": row["p"]}


def times_words(ratio):
    """1.92 -> 'nearly twice', 2.05 -> 'twice', 1.5 -> '1.5 times'."""
    words = {2: "twice", 3: "three times", 4: "four times"}
    n = round(ratio)
    if n in words and abs(ratio - n) <= 0.15:
        return ("nearly " if ratio < n - 0.03 else "") + words[n]
    return f"{ratio:.1f} times"


# ---------------- Robustness checks ----------------
def instamart_check(all_reviews, df, theme, other):
    """Does the gap survive if we drop every Swiggy review that mentions Instamart or groceries?"""
    window = all_reviews[all_reviews["app"] == "Swiggy"]
    sw, ot = df[df["app"] == "Swiggy"], df[df["app"] == other]
    mentions = sw["content"].str.contains(INSTAMART_PATTERN, case=False, regex=True)
    hit = is_complaint(sw, theme)
    x, n = (hit & ~mentions).sum(), (~mentions).sum()
    gap, low, high, p = two_proportion_test(x, n, is_complaint(ot, theme).sum(), len(ot))
    return {"window_reviews": len(window),
            "window_mentions": int(window["content"].str.contains(INSTAMART_PATTERN, case=False, regex=True).sum()),
            "hits": int(hit.sum()), "hits_mentioning": int((hit & mentions).sum()),
            "rate_without": x / n * 100, "other_rate": is_complaint(ot, theme).mean() * 100,
            "p": p, "survives": p < 0.05 and gap > 0}


def festival_check(df, theme):
    """Per app: complaint share on festival days (6-11 Nov) vs all other days."""
    fest = df["at"].dt.normalize().between(*FESTIVAL)
    hit = is_complaint(df, theme)
    out = {}
    for app in config.APPS:
        f, o = fest & (df["app"] == app), ~fest & (df["app"] == app)
        gap, low, high, p = two_proportion_test(hit[f].sum(), f.sum(), hit[o].sum(), o.sum())
        out[app] = {"festival": hit[f].mean() * 100, "other": hit[o].mean() * 100, "gap": gap,
                    "ci_low": low, "ci_high": high, "p": p, "spike": p < 0.05 and gap > 0}
    return out


# ---------------- Chart inputs ----------------
def daily_share(df, theme):
    """Per day and app: % of that day's classified reviews that complain about the theme."""
    d = df.assign(date=df["at"].dt.normalize(), hit=is_complaint(df, theme))
    g = d.groupby(["date", "app"]).agg(reviews=("hit", "size"), hits=("hit", "sum")).reset_index()
    g["share"] = g["hits"] / g["reviews"] * 100
    return g


def pain_points(app_df, complaint_themes):
    """Each complaint theme's share of this app's NEGATIVE reviews."""
    neg = app_df[app_df["sentiment"] == "negative"]
    t = (neg[neg["theme"].isin(complaint_themes)].groupby("theme")
         .agg(reviews=("reviewId", "size"), upvotes=("thumbsUpCount", "sum"), avg_rating=("score", "mean")))
    t["share"] = t["reviews"] / max(len(neg), 1)
    return t.sort_values("share", ascending=False).reset_index()
