# app.py - Step 4: the ReviewLens dashboard.   Run with:  streamlit run app.py
import json
import math
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # colour-blind-safe palette, always in this order
BAR_COLOUR = SERIES[0]
NOT_ACTIONABLE = {"Complaint: general", "Other"}  # real complaints, but nothing specific to fix
MIN_FOR_TEARDOWN = 500
START, END = pd.Timestamp(config.WINDOW_START), pd.Timestamp(config.WINDOW_END)
DISPLAY_END = pd.Timestamp(config.DISPLAY_END)  # daily charts only - partial last day hidden
WINDOW_LABEL = f"{START:%d %b} – {END:%d %b %Y}"


# ---------------- 1. Load data (re-read from disk at most every 5 minutes) ----------------
@st.cache_data(ttl=300)
def load_data():
    reviews = pd.read_csv(config.REVIEWS_CSV, parse_dates=["at"])
    day = reviews["at"].dt.normalize()
    reviews = reviews[(day >= START) & (day <= END)]  # comparison window only

    themes = json.loads(config.THEMES_JSON.read_text())
    names = [t["name"] for t in themes]

    if config.CLASSIFIED_CSV.exists():
        # on_bad_lines="skip": classify.py may be writing a line at this very moment
        labels = pd.read_csv(config.CLASSIFIED_CSV, usecols=["reviewId", "theme", "sentiment"],
                             on_bad_lines="skip")
        labels = labels[labels["theme"].isin(names)].drop_duplicates("reviewId")
    else:
        labels = pd.DataFrame(columns=["reviewId", "theme", "sentiment"])

    df = reviews.merge(labels, on="reviewId")  # inner join = only reviews that have AI labels
    summary = pd.read_csv(config.SCRAPE_SUMMARY_CSV)
    return reviews, df, names, summary


# ---------------- 2. Calculations (plain pandas, no Streamlit) ----------------
def pain_points(app_df, complaint_themes):
    """Each complaint theme's share of this app's NEGATIVE reviews."""
    neg = app_df[app_df["sentiment"] == "negative"]
    t = (neg[neg["theme"].isin(complaint_themes)]
         .groupby("theme")
         .agg(reviews=("reviewId", "size"), upvotes=("thumbsUpCount", "sum"), avg_rating=("score", "mean")))
    t["share"] = t["reviews"] / max(len(neg), 1)
    return t.sort_values("share", ascending=False).reset_index()


def complaint_rates(df, complaint_themes):
    """% of each app's classified reviews that are a negative review about each theme."""
    neg = df[df["sentiment"] == "negative"]
    counts = pd.crosstab(neg["theme"], neg["app"])
    totals = df.groupby("app").size()
    rates = counts.div(totals, axis=1) * 100  # percentages, so apps of different size compare fairly
    return rates.reindex(index=complaint_themes, columns=list(config.APPS)).fillna(0)


def two_proportion_test(x1, n1, x2, n2):
    """Is x1/n1 really different from x2/n2, or could the gap be luck?
    Returns the difference and its 95% confidence interval (in percentage points) and the p-value."""
    p1, p2 = x1 / n1, x2 / n2
    diff = p1 - p2
    # z-test: assume both apps share one true rate ("pooled"), and ask how surprising our gap would be
    pooled = (x1 + x2) / (n1 + n2)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    p_value = math.erfc(abs(diff) / se_pooled / math.sqrt(2)) if se_pooled > 0 else 1.0  # two-sided
    # 95% confidence interval: the range the true gap probably lies in
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return diff * 100, (diff - 1.96 * se) * 100, (diff + 1.96 * se) * 100, p_value


def competitor_tests(df, complaint_themes, app, other):
    """For each complaint theme: % of each app's reviews complaining about it, gap, 95% CI, p-value."""
    n1, n2 = (df["app"] == app).sum(), (df["app"] == other).sum()
    neg = df[df["sentiment"] == "negative"]
    rows = []
    for theme in complaint_themes:
        x1 = ((neg["app"] == app) & (neg["theme"] == theme)).sum()
        x2 = ((neg["app"] == other) & (neg["theme"] == theme)).sum()
        gap, low, high, p = two_proportion_test(x1, n1, x2, n2)
        rows.append({"theme": theme, app: x1 / n1 * 100, other: x2 / n2 * 100,
                     "gap": gap, "ci": f"{low:+.1f} to {high:+.1f}", "p": p})
    return pd.DataFrame(rows).set_index("theme").sort_values(app, ascending=False)


INSTAMART_PATTERN = r"instamart|grocer(?:y|ies)"


def instamart_check(all_reviews, df):
    """How many Swiggy reviews are about Instamart/grocery, and how much of 'Late delivery' they explain."""
    window = all_reviews[all_reviews["app"] == "Swiggy"]
    sw = df[df["app"] == "Swiggy"]
    mentions = sw["content"].str.contains(INSTAMART_PATTERN, case=False, regex=True)
    late = (sw["theme"] == "Late delivery") & (sw["sentiment"] == "negative")
    return {
        "window_reviews": len(window),
        "window_mentions": int(window["content"].str.contains(INSTAMART_PATTERN, case=False, regex=True).sum()),
        "classified": len(sw),
        "classified_mentions": int(mentions.sum()),
        "late_total": int(late.sum()),
        "late_mentions": int((late & mentions).sum()),
        # Swiggy's late-delivery rate if we ignore every Instamart/grocery review
        "late_rate_without": (late & ~mentions).sum() / max((~mentions).sum(), 1) * 100,
    }


def key_finding(df, complaint_themes):
    """One plain sentence: the largest gap that survives a Bonferroni-corrected significance test."""
    a, b = list(config.APPS)[:2]
    tests = competitor_tests(df, complaint_themes, a, b)
    alpha = 0.05 / len(tests)  # Bonferroni: stricter threshold because we test many themes at once
    sig = tests[tests["p"] < alpha]
    if sig.empty:
        return (f"No complaint theme differs significantly between {a} and {b} yet "
                f"(Bonferroni-corrected threshold p < {alpha:.4f}, {len(tests)} themes tested).")
    theme = sig["gap"].abs().idxmax()
    row = sig.loc[theme]
    worse, better = (a, b) if row["gap"] > 0 else (b, a)
    hi, lo = max(row[a], row[b]), min(row[a], row[b])
    times = f"{hi / lo:.1f}× as often as" if lo > 0 else "more often than"
    p = "p < 0.001" if row["p"] < 0.001 else f"p = {row['p']:.3f}"
    return (f"**{worse}** users complain about **{theme.lower()}** {times} {better} users: "
            f"{hi:.1f}% vs {lo:.1f}% of reviews, a gap of {abs(row['gap']):.1f} percentage points "
            f"({p}; significant even after Bonferroni correction across {len(tests)} themes).")


def daily_trend(app_df, themes):
    """Daily number of negative reviews for the given themes, with 0 for days with none."""
    neg = app_df[(app_df["sentiment"] == "negative") & app_df["theme"].isin(themes)]
    counts = neg.groupby([neg["at"].dt.normalize().rename("date"), "theme"]).size()
    grid = pd.MultiIndex.from_product([pd.date_range(START, DISPLAY_END), themes], names=["date", "theme"])
    return counts.reindex(grid, fill_value=0).reset_index(name="reviews")


def teardown_prompt(app, app_df, pain, rates):
    """Build the LLM prompt from computed numbers + a few real reviews. Nothing else is sent."""
    others = [a for a in config.APPS if a != app]
    neg = app_df[app_df["sentiment"] == "negative"]
    top5 = [t for t in pain["theme"] if t not in NOT_ACTIONABLE][:5]

    themes_data = []
    for rank, theme in enumerate(top5, 1):
        row = pain.set_index("theme").loc[theme]
        examples = neg[neg["theme"] == theme].nlargest(4, "thumbsUpCount")
        themes_data.append({
            "rank": rank,
            "theme": theme,
            "negative_reviews": int(row["reviews"]),
            "share_of_negative_reviews_pct": round(row["share"] * 100, 1),
            "pct_of_all_reviews": round(rates.loc[theme, app], 1),
            "competitor_pct_of_all_reviews": {o: round(rates.loc[theme, o], 1) for o in others},
            "total_upvotes": int(row["upvotes"]),
            "avg_star_rating": round(row["avg_rating"], 2),
            "most_upvoted_reviews": [
                {"upvotes": int(r.thumbsUpCount), "stars": int(r.score), "text": str(r.content)[:300]}
                for r in examples.itertuples()
            ],
        })

    data = {
        "app": app,
        "competitors": others,
        "period": WINDOW_LABEL,
        "reviews_analysed": len(app_df),
        "pct_negative": round((app_df["sentiment"] == "negative").mean() * 100, 1),
        "data_caveat": "Reviews under 4 words were removed; most of those were 4-5 stars, so ratings and "
                       "% negative are lower than the store's. Use the data to rank problems, not to "
                       "measure overall satisfaction.",
        "top_pain_themes": themes_data,
    }

    return f"""You are a senior product manager writing a product teardown of the {app} app
from Google Play reviews, for a startup leadership team.

STRICT RULES
- Use ONLY the numbers and reviews in DATA. Do not invent statistics, user counts or company facts.
- When you cite a number, copy it exactly from DATA. Quote reviews only from DATA (short excerpts).
- Root causes are hypotheses: label them as such.
- Effort is S, M or L with a one-line reason.

FORMAT (markdown)
## Executive summary
3-4 sentences: the biggest problems, how {app} compares with competitors, what to fix first.
## Pain points
For each theme in DATA, in rank order:
### <rank>. <theme>
- **Problem:** ...
- **Evidence:** numbers from DATA, incl. competitor comparison, plus 1-2 short quotes
- **Root-cause hypothesis:** ...
- **Proposed fix:** concrete product or operations change
- **Effort:** S/M/L - reason
## Top 3 fixes (ranked)
Numbered list; rank by impact (share of complaints, upvotes, gap vs competitor) against effort.

DATA
{json.dumps(data, indent=1, ensure_ascii=False)}"""


# ---------------- 3. Page ----------------
all_reviews, df, theme_names, summary = load_data()
complaint_themes = [t for t in theme_names if not t.startswith("Praise")]

st.title("🔍 ReviewLens")
st.caption("AI product teardowns from Play Store reviews")

total, done = len(all_reviews), len(df)
if done < total:
    st.info(f"Based on **{done:,} of {total:,}** reviews classified so far (random sample). "
            "Numbers update automatically as classification continues.")

with st.container(border=True):
    st.markdown("**🔑 Key finding**")
    st.markdown(key_finding(df, complaint_themes))

# Sidebar
app = st.sidebar.selectbox("App", list(config.APPS))
others = [a for a in config.APPS if a != app]
st.sidebar.caption(f"Comparison window: {WINDOW_LABEL}")
if st.sidebar.button("Refresh data now"):
    st.cache_data.clear()
    st.rerun()

mine = df[df["app"] == app]
if mine.empty:
    st.warning(f"No classified {app} reviews yet. Run classify.py first.")
    st.stop()

# --- Metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews classified", f"{len(mine):,}", help=f"of {len(all_reviews[all_reviews.app == app]):,} in the window")
c2.metric("Average rating", f"{mine['score'].mean():.2f} ★",
          help="Of reviews with 4+ words. Lower than the store rating by design - see 'About the data'.")
c3.metric("% negative", f"{(mine['sentiment'] == 'negative').mean():.0%}", help="Judged by the AI from the text")
c4.metric("Date range", f"{mine['at'].min():%d %b} – {mine['at'].max():%d %b}")

# --- Top pain points ---
st.header(f"Top pain points: {app}")
pain = pain_points(mine, complaint_themes)
left, right = st.columns([3, 2])
with left:
    bars = alt.Chart(pain).mark_bar(color=BAR_COLOUR, cornerRadiusEnd=4).encode(
        x=alt.X("share:Q", title="Share of negative reviews", axis=alt.Axis(format=".0%")),
        y=alt.Y("theme:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("theme:N", title="Theme"), alt.Tooltip("share:Q", title="Share", format=".1%"),
                 alt.Tooltip("reviews:Q", title="Reviews"), alt.Tooltip("upvotes:Q", title="Upvotes")],
    ).properties(height=max(260, 28 * len(pain)))
    st.altair_chart(bars, width="stretch")
with right:
    st.dataframe(
        pain[["theme", "reviews", "share", "upvotes", "avg_rating"]],
        hide_index=True, width="stretch",
        column_config={
            "theme": "Theme",
            "reviews": st.column_config.NumberColumn("Reviews"),
            "share": st.column_config.NumberColumn("Share", format="percent"),
            "upvotes": st.column_config.NumberColumn("Upvotes", help="Total 'helpful' votes on these reviews"),
            "avg_rating": st.column_config.NumberColumn("Avg ★", format="%.2f"),
        },
    )

# --- Competitor comparison ---
st.header("Competitor comparison")
other = others[0]  # the test compares two apps
st.caption(f"% of each app's reviews that complain about each theme, and the gap ({app} minus {other}) "
           "in percentage points (pp).")
rates = complaint_rates(df, complaint_themes)  # still used by the teardown prompt
comp = competitor_tests(df, complaint_themes, app, other)
comp["verdict"] = [
    ("▲ worse" if g > 0 else "▼ better") + " (significant)" if p < 0.05 else "not significant"
    for g, p in zip(comp["gap"], comp["p"])
]


def style_row(row):
    if row["p"] >= 0.05:
        return ["color: #8a8984"] * len(row)  # grey = can't tell the apps apart
    colour = "rgba(227, 73, 72, 0.22)" if row["gap"] > 0 else "rgba(27, 175, 122, 0.20)"
    return [f"background-color: {colour}; font-weight: 600"] * len(row)


styled = (comp.style
          .apply(style_row, axis=1)
          .format({app: "{:.1f}%", other: "{:.1f}%", "gap": "{:+.1f} pp",
                   "p": lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"}))
st.dataframe(styled, width="stretch", column_config={
    "_index": "Theme", "gap": "Gap", "ci": "95% CI of gap (pp)", "p": "p-value", "verdict": "Verdict"})
st.caption(
    "**How to read this:** a gap is *significant* when p < 0.05, i.e. a difference this large would happen "
    "less than 5% of the time by chance if both apps really had the same complaint rate, and then the 95% "
    "confidence interval does not include 0. Grey rows could be noise. With ~13 themes tested, expect about "
    "one 'significant' result by luck alone, so trust large, low-p gaps most.")

if "Swiggy" in config.APPS:
    im = instamart_check(all_reviews, df)
    st.caption(
        f"**Instamart note:** {im['window_mentions']:,} of {im['window_reviews']:,} Swiggy reviews in the window "
        f"mention Instamart or grocery. Among classified ones, {im['late_mentions']} of Swiggy's "
        f"{im['late_total']} late-delivery complaints mention them. The Swiggy app includes Instamart "
        f"(quick-commerce groceries) while Zomato's equivalent (Blinkit) is a separate app, so this comparison is "
        f"not perfectly like-for-like. Without those reviews, Swiggy's late-delivery rate is "
        f"{im['late_rate_without']:.1f}%.")

# --- Trend ---
top4 = [t for t in pain["theme"] if t not in NOT_ACTIONABLE][:4]
st.header("Daily trend of the top 4 pain points")
trend = daily_trend(mine, top4)
lines = alt.Chart(trend).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
    x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
    y=alt.Y("reviews:Q", title="Negative reviews per day"),
    color=alt.Color("theme:N", scale=alt.Scale(domain=top4, range=SERIES[: len(top4)]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=[alt.Tooltip("date:T", title="Date", format="%a %d %b"), alt.Tooltip("theme:N", title="Theme"),
             alt.Tooltip("reviews:Q", title="Reviews")],
).properties(height=320)
st.altair_chart(lines, width="stretch")
st.caption("Counts come from the classified sample, so they rise as classification continues - "
           "compare the shape of the lines, not their height. 'Complaint: general' and 'Other' are left "
           "out because they have no specific fix.")

# --- Teardown ---
st.header("AI teardown")
path = config.DATA_DIR / f"teardown_{app.lower()}.md"
enough = len(mine) >= MIN_FOR_TEARDOWN
prompt = teardown_prompt(app, mine, pain, rates)

with st.expander("What gets sent to the LLM"):
    st.caption(f"~{len(prompt) // 4:,} tokens. Only computed numbers and the 4 most-upvoted reviews "
               "for each of the top 5 pain themes.")
    st.code(prompt, language="markdown")

clicked = st.button("Regenerate teardown" if path.exists() else "Generate teardown", type="primary",
                    disabled=not (enough and config.ALLOW_TEARDOWN),
                    help=None if enough else f"Needs {MIN_FOR_TEARDOWN} classified {app} reviews (now {len(mine)}).")
if not config.ALLOW_TEARDOWN:
    st.caption("Generation is turned off in this public demo; the saved report below was generated by the author.")
elif not enough:
    st.caption(f"Available once {MIN_FOR_TEARDOWN} {app} reviews are classified (currently {len(mine)}).")

if clicked:
    import llm  # imported here so the dashboard still opens if no API key is set

    with st.spinner("Writing the teardown (about 30-60 seconds)..."):
        try:
            report, usage = llm.chat(prompt, max_tokens=4000, label="teardown")
        except llm.DailyLimitReached:
            st.error("Groq's daily token limit is used up. Try again later.")
        except Exception as e:
            st.error(f"LLM call failed: {e}")
        else:
            header = (f"_Generated {datetime.now():%d %b %Y %H:%M} by {config.LLM_MODEL} from "
                      f"{len(mine):,} classified {app} reviews ({WINDOW_LABEL})._\n\n")
            path.write_text(header + report, encoding="utf-8")
            st.success(f"Saved to {path.name} ({usage.total_tokens:,} tokens used)")

if path.exists():
    st.markdown(path.read_text(encoding="utf-8"))
    st.download_button("Download teardown (.md)", path.read_text(encoding="utf-8"), file_name=path.name)

# --- About the data ---
st.header("About the data")
st.write("Most short reviews we removed (under 4 words, like “good app”) were 4–5 stars, so the "
         "reviews left behind skew negative: **ReviewLens ranks problems; it does not measure overall "
         "customer satisfaction.**")
st.dataframe(
    summary[["app", "fetched", "avg_rating_all", "kept", "avg_rating_kept", "short_dropped", "pct_short_4_5_star"]],
    hide_index=True, width="stretch",
    column_config={
        "app": "App", "fetched": "Reviews fetched", "avg_rating_all": "Avg ★ (all)",
        "kept": "Kept (4+ words)", "avg_rating_kept": "Avg ★ (kept)", "short_dropped": "Short reviews dropped",
        "pct_short_4_5_star": st.column_config.NumberColumn("% of dropped that were 4–5★", format="%.1f%%"),
    },
)
st.caption(f"Newest 10,000 reviews per app from Google Play (India, English store). Analysis window "
           f"{WINDOW_LABEL}, the period both apps cover. Themes and sentiment by {config.LLM_MODEL}.")

# --- Review explorer ---
st.header("Review explorer")
theme = st.selectbox("Theme", theme_names, index=theme_names.index(top4[0]) if top4 else 0)
picked = mine[mine["theme"] == theme].sort_values("thumbsUpCount", ascending=False)
st.caption(f"{len(picked):,} {app} reviews, most upvoted first")
st.dataframe(
    picked[["thumbsUpCount", "score", "sentiment", "at", "content"]],
    hide_index=True, width="stretch", height=400,
    column_config={
        "thumbsUpCount": st.column_config.NumberColumn("👍", width="small"),
        "score": st.column_config.NumberColumn("★", width="small"),
        "sentiment": "Sentiment",
        "at": st.column_config.DatetimeColumn("Date", format="DD MMM"),
        "content": st.column_config.TextColumn("Review", width="large"),
    },
)
