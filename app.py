# app.py - the ReviewLens report page (layout only; all numbers come from analysis.py).
# Run with:  streamlit run app.py
import html
import re

import altair as alt
import pandas as pd
import streamlit as st

import analysis as an
import config

st.set_page_config(page_title="ReviewLens", layout="centered")

COLOURS = {"Swiggy": "#E8762E", "Zomato": "#A3213F"}  # one fixed colour per app in every chart (colour-blind checked)
MUTED = "#C9C8C1"        # anything that is not the point of a chart
INK_SOFT = "#6B6A64"     # secondary text
GITHUB_URL = "https://github.com/Kartikkhati/reviewlens"
LINKEDIN_URL = "https://www.linkedin.com/in/TODO"  # TODO: add my LinkedIn URL
APP_DOMAIN = alt.Scale(domain=list(COLOURS), range=list(COLOURS.values()))

st.markdown("""
<style>
.eyebrow  {font-size:.78rem; letter-spacing:.14em; font-weight:700; color:#1F5F5B; margin:.5rem 0 .5rem}
.headline {font-size:2.15rem; line-height:1.2; font-weight:700; color:#1D1D1B; margin:0 0 .8rem}
.subline  {font-size:.92rem; color:#6B6A64; margin-bottom:2rem}
.check    {font-weight:600; margin:1rem 0 .15rem}
.meta     {font-size:.82rem; color:#6B6A64; margin-top:.35rem}
a, a:visited {color:#1F5F5B !important}
</style>""", unsafe_allow_html=True)


def tidy(chart):
    """Same quiet look for every chart: light grid, soft axis text, no border."""
    return (chart.configure(background="transparent", font="sans-serif")
            .configure_axis(gridColor="#ECEBE6", domainColor="#D6D5CF", tickColor="#D6D5CF", labelColor="#3F3E3A",
                            titleColor=INK_SOFT, labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure_view(stroke=None)
            .configure_legend(labelFontSize=12, symbolSize=110, labelColor="#3F3E3A"))


def plain(text):
    """Escape markdown symbols so a review like '*worst* app $$$' shows exactly as written."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~$<])", r"\\\1", str(text))


def load_recommendations():
    """Read data/recommendations.md: a '## App' heading per app, then my numbered list -> {app: markdown}."""
    path = config.DATA_DIR / "recommendations.md"
    recs, current = {}, None
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                current = line[3:].strip()
                recs[current] = []
            elif current and line.strip():
                recs[current].append(line)
    return {app_name: "\n".join(lines) for app_name, lines in recs.items()}


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def html_line(css_class, text):
    """Show text in a styled block. html.escape turns '<' into '&lt;' so 'p < 0.001' isn't read as a tag."""
    st.markdown(f'<div class="{css_class}">{html.escape(text)}</div>', unsafe_allow_html=True)


# ---------------- Data (re-read from disk at most every 5 minutes) ----------------
@st.cache_data(ttl=300)
def load():
    return an.load()


all_reviews, df, theme_names, summary = load()
if df.empty:
    st.write("No classified reviews yet. Run classify.py first.")
    st.stop()

complaint_themes = [t for t in theme_names if not t.startswith("Praise")]
finding = an.key_finding(df, complaint_themes)
a, b = finding["a"], finding["b"]
total, done = len(all_reviews), len(df)
sample_note = f" · {done / total:.0%} classified so far (random sample)" if done < total else ""

# ---------------- Hero ----------------
if finding["theme"]:
    w, br, hi, lo = finding["worse"], finding["better"], finding["hi"], finding["lo"]
    how_often = f"{an.times_words(hi / lo)} as often as" if lo > 0 else "far more often than"
    headline = f"{w} users complain about {finding['theme'].lower()} {how_often} {br} users."
    subline = f"{hi:.1f}% vs {lo:.1f}% of reviews · based on {done:,} reviews, {an.PERIOD}{sample_note}"
else:
    headline = f"{a} and {b} users complain about the same things."
    subline = f"No complaint differs clearly between the apps · based on {done:,} reviews, {an.PERIOD}{sample_note}"

html_line("eyebrow", "REVIEWLENS")
html_line("headline", headline)
html_line("subline", subline)

# ---------------- The proof: dumbbell chart ----------------
tests = finding["tests"]
n_sig = finding["n_sig"]
if n_sig == 1:
    proof_title = f"{finding['theme']} is the only complaint where the apps really differ"
elif n_sig > 1:
    proof_title = f"The apps really differ on {n_sig} complaints"
else:
    proof_title = "No complaint differs clearly between the apps"
st.subheader(proof_title)
# TODO: rewrite in my words
st.caption(f"Share of each app's reviews that complain about each topic. Grey = gap could be chance.")

t = tests.drop(index=list(an.NOT_ACTIONABLE), errors="ignore")  # vague topics: nothing to compare or fix
t = t[(t[a] > 0) | (t[b] > 0)].copy()
t["low_pct"], t["high_pct"] = t[[a, b]].min(axis=1), t[[a, b]].max(axis=1)
order = t.sort_values("high_pct", ascending=False).index.tolist()
wide = t.reset_index()
long = wide.melt(id_vars=["theme", "significant"], value_vars=[a, b], var_name="app", value_name="pct")
long["key"] = long["app"].where(long["significant"], "No real difference")
long["label"] = long["pct"].map("{:.1f}%".format)

y = alt.Y("theme:N", sort=order, title=None, axis=alt.Axis(labelLimit=0, domain=False, ticks=False, labelPadding=8))
x_title = "% of the app's reviews"
rules = alt.Chart(wide).mark_rule(strokeWidth=2).encode(
    y=y, x=alt.X("low_pct:Q", title=x_title), x2="high_pct:Q",
    color=alt.condition("datum.significant", alt.value(INK_SOFT), alt.value("#E4E3DD")))
dots = alt.Chart(long).mark_circle(opacity=1, stroke="#FBFBF8").encode(
    y=y, x=alt.X("pct:Q", title=x_title),
    # grey dots: smaller and without the white ring, so two close dots merge cleanly instead of overlapping
    size=alt.condition("datum.significant", alt.value(140), alt.value(60)),
    strokeWidth=alt.condition("datum.significant", alt.value(2), alt.value(0)),
    color=alt.Color("key:N", scale=alt.Scale(domain=[a, b, "No real difference"],
                                             range=[COLOURS[a], COLOURS[b], MUTED]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=[alt.Tooltip("theme:N", title="Complaint"), alt.Tooltip("app:N", title="App"),
             alt.Tooltip("pct:Q", title="% of reviews", format=".1f")])
labels = alt.Chart(long[long["significant"]]).mark_text(dy=-14, fontSize=12, fontWeight=600, color="#1D1D1B").encode(
    y=y, x="pct:Q", text="label:N")
st.altair_chart(tidy((rules + dots + labels).properties(height=34 * len(order))), width="stretch", theme=None)

with st.expander("Statistical details", expanded=False):
    # TODO: rewrite in my words
    st.caption(f"A gap is real when p < {finding['alpha']:.4f}: 0.05 divided by the {len(tests)} topics tested "
               "(Bonferroni correction).  \nThe 95% interval is the range the true gap most likely falls in.")
    table = pd.DataFrame({
        "Complaint": tests.index,
        a: tests[a].map("{:.1f}%".format).values,
        b: tests[b].map("{:.1f}%".format).values,
        "Gap": tests["gap"].map("{:+.1f} pp".format).values,
        "95% interval": [f"{lo_:+.1f} to {hi_:+.1f} pp" for lo_, hi_ in zip(tests["ci_low"], tests["ci_high"])],
        "p-value": tests["p"].map(lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}").values,
        "Real difference": tests["significant"].map({True: "Yes", False: "No"}).values,
    })
    styled = table.style.apply(
        lambda r: [""] * len(r) if r["Real difference"] == "Yes" else ["color: #9A9993"] * len(r), axis=1)
    st.dataframe(styled, hide_index=True, width="stretch")

# ---------------- What I checked ----------------
if finding["theme"]:
    theme = finding["theme"]
    st.subheader("What I checked")

    html_line("check", f"Not random chance ({fmt_p(finding['p'])})")
    with st.expander("Details", expanded=False):
        # TODO: rewrite in my words
        st.write(f"If both apps had the same true rate, a gap this big would almost never happen by luck. "
                 f"I tested {len(tests)} topics, so I used a stricter bar: p < {finding['alpha']:.4f}.")

    if "Swiggy" in config.APPS:
        other = next(x for x in config.APPS if x != "Swiggy")
        im = an.instamart_check(all_reviews, df, theme, other)
        verdict = "Not driven by Instamart" if im["survives"] else "Partly driven by Instamart"
        html_line("check", f"{verdict} ({im['rate_without']:.1f}% without grocery reviews)")
        with st.expander("Details", expanded=False):
            # TODO: rewrite in my words
            st.write(f"The Swiggy app includes Instamart (groceries). Zomato's Blinkit is a separate app. "
                     f"{im['window_mentions']} of {im['window_reviews']:,} Swiggy reviews mention Instamart or groceries. "
                     f"Only {im['hits_mentioning']} of {im['hits']} {theme.lower()} complaints do. "
                     f"Without them, Swiggy is at {im['rate_without']:.1f}% vs {im['other_rate']:.1f}% for {other}. "
                     f"This is a keyword search, so grocery reviews that never say 'Instamart' are missed.")

    fest = an.festival_check(df, theme)
    spike = any(v["spike"] for v in fest.values())
    line = "Possible Diwali effect (higher on festival days)" if spike else "Not a Diwali spike (no festival increase)"
    html_line("check", line)
    with st.expander("Details", expanded=False):
        # TODO: rewrite in my words
        st.write("Diwali was on 8 Nov 2026. I compared festival days (6–11 Nov) with all other days.")
        for app_name, v in fest.items():
            st.write(f"{app_name}: {v['festival']:.1f}% on festival days vs {v['other']:.1f}% on other days "
                     f"({fmt_p(v['p'])}).")
        st.write("With about 40 classified reviews a day, a small festival effect could still be missed.")

# ---------------- Trend ----------------
trend_theme = finding["theme"] or "Late delivery"
st.subheader(f"{trend_theme} complaints, day by day")
MIN_DAILY = 15  # a day with fewer classified reviews is too noisy to plot (e.g. Zomato's half-scraped 27 Oct)
# TODO: rewrite in my words
st.caption(f"Share of each day's reviews, {an.PERIOD}. Days with fewer than {MIN_DAILY} reviews are hidden.")
daily = an.daily_share(df, trend_theme)
daily = daily[daily["reviews"] >= MIN_DAILY]
diwali = pd.DataFrame({"date": [pd.Timestamp("2026-11-08")], "label": ["Diwali"]})
lines = alt.Chart(daily).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
    x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, grid=False)),
    y=alt.Y("share:Q", title="% of the day's reviews"),
    color=alt.Color("app:N", scale=APP_DOMAIN, legend=alt.Legend(orient="top", title=None)),
    tooltip=[alt.Tooltip("date:T", title="Date", format="%a %d %b"), alt.Tooltip("app:N", title="App"),
             alt.Tooltip("share:Q", title="% of reviews", format=".1f"), alt.Tooltip("reviews:Q", title="Reviews that day")])
marker = alt.Chart(diwali).mark_rule(strokeDash=[4, 4], color="#9A9993").encode(x="date:T")
marker_text = alt.Chart(diwali).mark_text(align="left", dx=5, dy=8, color=INK_SOFT, fontSize=11).encode(
    x="date:T", y=alt.value(0), text="label:N")
st.altair_chart(tidy((marker + marker_text + lines).properties(height=300)), width="stretch", theme=None)

# ---------------- What to fix ----------------
fix_title = st.empty()  # placeholder: the heading names the app picked in the toggle just below it
app = st.segmented_control("App", list(config.APPS), default=list(config.APPS)[0], key="fix_app",
                           label_visibility="collapsed") or list(config.APPS)[0]
fix_title.subheader(f"What {app} should fix first")
mine = df[df["app"] == app]
pain = an.pain_points(mine, complaint_themes)
fix = pain[~pain["theme"].isin(an.NOT_ACTIONABLE)].reset_index(drop=True)
fix["colour"] = [COLOURS[app]] + [MUTED] * (len(fix) - 1)  # highlight only the top problem
fix["label"] = (fix["share"] * 100).round().astype(int).astype(str) + "%"
n_neg = (mine["sentiment"] == "negative").sum()
# TODO: rewrite in my words
st.caption(f"Share of {app}'s {n_neg:,} complaints. Vague ones like 'worst app' are not shown.")
fy = alt.Y("theme:N", sort=None, title=None, axis=alt.Axis(labelLimit=0, domain=False, ticks=False, labelPadding=8))
fbars = alt.Chart(fix).mark_bar(cornerRadiusEnd=4, height=18).encode(
    y=fy, x=alt.X("share:Q", axis=None), color=alt.Color("colour:N", scale=None),
    tooltip=[alt.Tooltip("theme:N", title="Complaint"), alt.Tooltip("label:N", title="Share"),
             alt.Tooltip("reviews:Q", title="Reviews")])
ftext = alt.Chart(fix).mark_text(align="left", dx=6, fontSize=12, color="#3F3E3A").encode(y=fy, x="share:Q", text="label:N")
st.altair_chart(tidy((fbars + ftext).properties(height=32 * len(fix))), width="stretch", theme=None)

# Connect this chart (share of complaints) to the headline (share of ALL reviews) - two different denominators
note_theme = finding["theme"] or (fix["theme"].iloc[0] if len(fix) else None)
if note_theme:
    of_complaints = pain.set_index("theme")["share"].get(note_theme, 0) * 100
    of_all = tests.loc[note_theme, app]
    in_headline = " (the number in the headline)" if note_theme == finding["theme"] else ""
    # TODO: rewrite in my words
    st.caption(f"{note_theme} is {of_complaints:.0f}% of {app}'s complaints, or {of_all:.1f}% of all its "
               f"reviews{in_headline}.")

# My own recommendations, written by hand in data/recommendations.md
st.markdown("**My recommendations**")
st.markdown(load_recommendations().get(app, "TODO: add recommendations to data/recommendations.md"))

# ---------------- Read the reviews ----------------
st.subheader("Read the reviews")


def reset_count():
    st.session_state["shown"] = 10


def show_more():
    st.session_state["shown"] = st.session_state.get("shown", 10) + 10


c1, c2 = st.columns([1, 2])
with c1:
    read_app = st.segmented_control("App ", list(config.APPS), default=list(config.APPS)[0], key="read_app",
                                    on_change=reset_count) or list(config.APPS)[0]
with c2:
    default_theme = theme_names.index(trend_theme) if trend_theme in theme_names else 0
    read_theme = st.selectbox("Topic", theme_names, index=default_theme, key="read_theme", on_change=reset_count)

picked = df[(df["app"] == read_app) & (df["theme"] == read_theme)].sort_values("at", ascending=False)
shown = st.session_state.get("shown", 10)
st.caption(f"{len(picked):,} reviews · newest first")
for r in picked.head(shown).itertuples():
    with st.container(border=True):
        st.markdown(plain(r.content))
        stars = "★" * int(r.score) + "☆" * (5 - int(r.score))
        html_line("meta", f"{stars}  ·  {r.at:%d %b %Y}")
if len(picked) > shown:
    st.button(f"Show more ({len(picked) - shown:,} left)", on_click=show_more)

# ---------------- About ----------------
st.divider()
st.subheader("About this project")
st.markdown(f"Built by **Kartik Khati** · [GitHub]({GITHUB_URL}) · [LinkedIn]({LINKEDIN_URL})")
# TODO: rewrite in my words
st.write("TODO: why I built this.")

st.markdown("**How it works**")
# TODO: rewrite in my words
st.markdown(
    "1. Downloaded the newest 10,000 Google Play reviews per app and kept those with 4 or more words.\n"
    "2. An LLM proposed complaint topics. I edited them into one shared list of 16.\n"
    "3. The LLM labelled every review with one topic and a sentiment, 25 reviews at a time.\n"
    "4. This page compares the apps with significance tests and checks the main finding.")

# Latest accuracy result from eval.py (None until I've hand-labelled and run the evaluation)
eval_file = config.DATA_DIR / "eval_results.csv"
latest_eval = pd.read_csv(eval_file).iloc[-1] if eval_file.exists() else None

st.markdown("**Limitations**")
# TODO: rewrite in my words
limits = [
    "Reviews skew negative, because short reviews (mostly positive) were removed. This ranks problems; "
    "it does not measure satisfaction.",
    "Upvote counts are small, because the reviews are recent.",
    "The Instamart check uses keywords, so it can miss grocery reviews.",
    "AI labels are not perfect. " + ("See accuracy below." if latest_eval is not None
                                     else "Accuracy check in progress."),
]
if done < total:
    limits.insert(1, f"{done / total:.0%} of reviews are classified so far. Numbers are from a random sample.")
st.markdown("\n".join(f"- {x}" for x in limits))

st.markdown("**The short-review filter**")
# TODO: rewrite in my words
st.caption("Most removed reviews were 4–5 stars, so the reviews left behind skew negative.")
st.dataframe(
    summary[["app", "fetched", "avg_rating_all", "kept", "avg_rating_kept", "pct_short_4_5_star"]],
    hide_index=True, width="stretch",
    column_config={
        "app": "App",
        "fetched": st.column_config.NumberColumn("Reviews fetched", format="localized"),
        "avg_rating_all": st.column_config.NumberColumn("Avg stars, all", format="%.2f"),
        "kept": st.column_config.NumberColumn("Kept (4+ words)", format="localized"),
        "avg_rating_kept": st.column_config.NumberColumn("Avg stars, kept", format="%.2f"),
        "pct_short_4_5_star": st.column_config.NumberColumn("Removed that were 4–5 stars", format="%.0f%%"),
    })

if latest_eval is not None:  # hidden completely until the evaluation exists
    sentiment = latest_eval["sentiment_accuracy"]
    sentiment_part = f" (sentiment: {sentiment:.0%})" if pd.notna(sentiment) else ""
    # TODO: rewrite in my words
    st.markdown(f"**Accuracy:** {latest_eval['theme_accuracy']:.0%} of topics matched my own labels on "
                f"{int(latest_eval['n'])} hand-labelled reviews{sentiment_part}.")
# TODO: rewrite in my words
st.caption(f"Data: Google Play, India, English store, {an.PERIOD}. Topics and sentiment by {config.LLM_MODEL}.")
