# 🔍 ReviewLens

**AI product teardowns from Play Store reviews.** ReviewLens reads thousands of Google Play reviews of competing Indian apps, groups complaints into themes with an LLM, tests which differences between the apps are statistically real, and writes a prioritised product teardown.

**Live demo:** [your-app.streamlit.app](https://your-app.streamlit.app) <!-- replace after deploying -->

![ReviewLens dashboard](docs/screenshot.png)
<!-- Add a screenshot: open the dashboard, take a full-window screenshot, save it as docs/screenshot.png -->

---

## The problem

A product or operations team at a food-delivery startup gets hundreds of app reviews a day, in English, Hindi and Hinglish. Reading them by hand doesn't scale, and star ratings don't say *what* is broken. ReviewLens answers three questions:

1. **What are users complaining about most?** (ranked pain points)
2. **Are we worse than the competitor, and is the gap real or noise?** (significance-tested comparison)
3. **What should we fix first?** (an AI-written teardown grounded only in computed numbers)

The first case study compares **Swiggy** and **Zomato**.

## Key finding

> **Swiggy users complain about late delivery 1.9× as often as Zomato users: 14.8% vs 7.7% of reviews**, a gap of **+7.1 percentage points** (95% CI +3.6 to +10.6 pp, p ≈ 0.0001).

*Based on 1,275 of 4,531 reviews classified so far (a random sample, 27 Oct – 12 Nov 2026). The dashboard recalculates this sentence from the data as classification continues.*

How the finding was checked:

| Check | Result |
|---|---|
| **Statistical significance**: two-proportion z-test for each of 13 complaint themes | Late delivery is the only significant gap. It still passes after **Bonferroni correction** (p < 0.05 / 13 = 0.0038), so it is very unlikely to be a lucky result from testing many themes. |
| **Instamart confound**: the Swiggy app includes Instamart (groceries), while Zomato's Blinkit is a separate app | 73 of 2,025 Swiggy reviews mention Instamart or grocery, but only **1 of 87** late-delivery complaints does. Excluding all of them, Swiggy's late-delivery rate is **15.2%**, so groceries don't explain the gap. |
| **Diwali effect**: maybe the gap is a festival rush (Diwali was 8 Nov 2026) | Late-delivery share on festival days (6–11 Nov) vs other days: Swiggy 15.9% vs 14.2% (p = 0.57), Zomato 8.7% vs 7.0% (p = 0.41). **No significant spike**, and the gap exists on ordinary days too. |

## How it works

```
Google Play ──► scrape.py ──► themes.py ──► classify.py ──► app.py (Streamlit)
               10k reviews    LLM proposes   LLM labels each    metrics, pain points,
               per app        a shared       review: theme +    significance tests,
                              theme list     sentiment          trends, AI teardown
                                                  │
                                              eval.py: compare with hand labels
```

| Step | File | What it does | Key design choices |
|---|---|---|---|
| 1 | `scrape.py` | Fetches the newest 10,000 reviews per app with `google-play-scraper`, page by page using a continuation token | Keeps only the review text, rating, upvotes, date and app version. **Usernames are never stored.** Drops reviews under 4 words, and measures the effect of that filter before applying it. |
| 2 | `themes.py` | Sends a balanced sample (70 negative + 30 positive, half per app) to the LLM, which proposes the themes | **One shared theme list** for both apps, so the comparison is fair. The list was then edited by hand into 16 themes with tie-break rules. |
| 3 | `classify.py` | Labels every review with one theme and a sentiment, 25 reviews per call | Shuffled with a fixed seed, so partial results are a random sample. Saves after every batch, so it can resume after a crash. Compact JSON output saves tokens. Paced to the free tier's per-minute and daily token limits, and keeps a token reserve. |
| 4 | `app.py` | Streamlit dashboard | Percentages, not counts. Two-proportion z-tests with confidence intervals. The teardown prompt contains **only computed numbers and real quotes**, and the LLM is told not to invent anything. |
| 5 | `eval.py` | Exports 150 reviews (balanced, separate seed) for hand-labelling without the AI's labels, then reports accuracy, a confusion table and the mistakes | The same written guide (`data/theme_guide.txt`) is given to both the human labeller and the LLM. |

**Stack:** Python 3.13 · pandas · Streamlit · Altair · OpenAI Python SDK pointed at Groq (`openai/gpt-oss-120b`). Any OpenAI-compatible provider works by changing `LLM_BASE_URL` and `LLM_MODEL`.

## Classification accuracy

| | Theme accuracy | Sentiment accuracy |
|---|---|---|
| Hand-labelled sample (n = 150) | [pending evaluation] | [pending evaluation] |

## About the data: selection bias

Reviews under 4 words ("good app", "nice 👍") carry no detail about *what* is wrong, so they were removed. Most of them were positive:

| App | Reviews fetched | Avg ★ (all) | Kept (4+ words) | Avg ★ (kept) | Short reviews dropped | % of dropped that were 4–5★ |
|---|---|---|---|---|---|---|
| Swiggy | 10,000 | 3.64 | 3,380 | 2.16 | 6,620 | 85.1% |
| Zomato | 10,000 | 4.00 | 2,506 | 2.59 | 7,494 | 86.8% |

**So ReviewLens ranks problems; it does not measure overall customer satisfaction.** The average ratings shown are much lower than the Play Store's (about 4.5★) by design.

## Limitations

- **Review bias:** reviewers are self-selected and skew towards strong opinions, and the 4-word filter removes mostly positive reviews (see above). Use the shares to *rank* problems, not to estimate how many customers are affected.
- **Unequal time coverage:** Zomato gets more reviews per day, so both apps are compared only over the window both cover (27 Oct – 12 Nov 2026). Partial first and last days are hidden from the daily charts.
- **Partial classification:** on Groq's free tier, about 80 batches fit in a day, so classification runs over several days. Until it finishes, results come from a random sample, and the dashboard says how large it is.
- **Small upvote counts:** the reviews are only days or weeks old, so "most upvoted" is based on very few votes.
- **Keyword-based Instamart check:** reviews about groceries that don't say "Instamart" or "grocery" are missed.
- **LLM labels are not ground truth:** one theme per review loses secondary complaints, and accuracy is measured against a single human labeller.
- **English-store reviews only** (`lang=en, country=in`), including Hinglish but not Devanagari-only reviews.

## Run it locally

```bash
git clone https://github.com/<your-username>/reviewlens.git
cd reviewlens
python3.13 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Create a .env file (never commit it):
#   LLM_API_KEY=your-groq-key
#   LLM_BASE_URL=https://api.groq.com/openai/v1
#   LLM_MODEL=openai/gpt-oss-120b

streamlit run app.py          # dashboard, using the data already in data/
```

To rebuild the data from scratch: `python scrape.py` → `python themes.py` (then edit `data/themes.json`) → `python classify.py` (re-run until done) → `python eval.py sample`, label `data/label_me.csv`, then `python eval.py`.

## Project structure

```
app.py            Streamlit dashboard
config.py         all settings (apps, limits, date window, file paths)
llm.py            LLM calls: retries, JSON extraction, token logging, daily-budget estimate
scrape.py         Step 1 - download reviews
themes.py         Step 2 - propose themes
classify.py       Step 3 - label reviews (resumable)
eval.py           Step 5 - accuracy against hand labels
data/             reviews.csv, themes.json, classified.csv, scrape_summary.csv, label_me.csv
```
