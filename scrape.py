# scrape.py - Step 1: download the newest Play Store reviews for each app.
# Output: data/reviews.csv (one row per review, no usernames).
import time

import pandas as pd
from google_play_scraper import Sort, reviews

import config

# The only fields we keep. Usernames and profile pictures are dropped on purpose (privacy).
KEEP = ["reviewId", "content", "score", "thumbsUpCount", "at", "appVersion"]


def fetch_reviews(app_name, package_id):
    """Fetch the newest reviews for one app, one page at a time."""
    rows = []
    token = None  # continuation_token = a bookmark telling Google where the last page ended

    while len(rows) < config.REVIEWS_PER_APP:
        page, token = reviews(
            package_id,
            lang=config.LANG,
            country=config.COUNTRY,
            sort=Sort.NEWEST,
            count=config.PAGE_SIZE,
            continuation_token=token,
        )
        if not page:  # no more reviews available
            break

        for r in page:
            row = {k: r.get(k) for k in KEEP}  # copy only the safe fields
            row["app"] = app_name
            rows.append(row)

        print(f"  {app_name}: {len(rows)} reviews fetched")
        if token is None:  # no bookmark means we reached the end
            break
        time.sleep(1)  # be polite to Google's servers between pages

    return rows[: config.REVIEWS_PER_APP]


def main():
    config.DATA_DIR.mkdir(exist_ok=True)

    all_rows = []
    for app_name, package_id in config.APPS.items():
        print(f"Scraping {app_name} ({package_id})...")
        all_rows += fetch_reviews(app_name, package_id)

    df = pd.DataFrame(all_rows)
    fetched = len(df)

    # Clean up: remove duplicates, then mark reviews that are too short to be useful
    df = df.drop_duplicates(subset="reviewId")
    df["content"] = df["content"].fillna("").str.strip()
    df["is_long"] = df["content"].str.split().str.len() >= config.MIN_WORDS

    # Measure the effect of the filter BEFORE applying it (shows the selection bias)
    summary = df.groupby("app").apply(summarise, include_groups=False).reset_index()
    summary.to_csv(config.SCRAPE_SUMMARY_CSV, index=False)

    df = df[df["is_long"]].drop(columns="is_long")
    df.to_csv(config.REVIEWS_CSV, index=False)

    print(f"\nFetched {fetched} reviews, kept {len(df)} after dropping duplicates and reviews under {config.MIN_WORDS} words.")
    print(summary.to_string(index=False))
    print(f"Saved to {config.REVIEWS_CSV} and {config.SCRAPE_SUMMARY_CSV}")


def summarise(g):
    """Per-app numbers comparing ALL fetched reviews with the KEPT (4+ word) ones."""
    kept, short = g[g["is_long"]], g[~g["is_long"]]
    dates = pd.to_datetime(kept["at"])
    return pd.Series({
        "fetched": len(g),
        "avg_rating_all": round(g["score"].mean(), 2),
        "kept": len(kept),
        "avg_rating_kept": round(kept["score"].mean(), 2),
        "short_dropped": len(short),
        "pct_short_4_5_star": round((short["score"] >= 4).mean() * 100, 1),
        "oldest": dates.min().date(),
        "newest": dates.max().date(),
        "days": (dates.max() - dates.min()).days + 1,
    })


if __name__ == "__main__":
    main()
