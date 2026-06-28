"""
scrape_reddit.py
================
Scrape all comments from a Reddit thread and save to:
  - data/reddit_ptsd_comments.xlsx   (human-reviewable Excel)
  - data/golden_dataset.json         (pipeline golden dataset schema)

Usage:
    python scrape_reddit.py

Requires a Reddit API app (free, takes 2 minutes to create):
  1. Go to https://www.reddit.com/prefs/apps
  2. Click "create another app..." at the bottom
  3. Name: anything (e.g. "ptsd-research")
     Type: select "script"
     Redirect URI: http://localhost:8080
  4. Click "create app" — note your CLIENT_ID and CLIENT_SECRET
  5. Set them below OR in a .env file:
       REDDIT_CLIENT_ID=...
       REDDIT_CLIENT_SECRET=...
       REDDIT_USERNAME=...      (your Reddit username)
       REDDIT_PASSWORD=...      (your Reddit password)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config — fill in or set via environment variables / .env
# ---------------------------------------------------------------------------

THREAD_URL = "https://www.reddit.com/r/ptsd/comments/177i61u/what_does_ptsd_feel_like/"

CLIENT_ID     = os.environ.get("REDDIT_CLIENT_ID",     "")
CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
USERNAME      = os.environ.get("REDDIT_USERNAME",      "")
PASSWORD      = os.environ.get("REDDIT_PASSWORD",      "")

OUT_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Credential check
# ---------------------------------------------------------------------------


def _check_credentials() -> None:
    missing = [k for k, v in {
        "REDDIT_CLIENT_ID": CLIENT_ID,
        "REDDIT_CLIENT_SECRET": CLIENT_SECRET,
        "REDDIT_USERNAME": USERNAME,
        "REDDIT_PASSWORD": PASSWORD,
    }.items() if not v.strip()]
    if missing:
        print("\nMissing Reddit API credentials:", ", ".join(missing))
        print("\nHow to get credentials (free, ~2 minutes):")
        print("  1. Go to https://www.reddit.com/prefs/apps")
        print("  2. Click 'create another app...'")
        print("  3. Name: anything | Type: script | Redirect URI: http://localhost:8080")
        print("  4. Click 'create app' — copy CLIENT_ID (under app name) and CLIENT_SECRET")
        print("\nThen run with environment variables:")
        print("  $env:REDDIT_CLIENT_ID='your_id'")
        print("  $env:REDDIT_CLIENT_SECRET='your_secret'")
        print("  $env:REDDIT_USERNAME='your_reddit_username'")
        print("  $env:REDDIT_PASSWORD='your_reddit_password'")
        print("  python scrape_reddit.py")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------


def scrape(thread_url: str) -> list[dict]:
    import praw

    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        username=USERNAME,
        password=PASSWORD,
        user_agent="ptsd-research-scraper/1.0",
    )

    submission = reddit.submission(url=thread_url)
    print(f"Thread: {submission.title}")
    print(f"Expanding all comments (this may take a minute)…")
    submission.comments.replace_more(limit=None)  # fetch all collapsed comment chains

    records: list[dict] = []
    _walk(submission.comments.list(), records, submission.id)
    return records


def _walk(comments, records: list[dict], submission_id: str) -> None:
    for c in comments:
        body = getattr(c, "body", "").strip()
        if not body or body in {"[deleted]", "[removed]"}:
            continue
        records.append({
            "id": c.id,
            "author": str(c.author) if c.author else "[deleted]",
            "text": body,
            "score": c.score,
            "depth": c.depth,
            "parent_id": c.parent_id,
            "created_utc": datetime.fromtimestamp(
                c.created_utc, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "permalink": f"https://www.reddit.com{c.permalink}",
        })


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def to_excel(comments: list[dict], out_path: Path) -> None:
    """Write the scraped comments to an Excel file with a frozen header and sized columns."""
    df = pd.DataFrame(comments, columns=[
        "id", "author", "text", "score", "depth", "parent_id", "created_utc", "permalink"
    ])
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="comments")
        ws = writer.sheets["comments"]
        ws.freeze_panes = "A2"
        col_widths = {
            "id": 12, "author": 20, "text": 80, "score": 8,
            "depth": 7, "parent_id": 15, "created_utc": 20, "permalink": 55,
        }
        for col_idx, col_name in enumerate(df.columns, start=1):
            ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = col_widths.get(col_name, 15)
    print(f"Excel saved  → {out_path}  ({len(df)} rows)")


def to_golden_json(comments: list[dict], out_path: Path) -> None:
    """Write the comments as a golden-dataset JSON (text/source/labels schema, labels empty)."""
    records = [
        {"text": c["text"], "source": f"reddit/{c['id']}", "labels": []}
        for c in comments
    ]
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Golden JSON  → {out_path}  ({len(records)} records)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        # Re-read after loading
        globals().update({
            "CLIENT_ID":     os.environ.get("REDDIT_CLIENT_ID",     CLIENT_ID),
            "CLIENT_SECRET": os.environ.get("REDDIT_CLIENT_SECRET", CLIENT_SECRET),
            "USERNAME":      os.environ.get("REDDIT_USERNAME",      USERNAME),
            "PASSWORD":      os.environ.get("REDDIT_PASSWORD",      PASSWORD),
        })
    except ImportError:
        pass

    _check_credentials()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    comments = scrape(THREAD_URL)
    print(f"Total comments: {len(comments)}")

    if not comments:
        print("No comments found.")
    else:
        to_excel(comments, OUT_DIR / "reddit_ptsd_comments.xlsx")
        to_golden_json(comments, OUT_DIR / "golden_dataset.json")
        print("Done.")
