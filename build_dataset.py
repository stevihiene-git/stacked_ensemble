"""
Reddit Dataset Preparation Script
===================================
Rebuilds the corrected, enriched dataset used in Chapter 4 from the raw
uploaded files. Documents and fixes two real data-quality issues found in
the source export (see Chapter 3, Section 3.3.1):

  1. Pre-computed URL-lexical features were computed on the Reddit permalink
     instead of the actual embedded external URL (100% of rows affected).
     Fix: extract the real external URL from raw post title/body text and
     recompute all URL-lexical features on the correct link.

  2. The primary raw-post-text export (reddit_posts.csv) has broken CSV
     structure (unescaped commas/newlines in free-text fields), corrupting
     roughly two-thirds of rows on parse.
     Fix: use new_all_posts.csv instead, which is a properly quoted export
     of the same underlying crawl.

Also adds a smoothed subreddit-level phishing-rate feature, found during
exploratory analysis to be, by far, the single strongest predictor
available (r ~= 0.43 vs <=0.03 for every other individual feature).

USAGE
-----
    python build_dataset.py --uploads_dir /path/to/uploaded/csvs --out enriched_dataset_final.csv
"""

import argparse
import os
import re
import pandas as pd
from urllib.parse import urlparse

MD_LINK = re.compile(r'\]\((https?://[^\s\)]+)\)')
BARE_URL = re.compile(r'(https?://[^\s\)\]]+)')

SMOOTHING_K = 15  # subreddit target-encoding smoothing constant


def extract_external_url(row):
    """Extracts the first non-Reddit URL from a post's title+body text."""
    text = f"{row.get('selftext', '')} {row.get('title', '')}"
    if not isinstance(text, str):
        return None
    urls = MD_LINK.findall(text) or BARE_URL.findall(text)
    for u in urls:
        u = u.rstrip(".,)")
        if "reddit.com" not in u and "redd.it" not in u:
            return u
    return None


def url_features(u: str) -> pd.Series:
    """Computes lexical features from a real (correctly-identified) URL."""
    try:
        domain = urlparse(u).netloc
    except Exception:
        domain = ""
    return pd.Series({
        "link_length": len(u),
        "is_https": int(u.lower().startswith("https")),
        "num_dots": u.count("."),
        "num_digits": sum(c.isdigit() for c in u),
        "num_special_chars": sum(not c.isalnum() and c not in ".:/" for c in u),
        "domain_length": len(domain),
    })


def build_dataset(uploads_dir: str) -> pd.DataFrame:
    posts = pd.read_csv(os.path.join(uploads_dir, "new_all_posts.csv"),
                         encoding="latin1", on_bad_lines="skip")
    posts = posts.drop_duplicates(subset=["id"]).copy()

    feat = pd.read_csv(os.path.join(uploads_dir, "reddit_post_features.csv"),
                        encoding="latin1", on_bad_lines="skip")
    feat = feat.drop_duplicates(subset=["PostID"]).copy()

    # Step 1: extract the real external URL from raw text
    posts["extracted_url"] = posts.apply(extract_external_url, axis=1)
    posts_with_url = posts[posts["extracted_url"].notna()].copy()
    print(f"Posts with extractable external URL: {len(posts_with_url)} / {len(posts)} "
          f"({len(posts_with_url) / len(posts):.1%})")

    # Step 2: recompute URL-lexical features on the CORRECT url
    url_feats = posts_with_url["extracted_url"].apply(url_features)
    posts_with_url = pd.concat(
        [posts_with_url.reset_index(drop=True), url_feats.reset_index(drop=True)], axis=1
    )

    # Step 3: Reddit-side (non-URL-dependent) features, computed directly
    posts_with_url["username_length"] = posts_with_url["username"].astype(str).str.len()
    posts_with_url["username_digit_count"] = posts_with_url["username"].astype(str).apply(
        lambda s: sum(c.isdigit() for c in s)
    )
    posts_with_url["title_length"] = posts_with_url["title"].astype(str).str.len()

    # Step 4: merge label + account age from the feature file (not URL-dependent, safe to reuse)
    merged = posts_with_url.merge(
        feat[["PostID", "isVTPhish", "UserAge"]], left_on="id", right_on="PostID", how="inner"
    )
    merged["label"] = merged["isVTPhish"].astype(int)
    merged = merged.rename(columns={"UserAge": "account_age", "score": "post_score"})

    # Step 5: smoothed subreddit phishing-rate encoding
    global_rate = merged["label"].mean()
    grp = merged.groupby("subreddit")["label"].agg(["sum", "count"])
    grp["subreddit_phish_rate"] = (grp["sum"] + SMOOTHING_K * global_rate) / (grp["count"] + SMOOTHING_K)
    merged = merged.merge(grp[["subreddit_phish_rate"]], on="subreddit", how="left")

    final_cols = [
        "id", "link_length", "is_https", "num_dots", "num_digits", "num_special_chars", "domain_length",
        "username_length", "username_digit_count", "title_length", "post_score", "num_comments",
        "account_age", "subreddit_phish_rate", "label",
    ]
    final_df = merged[final_cols].dropna().rename(columns={"id": "PostID"})
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the corrected, enriched Reddit dataset")
    parser.add_argument("--uploads_dir", type=str, required=True,
                         help="Directory containing new_all_posts.csv and reddit_post_features.csv")
    parser.add_argument("--out", type=str, default="enriched_dataset_final.csv")
    args = parser.parse_args()

    df = build_dataset(args.uploads_dir)
    df.to_csv(args.out, index=False)
    print(f"\nFinal dataset: {df.shape[0]} rows -> saved to {args.out}")
    print(f"Label balance:\n{df['label'].value_counts()}")
