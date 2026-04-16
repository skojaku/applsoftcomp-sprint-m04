#!/usr/bin/env python3
"""Fetch the current S&P 500 company list (name + sector) from Wikipedia
and save it to data/sp500.csv.

Usage:
    python scripts/fetch_sp500.py
"""

import csv
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

# Wikipedia's S&P 500 list page (updated regularly)
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# GICS sector name mapping (Wikipedia uses full GICS names)
SECTOR_RENAME = {
    "Information Technology": "Information Technology",
    "Communication Services": "Communication Services",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Health Care": "Health Care",
    "Financials": "Financials",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Materials": "Materials",
    "Real Estate": "Real Estate",
}

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sp500.csv"


def fetch_sp500() -> pd.DataFrame:
    """Read the S&P 500 constituents table from Wikipedia and return
    a DataFrame with columns [name, sector]."""
    tables = pd.read_html(WIKI_URL)
    # The first table on the page is the current S&P 500 list
    df = tables[0]

    # Column names vary slightly over time; find the right ones
    name_col = next(c for c in df.columns if "security" in c.lower())
    sector_col = next(c for c in df.columns if "gics" in c.lower() and "sector" in c.lower())

    df = df[[name_col, sector_col]].copy()
    df.columns = ["name", "sector"]

    # Normalize sector names (strip whitespace, standardize)
    df["sector"] = df["sector"].str.strip().map(
        lambda s: SECTOR_RENAME.get(s, s)
    )

    # Sort by sector then name for readability
    df = df.sort_values(["sector", "name"]).reset_index(drop=True)
    return df


def main():
    print(f"Fetching S&P 500 list from {WIKI_URL} ...")
    df = fetch_sp500()
    print(f"Found {len(df)} companies across {df['sector'].nunique()} sectors")

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    # Print a quick summary
    print("\nCompanies per sector:")
    for sector, count in df["sector"].value_counts().sort_index().items():
        print(f"  {sector}: {count}")


if __name__ == "__main__":
    main()