#!/usr/bin/env bash
# Reproducible pipeline template.
#
# Replace the body of this script with whatever commands regenerate your
# final figure from scratch. A grader should be able to clone your repo
# and run `bash run.sh` to get your output.
#
# Keep it deterministic:
#   - No manual steps.
#   - Data goes in `data/`, figures go in `figures/`.
#   - If you download/scrape data, do it here too (see the S&P 500
#     example below for a starting pattern).

set -euo pipefail

mkdir -p data figures

# ---------------------------------------------------------------------------
# Step 1 — (re)generate raw data.
#
# For the three provided datasets, the CSVs are already committed — you do
# not have to regenerate them. But if you want your pipeline to pull the
# freshest version (e.g., for S&P 500), call a fetcher here. Example:
#
#   uvx --with pandas --with lxml python scripts/fetch_sp500.py
#
# If you bring your own data, put the download / scrape / assemble step
# here so a grader can reproduce it without any manual clicking.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 2 — run the analysis and save the figure.
#
# Adapt the filename to whatever you named your submission.
# ---------------------------------------------------------------------------
uvx marimo run --sandbox submission.py --output figures/scatter.png

echo "Done. See figures/scatter.png"
