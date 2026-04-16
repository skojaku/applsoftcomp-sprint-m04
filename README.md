# Sprint M05 — Semantic Axes

Build a **semantic map**: pick terms, design two semantic axes from opposing word sets, produce one publication-quality 2D scatterplot.

> [!IMPORTANT]
> **Fork first.** Without forking, your Codespace opens on the original repo and you can't push. Fork → clone your fork → work on it.

## Getting oriented

1. Read this README.
2. Open `assignment.py` with `uvx marimo edit --sandbox assignment.py`. This is a **worked example** on the universities dataset — not the deliverable.
3. Pick a case study, build your own pipeline.

## Pick a case study

One of the three, or bring your own.

| File | Case study | N | Extra columns |
|---|---|---|---|
| `data/universities.csv` | U.S. higher-ed institutions | 157 | `type`, `region` |
| `data/sp500.csv` | S&P 500 (sample) | 203 | `sector` |
| `data/chemicals.csv` | Chemicals / materials | 179 | `class` |

**Own data:** ≥ 100 terms, ≥ 1 categorical attribute for color/shape.

## Tasks

### 1. Two semantic axes

Each axis: 3–6 words for **+ pole**, 3–6 for **− pole**. Good axis:

- Well-separated poles (cosine distance between centroids ≥ 0.3).
- Spreads the data, not piled at midpoint.
- One-sentence interpretation.

Axes should capture different, ideally orthogonal aspects — redundant axes waste half the plot.

### 2. One scatterplot

Plot each term at `(axis1, axis2)`. Encode categorical attributes with **color** and **shape**. Follow data-viz principles:

- **Clarity:** readable symbols/text, no overlapping labels, no default matplotlib-blue soup.
- **Group separability:** groups distinguishable at a glance.
- **Colorblind-friendly:** Okabe–Ito, viridis, etc. Redundantly encode with shape.
- **Pre-attentive attention:** use color/size/position to pull the eye to your story.
- **Gestalt:** proximity, similarity; zero lines, quadrant annotations, text anchors help.

`assignment.py`'s plot is a floor, not a ceiling.

### 3. Observations

2–4 paragraphs in notebook or `REPORT.md`:

- What separates along each axis?
- Most **surprising** point/group — what does it say about the embedding?
- What would a **third axis** capture?

## Deliverable

Your repo must have:

- **Code** for the figure (marimo / Jupyter / `.py`).
- **Reproducible pipeline** — `run.sh` / Makefile / Snakemake that regenerates the figure from scratch. One command, fresh clone. Starter `run.sh` included.
- **Raw data** (CSV in `data/`).
- **Final figure** (PNG/PDF).
- **Observations** inline or in `REPORT.md`.

Submit by pushing and posting the URL to Brightspace.

## Evaluation

| Criterion | What we look for |
|---|---|
| **Atomic git history** | Small focused commits, meaningful messages. Not `final`, `final2`, `final-real`. |
| **Reproducible pipeline** | `bash run.sh` regenerates data + figure on a fresh clone. No manual steps. |
| **Documentation** | Comments where non-obvious. Notebook/report explains *why* each axis and *what* the figure shows. |
| **Viz quality** | Follows Task 2 principles. Clear, separated, colorblind-friendly, deliberate. |
| **Task completion** | Two axes, one scatterplot, observations — all present. |

## FAQ

**Teams?** Yes. Team repo, list members in `REPORT.md`, all submit same URL.

**Embedding model?** `all-MiniLM-L6-v2` default. Larger OK, document the swap.

**Unimodal/boring axis?** That's information. Try named entities over abstractions, or poles that *should* separate your data. Iterate.
