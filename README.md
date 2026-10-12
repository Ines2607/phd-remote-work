# remote_work_utils

Detect home, work, remote-work days and third-place visits from GPS dwell data.

## Author
Inessa Tregubova, PhD student at the Hebrew University of Jerusalem

## Project
This repository supports the PhD *"The effect of remote and hybrid working on urban equilibrium and post-pandemic urban recovery."*  
The empirical work is Gush Dan (Tel Aviv metropolitan area), 2019–2023: commuting, home relocation, and the places people use when they work away from the office.

The published code is meant to be reused on similar GPS dwell data. Device-level mobility is **not** in this repository. The runnable path uses synthetic dwells; the research notebooks are a cleaned record of the Gush Dan analysis and need the authors' data and BigQuery project.

## What you can do here
- Infer a home and a workplace from monthly dwells (nights, weekends, Israeli holidays).
- Flag remote-work days with the paper's legacy metric (`flag_remote_work_smoothed_max_thresh`).
- Score third places (visit patterns, PU model, place2vec).
- Reproduce the pipeline on synthetic data in `demo/`.

## Repository layout

| Path | Role |
|---|---|
| `remote_work_utils/` | Python package (home, work, remote work, third places, place2vec). Settings come from JSON configs, not hardcoded paths. |
| `configs/` | Shared defaults (`template.json`), Gush Dan (`regions/gushdan.json`), and smaller demo overrides. |
| `demo/` | Public walk-through on synthetic data (`00`–`06`). Start here. |
| `research/` | Cleaned copies of the paper notebooks, numbered in pipeline order (`01` home/work → `05` statistical-area remote work). They are a research record, not a second demo. |
| `pipelines/` | CLI: generate demo months; run home → work → remote work for one or more months. |
| `results/` | Aggregate tables and figures (no device IDs; small cells suppressed). |
| `tests/` | Unit tests and a regression check against the original monthly scripts. |
| `tools/` | Repo checks (no device-level data, no hardcoded paths). |
| `geofunctions/` | Older GIS helpers still used by some research notebooks. |
| `conference_slides/` | PDFs from talks (when present). |

Research folders match the demo topics after data generation:

1. `research/01-home-work/` — home relocation panel  
2. `research/02-remote-work/` — metric, thresholds, validation (plus `legacy/` scripts)  
3. `research/03-third-places/` — POI flags, PU model, neighbourhood effects  
4. `research/04-place2vec/` — embeddings and n-grams  
5. `research/05-stat-area-remote-work/` — statistical-area monthly / quarterly / panel datasets  

## Demo notebooks
Run from `demo/` after installing the package. All use synthetic devices (`synthetic-…` IDs).

| Notebook | Content |
|---|---|
| `00-synthetic-data` | Draw a small city and three months of dwells (`configs/demo.json`) |
| `01-dwell-statistics` | Coverage and outliers |
| `02-home-work-detection` | Home and workplace |
| `03-remote-work-metric` | Office / remote / unknown days |
| `04-third-places` | Candidate places and PU scores |
| `05-place2vec` | Place embeddings vs work context |
| `06-mobility` | Radius of gyration and related metrics |

## Data
**In this repo:** synthetic dwells and places under `demo/data/` (created by notebook `00` or `python pipelines/make_demo_data.py`), plus public aggregates under `results/`.

**Used in the dissertation, not published here:** Unicast/Habidatum GPS dwells for Gush Dan (2019–2023). Research notebooks also join Israel Census 2022, statistical-area boundaries, and (for validation) public-transport ticket totals.

To point the library at your own files, copy `settings.example.ini` to `settings.ini` (gitignored) and use a region config such as `configs/regions/gushdan.json`. Expected dwell columns are those used in `demo/00` (`identifier`, timestamps, coordinates, optional `geohash` / `province`).

## Install and run

```bash
git clone https://github.com/Ines2607/phd-remote-work.git
cd phd-remote-work
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[place2vec,econ,dev]"
