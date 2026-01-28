# ActDetect — Activity Labelling from Label-Scarce GPS Stays (Paris → London)

This repository contains a reproducible pipeline for **stay-level activity labelling** on large-scale, passively collected GPS/stay data **without ground-truth labels**, validated on a **Paris GPS survey dataset with ground-truth purposes** and transferred to **London-region users**.

The core idea is a **hybrid transfer pipeline**:
- a **supervised per-stay classifier** trained on Paris ground truth (GBDT/MLP),
- **HMM/Viterbi sequential smoothing** to reduce fragmentation and enforce plausible transitions,
- **iterative self-profiling** to improve long-horizon consistency,
- **primary home/work consolidation** to produce OD-ready anchors for commute analysis.

---

## Repository structure

- `notebooks/`  
  End-to-end notebooks (recommended execution order below).
- `src/`  
  Small reusable utilities used by notebooks:
  - `utils_time.py` time zone conversion + cross-midnight splitting
  - `utils_split.py` cohort split helpers
  - `regularity.py` regularity reports / summaries
  - `viz_style.py` scientific plotting style
  - `config.py` shared constants (time zones, label abbreviations, etc.)
- `assets/`  
  Local data inputs (not committed if restricted).
- `outputs/` *(generated)*  
  Figures, tables, intermediate parquet outputs.
- `prev/`  
  Archive / earlier iterations.

---

## Activity taxonomy

Unified 7-class taxonomy used throughout:
- `HOME`, `WORK`, `STUDY`, `PURCHASE`, `LEISURE`, `HEALTH`, `OTHER`

Abbreviations for motif plots are defined in `src/config.py`.

---

## Data (not included)

This project uses privacy-preserved GPS datasets and POI sources that may be restricted.

- **Paris ground-truth dataset**: Paris GPS survey dataset with ground-truth purposes (NetMob25).
- **UK GPS/stay dataset**: long-horizon passively collected UK GPS/stay data; current experiments focus on London-region users.
- **POIs**:
  - UK: Ordnance Survey Points of Interest (OS POI)
  - Paris: OpenStreetMap POIs extracted from Geofabrik regional exports

---

## Quickstart

### Environment
Create a Python environment compatible with Jupyter notebooks (Python 3.9 recommended).

```bash
conda create -n actdetect python=3.9 -y
conda activate actdetect
pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib pyarrow h3
pip install geopandas shapely folium ipywidgets
