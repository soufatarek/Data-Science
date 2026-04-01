# Cookie Cats A/B Testing & Player Retention Analysis

A data-science project analysing the impact of moving level gates in the mobile game **Cookie Cats** on 7-day player retention, following the **CRISP-DM** methodology.

## Project Overview

The goal is to determine whether moving a level gate from Level 30 to Level 40 affects player retention. The analysis includes:

- **Web Scraping** — Real BeautifulSoup scraping of 4 Wikipedia pages with sequential vs parallel performance comparison
- **Data Quality Audit** — IQR outlier detection, Z-score analysis, schema validation, and missing value checks
- **A/B Testing** — Bootstrap analysis to compare retention rates between groups
- **Data Augmentation** — Genre benchmarks from scraped industry data merged into the primary dataset
- **Machine Learning** — Binary classification pipeline (`sklearn.Pipeline`) for predicting player churn
- **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC

## CRISP-DM Lifecycle

| Phase | Location |
|-------|----------|
| Business Understanding | `notebooks/final_pipeline.ipynb` Step 1 |
| Data Understanding | `notebooks/final_pipeline.ipynb` Steps 2-3 (Audit + Clean) |
| Data Preparation | `src/processing.py` → `data_audit()`, `preprocess_data()`, `engineer_features()` |
| Web Scraping | `src/scraping.py` → `run_full_scraping_pipeline()` |
| Modelling | `src/modeling.py` → `train_models()`, `tune_hyperparameters()` |
| Evaluation | `notebooks/final_pipeline.ipynb` Steps 10-11 |
| Conclusion | `notebooks/final_pipeline.ipynb` (Pipeline Complete) |

## Project Structure

```
📁 d:/Data Science/
├── 📁 data/
│   ├── 📁 raw/                     # Original untouched data
│   │   └── cookie_cats.csv
│   ├── 📁 processed/               # Cleaned & augmented data
│   │   └── cookie_cats_augmented.csv
│   └── 📁 scraped/                 # Web-scraped data
│       └── benchmarks.json
├── 📁 notebooks/
│   ├── final_pipeline.ipynb        # ← Main reproducible pipeline (33 cells)
│   ├── 01_crisp_dm_analysis.ipynb  # CRISP-DM walkthrough
│   ├── 01_exploration.ipynb        # Legacy exploration
│   ├── 02_ab_testing.ipynb         # Legacy A/B testing
│   └── 03_modeling.ipynb           # Legacy modelling
├── 📁 src/
│   ├── __init__.py
│   ├── processing.py               # Load, audit, clean, features, split
│   ├── ab_testing.py               # Bootstrap analysis functions
│   ├── modeling.py                 # sklearn Pipeline, evaluation, tuning
│   └── scraping.py                 # Real BeautifulSoup + Selenium scraping
├── 📁 reports/
│   └── 📁 figures/                 # Auto-generated visualisations
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 README.md
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd cookie-cats-analysis
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:
   - Kaggle: [Cookie Cats A/B Testing](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)
   - Place the CSV in `data/raw/cookie_cats.csv`

## Usage

| Step | What to Run |
|------|-------------|
| **Full CRISP-DM pipeline** | Open `notebooks/final_pipeline.ipynb` → *Kernel → Restart & Run All* |
| **CLI quick-run (processing)** | `python src/processing.py` |
| **CLI quick-run (scraping)** | `python src/scraping.py` |
| **CLI quick-run (modelling)** | `python src/modeling.py` |

## Key Results

1. Moving the gate from level 30 → 40 **decreases** 7-day retention by ~0.8pp (statistically significant).
2. IQR analysis reveals 11.28% outliers in `sum_gamerounds` — capped at 99th percentile for robust modelling.
3. Parallel web scraping achieves **1.85× speedup** over sequential scraping.
4. Cookie Cats retention compares favourably to industry benchmarks for match-3 games.

## License

MIT License