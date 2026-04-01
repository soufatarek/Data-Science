# Cookie Cats A/B Testing & Player Retention Analysis

A data-science project analysing the impact of moving level gates in the mobile game **Cookie Cats** on 7-day player retention, following the **CRISP-DM** methodology.

## Project Overview

The goal is to determine whether moving a level gate from Level 30 to Level 40 affects player retention. The analysis includes:

- **Web Scraping** — Real BeautifulSoup scraping of 4 Wikipedia pages with sequential vs parallel performance comparison
- **Data Quality Audit** — IQR outlier detection, Z-score analysis, schema validation, and missing value checks
- **A/B Testing** — Bootstrap, Chi-square, Cohen's h effect size, and Mann-Whitney U tests
- **Data Augmentation** — Genre benchmarks from scraped industry data merged into the primary dataset
- **Machine Learning** — Binary classification pipeline (`imblearn.Pipeline`) with SMOTE for class imbalance
- **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC with metric justifications
- **Scalability Discussion** — Pandas vs Spark, time/space complexity, and big data architecture

## CRISP-DM Lifecycle

| Phase | Location |
|-------|----------|
| Business Understanding | `notebooks/final_pipeline.ipynb` Step 1 (9 subsections) |
| Data Understanding | `notebooks/04_eda_visualizations.ipynb` (16 plots) + `final_pipeline.ipynb` Steps 2–3 |
| Data Preparation | `src/processing.py` → `data_audit()`, `preprocess_data()`, `engineer_features()` |
| Web Scraping | `src/scraping.py` → `run_full_scraping_pipeline()` |
| Modelling | `src/modeling.py` → `train_models()`, `tune_hyperparameters()` |
| Evaluation | `notebooks/final_pipeline.ipynb` Steps 10–12 |
| Scalability | `notebooks/final_pipeline.ipynb` Step 13 |
| Conclusion | `notebooks/final_pipeline.ipynb` (Pipeline Complete) |

## Project Structure

```
📁 d:/Data Science/
├── 📁 data/
│   ├── 📁 raw/                     # Original untouched data
│   │   └── cookie_cats.csv
│   ├── 📁 processed/               # Cleaned & augmented data
│   │   ├── cookie_cats_clean.csv   # After preprocessing
│   │   └── cookie_cats_augmented.csv   # After scraping merge
│   └── 📁 scraped/                 # Web-scraped data
│       ├── benchmarks.json
│       └── industry_benchmarks.csv
├── 📁 notebooks/
│   ├── final_pipeline.ipynb        # ← Main reproducible pipeline (39 cells)
│   ├── 04_eda_visualizations.ipynb # ← Dedicated EDA (40 cells, 16 plots)
│   ├── 01_crisp_dm_analysis.ipynb  # CRISP-DM walkthrough
│   ├── 01_exploration.ipynb        # ⚠️ Deprecated → 04_eda
│   ├── 02_ab_testing.ipynb         # ⚠️ Deprecated → final_pipeline
│   └── 03_modeling.ipynb           # ⚠️ Deprecated → final_pipeline
├── 📁 src/
│   ├── __init__.py
│   ├── processing.py               # Load, audit, clean, features, split
│   ├── ab_testing.py               # Bootstrap + chi-square + Cohen's h + Mann-Whitney U
│   ├── modeling.py                 # imblearn Pipeline, evaluation, tuning, persistence
│   └── scraping.py                 # Real BeautifulSoup scraping (sequential + parallel)
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
| **EDA visualisations** | Open `notebooks/04_eda_visualizations.ipynb` → *Kernel → Restart & Run All* |
| **CLI quick-run (processing)** | `python src/processing.py` |
| **CLI quick-run (scraping)** | `python src/scraping.py` |
| **CLI quick-run (modelling)** | `python src/modeling.py` |

## Key Results

1. Moving the gate from level 30 → 40 **decreases** 7-day retention by ~0.8pp (statistically significant).
2. Multiple tests confirm significance: Bootstrap (p < 0.05), Chi-square (p < 0.05), with a negligible-to-small Cohen's h effect size.
3. IQR analysis reveals 11%+ outliers in `sum_gamerounds` — capped at 99th percentile for robust modelling.
4. Parallel web scraping achieves **1.85× speedup** over sequential scraping.
5. Cookie Cats retention compares favourably to industry benchmarks for match-3 games.

## License

MIT License