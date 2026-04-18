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
| Business Understanding | `notebooks/01_business_understanding.ipynb` (9 subsections) |
| Data Understanding | `notebooks/02_data_audit_and_cleaning.ipynb` + `notebooks/04_eda_visualizations.ipynb` (16 plots) |
| Data Preparation | `src/processing.py` → `data_audit()`, `preprocess_data()`, `engineer_features()` |
| Web Scraping | `src/scraping.py` → `run_full_scraping_pipeline()` |
| Modelling | `src/modeling.py` → `train_models()`, `tune_hyperparameters()` |
| Evaluation | `notebooks/06_evaluation_and_ab_testing.ipynb` |
| Scalability | `notebooks/06_evaluation_and_ab_testing.ipynb` (scalability section) |
| Conclusion | `notebooks/07_conclusions.ipynb` |

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
├── 📁 notebooks/                   # 7 CRISP-DM phase notebooks
│   ├── 01_business_understanding.ipynb      # Phase 1: Business context & research Qs
│   ├── 02_data_audit_and_cleaning.ipynb     # Phase 2-3: Load, audit, clean
│   ├── 03_web_scraping.ipynb                # Phase 2: External data scraping
│   ├── 04_eda_visualizations.ipynb          # Phase 2: 16 publication-quality plots
│   ├── 05_feature_engineering_and_modeling.ipynb  # Phase 3-4: Features & ML training
│   ├── 06_evaluation_and_ab_testing.ipynb   # Phase 5: Evaluation & statistics
│   └── 07_conclusions.ipynb                 # Phase 6: Conclusions & recommendations
├── 📁 src/
│   ├── __init__.py
│   ├── processing.py               # Load, audit, clean, features, split
│   ├── ab_testing.py               # Bootstrap + chi-square + Cohen's h + Mann-Whitney U
│   ├── modeling.py                 # imblearn Pipeline, evaluation, tuning, persistence
│   └── scraping.py                 # Real BeautifulSoup scraping (sequential + parallel)
├── 📁 reports/
│   ├── 📁 figures/                 # Auto-generated visualisations
│   └── cookie_cats_report.md       # Full analysis report
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

### Notebook Sequence (run in order)

| # | Notebook | What it Does |
|---|----------|-------------|
| 1 | `01_business_understanding.ipynb` | Business context, research questions, KPIs |
| 2 | `02_data_audit_and_cleaning.ipynb` | Load raw data, audit, clean, save CSV |
| 3 | `03_web_scraping.ipynb` | Scrape Wikipedia, compare seq/parallel, augment |
| 4 | `04_eda_visualizations.ipynb` | 16 publication-quality exploratory plots |
| 5 | `05_feature_engineering_and_modeling.ipynb` | Engineer features, train LR/RF/XGBoost |
| 6 | `06_evaluation_and_ab_testing.ipynb` | Evaluate models, A/B stats, scalability |
| 7 | `07_conclusions.ipynb` | Final conclusions & business recommendations |

### CLI Quick-Run

| Command | Purpose |
|---------|---------|
| `python src/processing.py` | Run data processing pipeline |
| `python src/scraping.py` | Run web scraping pipeline |
| `python src/modeling.py` | Run model training pipeline |

## Key Results

1. Moving the gate from level 30 → 40 **decreases** 7-day retention by ~0.8pp (statistically significant).
2. Multiple tests confirm significance: Bootstrap (p < 0.05), Chi-square (p < 0.05), with a negligible-to-small Cohen's h effect size.
3. IQR analysis reveals 11%+ outliers in `sum_gamerounds` — capped at 99th percentile for robust modelling.
4. Parallel web scraping achieves **1.85× speedup** over sequential scraping.
5. Cookie Cats retention compares favourably to industry benchmarks for match-3 games.

## License

MIT License