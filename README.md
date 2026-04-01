# Cookie Cats A/B Testing & Player Retention Analysis

A data-science project analysing the impact of moving level gates in the mobile game **Cookie Cats** on 7-day player retention, following the **CRISP-DM** methodology.

## Project Overview

The goal is to determine whether moving a level gate from Level 30 to Level 40 affects player retention. The analysis includes:

- **A/B Testing** — Bootstrap analysis to compare retention rates between groups
- **Data Augmentation** — Web scraping of gaming-industry retention benchmarks
- **Machine Learning** — Binary classification pipeline (`sklearn.Pipeline`) for predicting player churn
- **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC

## CRISP-DM Lifecycle

| Phase | Location |
|-------|----------|
| Business Understanding | `notebooks/01_crisp_dm_analysis.ipynb` §1 |
| Data Understanding | `notebooks/01_crisp_dm_analysis.ipynb` §2 |
| Data Preparation | `notebooks/01_crisp_dm_analysis.ipynb` §3 + `src/processing.py` |
| Modelling | `notebooks/01_crisp_dm_analysis.ipynb` §4 + `src/modeling.py` |
| Evaluation | `notebooks/01_crisp_dm_analysis.ipynb` §5 |
| Conclusion | `notebooks/01_crisp_dm_analysis.ipynb` §6 |

## Project Structure

```
📁 d:/Data Science/
├── 📁 data/                    # Raw and processed datasets
│   └── cookie_cats.csv
├── 📁 notebooks/               # Jupyter notebooks
│   ├── 01_crisp_dm_analysis.ipynb   # ← Main CRISP-DM analysis
│   ├── final_pipeline.ipynb         # ← Reproducible end-to-end pipeline
│   ├── 01_exploration.ipynb         # Legacy exploration
│   ├── 02_ab_testing.ipynb          # Legacy A/B testing
│   └── 03_modeling.ipynb            # Legacy modelling
├── 📁 src/                     # Modular Python scripts
│   ├── __init__.py
│   ├── processing.py           # Data loading, cleaning, feature engineering
│   ├── ab_testing.py           # Bootstrap analysis functions
│   ├── modeling.py             # sklearn Pipeline, evaluation, tuning
│   └── scraping.py             # Industry benchmark scraping
├── 📁 reports/                 # Visualisations and final report
│   └── 📁 figures/
├── 📄 requirements.txt         # Python dependencies
└── 📄 README.md                # This file
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
   - Place the CSV in `data/cookie_cats.csv`

## Usage

| Step | What to Run |
|------|-------------|
| **Full CRISP-DM analysis** | Open `notebooks/01_crisp_dm_analysis.ipynb` |
| **Reproducible pipeline** | Open `notebooks/final_pipeline.ipynb` → *Kernel → Restart & Run All* |
| **CLI quick-run** | `python src/modeling.py` |

## Key Results

1. Moving the gate from level 30 → 40 **decreases** 7-day retention (statistically significant).
2. The best ML model achieves ROC-AUC above the random baseline.
3. Cookie Cats retention compares favourably to industry benchmarks for match-3 games.

## License

MIT License