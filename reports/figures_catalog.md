# Cookie Cats Project — Figure Catalog

This document provides a descriptive catalog of all visual assets generated during the project. These figures are located in the `reports/figures/` directory and map to various stages of the CRISP-DM methodology, from Exploratory Data Analysis (EDA) to statistical evaluation and machine learning model comparison.

## 1. Exploratory Data Analysis (EDA) & Data Quality

| File Name | Description | Key Insight |
| :--- | :--- | :--- |
| `eda_01_gamerounds_dist.png` | Histogram showing the distribution of the `sum_gamerounds` feature. | Highlights the extreme right skew in the raw data, with most players playing very few rounds. |
| `eda_02_group_sizes.png` | Bar chart displaying player counts for control and treatment. | Confirms a near-equal, balanced split of users between `gate_30` (44,700) and `gate_40` (45,489) groups. |
| `eda_03_retention_overview.png` | Grouped bar chart of baseline Day 1 and Day 7 retention across the dataset. | Sets the baseline averages: ~45% Day 1 retention and ~19% Day 7 retention. |
| `eda_04_retention_by_version.png` | Comparison of retention rates split by A/B group assignment. | Crucial visual showing the drop in both D1 and D7 retention for the Gate 40 group. |
| `eda_05_rounds_by_version.png` | Boxplots comparing the distribution of game rounds for each gate. | Shows that despite the retention drop, the distribution of rounds played remains relatively similar between groups. |
| `eda_06_rounds_vs_retention.png` | Scatter or line plot visualizing game rounds played vs. retention probability. | Confirms the positive correlation: higher engagement (rounds) strongly correlates with higher retention probability. |
| `eda_07_outlier_capping.png` | Side-by-side distribution plots of `sum_gamerounds` before and after capping at the 99th percentile (493). | Demonstrates the effective mitigation of extreme outlier values (originally up to 49k rounds) for modeling. |
| `eda_08_iqr_boundaries.png` | Visualization highlighting the data distribution against the calculated Interquartile Range (IQR) fences. | Shows the large volume of statistical "outliers" (11.28% of data falls outside the Q3+1.5*IQR bounds limit). |
| `eda_09_engagement_tiers.png` | Bar chart of the categorized `gamerounds_bin` feature (inactive, casual, moderate, active, hardcore). | Reveals that the vast majority of the player base falls into the "casual" tier (1-20 rounds). |
| `eda_10_correlation_heatmap.png` | Matrix showing Pearson correlation coefficients between numeric features and target variables. | Indicates that `retention_1` is the single strongest predictor for `retention_7`. |
| `eda_11_interaction_feature.png` | Analysis plot showing the distinct distributions for the engineered `retention_1_x_rounds` interaction term. | Shows how day-1 retention combined with play volume yields a strong combined signal. |
| `eda_12_retention_funnel.png` | Funnel-style chart illustrating player drop-off over time. | Visualizes the steep ~26 percentage point drop-off from Day 1 to Day 7. |
| `eda_13_transition_matrix.png` | Matrix expressing state transition probabilities from Day 1 state to Day 7 state. | Highlights that players retained on D1 are multiple times more likely to be retained on D7. |
| `eda_14_retention_by_quantile.png` | Bar chart breaking down retention performance by player engagement quantiles. | Confirms that top-quartile highly engaged players boast the highest overall retention figures. |
| `eda_15_zero_round_players.png` | Focus plot profiling the players who recorded exactly 0 game rounds. | Assesses the "install-but-never-play" cohort which makes up ~4.4% of the dataset. |
| `eda_16_three_way_analysis.png` | Multi-dimensional faceted analysis looking at Gate placement × Engagement level × Retention behavior. | Isolates how different engagement tiers uniquely react to the gate placement change. |

## 2. A/B Testing & Statistical Evaluation

| File Name | Description | Key Insight |
| :--- | :--- | :--- |
| `bootstrap_results.png` | Visualizes the 95% confidence intervals from a 1,000-resample bootstrap test measuring differences in retention. | Provides visual, statistical proof that the true mean difference in Day 7 retention lies below zero (favoring Gate 30). |
| `retention_comparison.png` | A streamlined, presentation-ready bar plot contrasting the retention percentages of the two groups. | The primary, clear-cut visual summarizing the overarching finding of the A/B experiment. |

## 3. Machine Learning Modeling

| File Name | Description | Key Insight |
| :--- | :--- | :--- |
| `model_comparison.png` | Bar chart evaluating trained classifiers based on primary classification metrics (ROC-AUC, F1-Score). | Identifies XGBoost as the top-performing model for predicting Day 7 retention behavior. |
| `roc_curves.png` | Receiver Operating Characteristic (ROC) plots for the evaluated models. | Visual proof of the models' discrimination ability (True Positives vs False Positives) across all threshold levels. |
| `confusion_matrices.png` | Grid of confusion matrices (heatmaps) plotting True/False Positives and Negatives for the various models. | Details how models (trained via SMOTE class-balancing) handle the tradeoff between precision and recall on the minority class. |

## 4. Multi-Panel Summaries (Ideal for Posters/Presentations)

| File Name | Description | Suggested Use |
| :--- | :--- | :--- |
| `eda_overview.png` | A multi-panel compilation (likely 4-panel) of the most important Exploratory Data Analysis findings. | **Highly recommended as the centerpiece visual block for your A3 poster** to summarize the dataset at a glance. |
| `feature_eda.png` | A grid format subplot providing an overview of the distributions of the engineered modeling features. | Good for technical reports to showcase the data preparation phase outputs prior to machine learning. |
