# Viva & Defense Preparation Guide

This document is designed to help you prepare for the **Repo Defense (Q&A)** and **Report Defense (Viva)** portions of your assignment. The marking criteria states that **First Tier (Exemplary)** students must be able to "articulate the business value of their findings", "defend their ethical analysis", and "justify architectural choices (e.g., 'Why a Class?')."

Below are the key technical and methodological decisions made in this project, along with the justifications you should use if asked.

---

## 1. Data Cleaning & Outlier Handling

**Question: Why did you cap the outliers at the 99th percentile instead of deleting them or using a logarithmic transformation?**
*   **Justification:** Deleting them would mean losing data on our most "hardcore" (and likely most profitable) players. A log transformation squashes the scale but doesn't handle the extreme skew as cleanly for some of our distance-based ML models. By capping at the 99th percentile (493 rounds), we preserve the signal that these players are highly engaged without letting a single anomalous value (like the player with 49,854 rounds) completely distort our mean averages and our model weights.

**Question: Why did you use the Interquartile Range (IQR) to detect anomalies initially?**
*   **Justification:** The game rounds feature, like most behavioral data, is extremely right-skewed (not normally distributed). Using Z-scores assumes a normal (Gaussian) distribution, which would give us wildly inaccurate outlier bounds. IQR is robust to non-normal distributions because it relies on percentiles/medians rather than means and standard deviations.

---

## 2. Machine Learning & Class Imbalance

**Question: Why did you use SMOTE?**
*   **Justification:** The dataset suffers from severe class imbalance regarding Day 7 retention: roughly 81% of players churn (Class 0) and only 19% are retained (Class 1). If we train a model on this raw data, the model can achieve 81% accuracy simply by guessing "Churn" every single time, learning nothing. SMOTE (Synthetic Minority Oversampling Technique) solves this by artificially synthesizing new, plausible examples of the minority class (retained players) by interpolating between existing data points.

**Question: Why is SMOTE applied inside an `imblearn.Pipeline` rather than just on the raw data?**
*   **Justification:** This prevents **Data Leakage**. If we apply SMOTE *before* cross-validation splitting, synthetic data generated from the test set leaks into the training set, giving us artificially high (and fake) performance scores. Putting it inside the pipeline guarantees that oversampling only happens on the *training* fold during each validation step.

**Question: Why did you prioritize ROC-AUC and F1-Score instead of Accuracy?**
*   **Justification:** Because of the 81/19 class imbalance. Accuracy is a deceptive metric here. F1-Score takes the harmonic mean of Precision and Recall, proving the model is actually capturing the minority class. ROC-AUC evaluates the model's ability to distinguish between the two classes across *all* probability thresholds, making it the most robust metric for comparing the raw discriminatory power of different algorithms like Random Forest and XGBoost.

---

## 3. Statistical & A/B Testing

**Question: What is a Bootstrap test and why did you use it instead of just a T-test?**
*   **Justification:** A Bootstrap test is a non-parametric resampling technique. Since our data (game rounds and retention) is not normally distributed, parametric tests like the Student's T-test can be weak or invalid. Bootstrapping involves drawing thousands of random samples from our data, with replacement, to build an empirical distribution of the means. This allows us to calculate highly accurate 95% confidence intervals without relying on strict statistical assumptions.

**Question: You used a Chi-Square test for the retention rates. Why?**
*   **Justification:** Retention is a categorical, categorical variable (True/False vs Gate_30/Gate_40). Chi-Square is the standard statistical test for checking if there is a significant association between two categorical variables. 

**Question: Why include Cohen's *h*?**
*   **Justification:** Statistical significance (p-value) only tells us *if* an effect exists; it does not tell us how *large* or important the effect is. Because our sample size is massive (90,000+ players), even a tiny, meaningless difference will register as statistically significant ($p < 0.05$). Cohen's *h* measures the **effect size**. In our case, $|h| \approx 0.02$, showing that while the drop in retention is real, the practical magnitude of the effect is almost negligible.

---

## 4. Software Engineering & Scalability

**Question: Why did you separate the code into `.py` scripts (`src/`) and notebooks (`notebooks/`)?**
*   **Justification:** Notebooks are excellent for EDA, visualizations, and narrative flow, but they are terrible for software engineering (version control is messy, code is hard to reuse, state becomes tangled). Moving core logic (like scraping, cleaning, and model training) into Python modules (`src/`) makes the code modular, reusable, testable, and compliant with PEP8 software engineering standards. This is a primary requirement for the "Exemplary" tier of standard project frameworks.

**Question: How did you implement parallel scraping, and why?**
*   **Justification:** I utilized Python's `concurrent.futures.ThreadPoolExecutor`. Web scraping is inherently an I/O bound task (most of the time is spent waiting for the network to respond). By using multiple threads, we can send requests to all 4 Wikipedia pages concurrently rather than waiting for one to finish before starting the next. This reduced our script execution time by roughly 2.6x.

**Question: Discuss the Big Data scalability of your pipeline. How would you handle this if Cookie Cats had 500 million players?**
*   **Justification:** Currently, we use Pandas and Scikit-learn. These operate in-memory on a single node (Time/Space complexity is roughly $\mathcal{O}(n)$ bound to RAM). If the data scaled to hundreds of gigabytes, Pandas would crash with an OutOfMemory error. To scale, we would migrate the pipeline to Apache Spark (PySpark) or Dask. This transitions the architecture to a distributed cluster system where data and computations are partitioned across multiple worker nodes, changing the complexity to $\mathcal{O}(n/p)$, where $p$ is the number of partitions/workers.

---

## 5. Ethics & Business

**Question: What are the ethical implications of this project?**
*   **Justification:** From a data perspective, the dataset is safe as it is fully anonymized—`userid` cannot be tied to Personally Identifiable Information (PII). From a methodology standpoint, our ML model prevents algorithmic bias by not including any demographic data (race, gender, location). From a business ethics perspective, A/B testing "waiting gates" borders on *dark patterns*—we are manipulating player psychology (frustration vs anticipation) to maximize engagement and monetization. We must ensure the game remains enjoyable and fair, rather than purely exploitative.

**Question: What is your final business recommendation and why?**
*   **Justification:** Keep the gate at Level 30. The data provides statistically significant proof that pushing the gate back to Level 40 decreases 7-day retention by roughly 0.82 percentage points. While 0.8% seems small, at the scale of millions of users, that translates to massive losses in total daily active users (DAUs) and long-term ad/in-app purchase revenue. Furthermore, our scraping benchmarks show that an ~19% D7 retention rate is highly competitive within the Match-3 genre.
