import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="Cookie Cats A/B Test Dashboard", page_icon="🐱", layout="wide")

# Theme setup
sns.set_theme(style="whitegrid")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cookie_cats_clean.csv")

def main():
    st.sidebar.title("🐱 Cookie Cats A/B Testing")
    st.sidebar.markdown("Explore the insights from the Gate 30 vs Gate 40 A/B test.")
    
    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    st.title("Interactive Data Science Dashboard")
    st.markdown("This dashboard presents the results of the A/B test conducted in the Cookie Cats game.")

    # KPI Metrics
    st.header("Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", f"{len(df):,}")
    with col2:
        st.metric("Avg Game Rounds", f"{df['sum_gamerounds'].mean():.2f}")
    with col3:
        st.metric("Day 1 Retention", f"{df['retention_1'].mean() * 100:.2f}%")
    with col4:
        st.metric("Day 7 Retention", f"{df['retention_7'].mean() * 100:.2f}%")

    st.divider()

    # Engagement Insights
    st.header("Engagement Insights")
    st.subheader("Distribution of Game Rounds")
    
    # Filtering for visualization
    max_rounds = st.slider("Filter Max Game Rounds for Visualization", min_value=10, max_value=500, value=100)
    
    filtered_df = df[df['sum_gamerounds'] <= max_rounds]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=filtered_df, x="sum_gamerounds", hue="version", multiple="layer", bins=30, ax=ax, alpha=0.6)
    ax.set_title(f"Game Rounds Distribution (<= {max_rounds} rounds)")
    ax.set_xlabel("Game Rounds")
    st.pyplot(fig)

    st.divider()

    # Retention A/B Testing
    st.header("Retention A/B Testing Analysis")
    
    col1, col2 = st.columns(2)
    
    retention_metrics = df.groupby('version')[['retention_1', 'retention_7']].mean().reset_index()
    
    # Provide backward compatibility for older seaborn/matplotlib
    with col1:
        st.subheader("Day 1 Retention")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=retention_metrics, x="version", y="retention_1", hue="version", ax=ax1, palette="Blues_d", legend=False)
        ax1.set_ylim(0, 0.6)
        ax1.set_ylabel("Retention Rate")
        for i, v in enumerate(retention_metrics['retention_1']):
            ax1.text(i, v + 0.01, f"{v*100:.2f}%", ha='center')
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Day 7 Retention")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=retention_metrics, x="version", y="retention_7", hue="version", ax=ax2, palette="Greens_d", legend=False)
        ax2.set_ylim(0, 0.3)
        ax2.set_ylabel("Retention Rate")
        for i, v in enumerate(retention_metrics['retention_7']):
            ax2.text(i, v + 0.01, f"{v*100:.2f}%", ha='center')
        st.pyplot(fig2)

    st.subheader("Statistical Significance")
    st.markdown("""
    To verify whether the observed differences between Gate 30 and Gate 40 are statistically significant, 
    we perform a Chi-Square test of independence.
    """)
    
    # Perform chi-square test dynamically
    contingency_table_1 = pd.crosstab(df['version'], df['retention_1'])
    chi2_1, p_1, _, _ = stats.chi2_contingency(contingency_table_1)
    
    contingency_table_7 = pd.crosstab(df['version'], df['retention_7'])
    chi2_7, p_7, _, _ = stats.chi2_contingency(contingency_table_7)
    
    # Use pandas syntax safely
    g30_r1 = retention_metrics.loc[retention_metrics['version'] == 'gate_30', 'retention_1'].values[0]
    g30_r7 = retention_metrics.loc[retention_metrics['version'] == 'gate_30', 'retention_7'].values[0]
    g40_r1 = retention_metrics.loc[retention_metrics['version'] == 'gate_40', 'retention_1'].values[0]
    g40_r7 = retention_metrics.loc[retention_metrics['version'] == 'gate_40', 'retention_7'].values[0]

    st.table(pd.DataFrame({
        "Metric": ["Day 1 Retention", "Day 7 Retention"],
        "Gate 30 Mean": [f"{g30_r1*100:.2f}%", f"{g30_r7*100:.2f}%"],
        "Gate 40 Mean": [f"{g40_r1*100:.2f}%", f"{g40_r7*100:.2f}%"],
        "Difference (Absolute)": [f"{(g40_r1 - g30_r1)*100:.3f}%", f"{(g40_r7 - g30_r7)*100:.3f}%"],
        "p-value": [f"{p_1:.4f}", f"{p_7:.4f}"],
        "Significant? (p < 0.05)": ["Yes" if p_1 < 0.05 else "No", "Yes" if p_7 < 0.05 else "No"]
    }))
    
    st.markdown("""
    **Conclusion:**
    The A/B test results show a significant difference in retention limit, particularly for Day 7 retention. Keeping the gate at level 30 has a statistically stronger impact on retaining players than moving it to level 40.
    """)

if __name__ == "__main__":
    main()
