"""
A/B Testing module for Cookie Cats project.

This module implements bootstrap analysis for comparing retention rates
between different experimental groups.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

def bootstrap_retention(data: pd.DataFrame, n_bootstraps: int = 1000, random_state: int = 42) -> List[float]:
    """
    Perform bootstrap resampling for retention rates.
    
    Args:
        data: DataFrame containing retention data
        n_bootstraps: Number of bootstrap samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        List of bootstrap retention rates
    """
    np.random.seed(random_state)
    boot_means = []
    
    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        # Sample with replacement
        sample = data.sample(frac=1, replace=True)
        # Calculate retention rate
        boot_mean = sample['retention_7'].mean()
        boot_means.append(boot_mean)
    
    return boot_means

def calculate_confidence_intervals(bootstrap_samples: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence intervals from bootstrap samples.
    
    Args:
        bootstrap_samples: List of bootstrap sample statistics
        confidence_level: Confidence level for the interval
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    lower = (1 - confidence_level) / 2
    upper = 1 - (1 - confidence_level) / 2
    
    ci_lower = np.percentile(bootstrap_samples, lower * 100)
    ci_upper = np.percentile(bootstrap_samples, upper * 100)
    
    return ci_lower, ci_upper

def calculate_p_value(bootstrap_diff: List[float], observed_diff: float) -> float:
    """
    Calculate p-value from bootstrap distribution.
    
    Args:
        bootstrap_diff: List of bootstrap differences
        observed_diff: Observed difference in retention rates
        
    Returns:
        p-value
    """
    if observed_diff > 0:
        p_value = (np.array(bootstrap_diff) >= observed_diff).mean()
    else:
        p_value = (np.array(bootstrap_diff) <= observed_diff).mean()
    
    return p_value

def analyze_ab_test(gate_30: pd.DataFrame, gate_40: pd.DataFrame, 
                    n_bootstraps: int = 1000, confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Perform complete A/B test analysis using bootstrap.
    
    Args:
        gate_30: DataFrame for gate_30 group
        gate_40: DataFrame for gate_40 group
        n_bootstraps: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary containing analysis results
    """
    # Calculate observed retention rates
    retention_30 = gate_30['retention_7'].mean()
    retention_40 = gate_40['retention_7'].mean()
    observed_diff = retention_40 - retention_30
    
    # Perform bootstrap resampling
    boot_30 = bootstrap_retention(gate_30, n_bootstraps)
    boot_40 = bootstrap_retention(gate_40, n_bootstraps)
    boot_diff = np.array(boot_40) - np.array(boot_30)
    
    # Calculate confidence intervals
    ci_30 = calculate_confidence_intervals(boot_30, confidence_level)
    ci_40 = calculate_confidence_intervals(boot_40, confidence_level)
    ci_diff = calculate_confidence_intervals(boot_diff, confidence_level)
    
    # Calculate p-value
    p_value = calculate_p_value(boot_diff, observed_diff)
    
    # Return analysis results
    return {
        'observed_retention': {
            'gate_30': retention_30,
            'gate_40': retention_40,
            'difference': observed_diff
        },
        'bootstrap_samples': {
            'gate_30': boot_30,
            'gate_40': boot_40,
            'difference': boot_diff.tolist()
        },
        'confidence_intervals': {
            'gate_30': ci_30,
            'gate_40': ci_40,
            'difference': ci_diff,
            'confidence_level': confidence_level
        },
        'p_value': p_value,
        'sample_sizes': {
            'gate_30': len(gate_30),
            'gate_40': len(gate_40)
        }
    }

def plot_bootstrap_results(analysis_results: Dict[str, Any], save_path: str = None) -> None:
    """
    Create visualizations for bootstrap analysis results.
    
    Args:
        analysis_results: Dictionary containing analysis results
        save_path: Path to save the plot (optional)
    """
    # Set up the figure
    plt.figure(figsize=(18, 5))
    
    # Extract results
    boot_30 = analysis_results['bootstrap_samples']['gate_30']
    boot_40 = analysis_results['bootstrap_samples']['gate_40']
    boot_diff = analysis_results['bootstrap_samples']['difference']
    
    retention_30 = analysis_results['observed_retention']['gate_30']
    retention_40 = analysis_results['observed_retention']['gate_40']
    observed_diff = analysis_results['observed_retention']['difference']
    
    # Gate 30 distribution
    plt.subplot(1, 3, 1)
    sns.histplot(boot_30, bins=30, kde=True)
    plt.axvline(retention_30, color='red', linestyle='--', label='Observed')
    plt.title('Gate 30 Retention Bootstrap')
    plt.xlabel('7-day Retention Rate')
    plt.ylabel('Frequency')
    
    # Gate 40 distribution
    plt.subplot(1, 3, 2)
    sns.histplot(boot_40, bins=30, kde=True)
    plt.axvline(retention_40, color='red', linestyle='--', label='Observed')
    plt.title('Gate 40 Retention Bootstrap')
    plt.xlabel('7-day Retention Rate')
    
    # Difference distribution
    plt.subplot(1, 3, 3)
    sns.histplot(boot_diff, bins=30, kde=True)
    plt.axvline(observed_diff, color='red', linestyle='--', label='Observed')
    plt.axvline(0, color='black', linestyle='-', label='No Difference')
    plt.title('Retention Difference Bootstrap')
    plt.xlabel('Difference in 7-day Retention')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_retention_comparison(analysis_results: Dict[str, Any], save_path: str = None) -> None:
    """
    Create bar plot comparing retention rates with confidence intervals.
    
    Args:
        analysis_results: Dictionary containing analysis results
        save_path: Path to save the plot (optional)
    """
    # Extract results
    retention_30 = analysis_results['observed_retention']['gate_30']
    retention_40 = analysis_results['observed_retention']['gate_40']
    ci_30 = analysis_results['confidence_intervals']['gate_30']
    ci_40 = analysis_results['confidence_intervals']['gate_40']
    
    plt.figure(figsize=(10, 6))
    
    # Plot observed retention rates
    plt.bar(['gate_30', 'gate_40'], [retention_30, retention_40],
            yerr=[[retention_30 - ci_30[0], retention_40 - ci_40[0]],
                  [ci_30[1] - retention_30, ci_40[1] - retention_40]],
            capsize=10, color=['blue', 'orange'], alpha=0.7)
    
    plt.title('7-day Retention Rates with 95% Confidence Intervals')
    plt.ylabel('7-day Retention Rate')
    plt.ylim(0, max(retention_30, retention_40) * 1.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

# Example usage
if __name__ == "__main__":
    from processing import load_data, preprocess_data, create_ab_groups
    
    # Load and prepare data
    try:
        df = load_data()
        df_clean = preprocess_data(df)
        gate_30, gate_40 = create_ab_groups(df_clean)
        
        # Perform A/B test analysis
        results = analyze_ab_test(gate_30, gate_40)
        
        # Print results
        print("\nA/B Test Results:")
        print(f"Gate 30 retention: {results['observed_retention']['gate_30']:.4f}")
        print(f"Gate 40 retention: {results['observed_retention']['gate_40']:.4f}")
        print(f"Observed difference: {results['observed_retention']['difference']:.4f}")
        print(f"95% CI for difference: [{results['confidence_intervals']['difference'][0]:.4f}, {results['confidence_intervals']['difference'][1]:.4f}]")
        print(f"p-value: {results['p_value']:.4f}")
        
        # Create visualizations
        plot_bootstrap_results(results)
        plot_retention_comparison(results)
        
    except FileNotFoundError as e:
        print(e)