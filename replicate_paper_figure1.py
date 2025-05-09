#!/usr/bin/env python
"""
Script to replicate Figure 1 from Lewellen and Lewellen (2016) in the exact format shown in the paper.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from datetime import datetime

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules with correct paths
try:
    from utils import winsorize
except ModuleNotFoundError:
    from src.utils import winsorize
from src.variable_construction import construct_variables
from src.sample_preparation import prepare_regression_sample

def generate_paper_figure1():
    """
    Generate Figure 1: Cross-Sectional Distribution of Cash Flow Measures exactly as in the paper.
    """
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw_merged_data.parquet')
    print(f'Loading data from {data_path}')
    data = pd.read_parquet(data_path)
    print(f'Loaded {len(data)} rows')
    
    # Process data
    print('Constructing variables...')
    constructed_data = construct_variables(data)
    print(f'Constructed variables for {len(constructed_data)} rows')
    del data
    
    print('Preparing regression sample...')
    regression_sample = prepare_regression_sample(constructed_data)
    print(f'Prepared regression sample with {len(regression_sample)} rows')
    del constructed_data
    
    # Filter to sample period 1971-2009 inclusive, as in Lewellen & Lewellen (2016)
    regression_sample = regression_sample[(regression_sample['fyear'] >= 1971) & (regression_sample['fyear'] <= 2009)]
    print(f'Filtered regression sample to fiscal years 1971-2009 inclusive: {len(regression_sample)} rows')
    
    # Create directory for plots
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Generating Paper-Style Figure 1: Cross-Sectional Distribution of Cash Flow Measures")
    
    # We'll need to compute cash flow measures to match the paper
    # CF = PROF + DEPR + OTHCF (as per the paper)
    # We'll use cash_flow_scaled as our CF
    # We'll create a proxy for PROF+DEPR as the "traditional" cash flow measure
    regression_sample['cf'] = regression_sample['cash_flow_scaled']
    regression_sample['prof_plus_depr'] = regression_sample['trad_cash_flow_scaled']
    
    # Group by year and compute statistics
    yearly_stats = regression_sample.groupby('fyear').agg({
        'cf': ['mean', 'std'],
        'prof_plus_depr': ['mean', 'std']
    })
    
    # Calculate correlation by year
    correlations = []
    for year, group in regression_sample.groupby('fyear'):
        if len(group) > 1:  # Need at least 2 points for correlation
            corr = group['cf'].corr(group['prof_plus_depr'])
            correlations.append((year, corr))
    
    correlation_df = pd.DataFrame(correlations, columns=['fyear', 'correlation'])
    
    # Setup figure with three panels (subplots)
    plt.figure(figsize=(10, 7))
    plt.suptitle("FIGURE 1", fontsize=14, y=0.98)
    plt.figtext(0.5, 0.93, "Cross-Sectional Distribution of Cash Flow Measures", 
                ha='center', fontsize=12)
    
    # Add the description text from the paper
    description = (
        "Figure 1 plots the annual cross-sectional mean, standard deviation, and correlation of cash flow (CF) and income before "
        "extraordinary items plus depreciation (PROF+DEPR). The variables are scaled by average net assets during the year "
        "and winsorized at their 1st and 99th percentiles. Data come from Compustat. The sample consists of all nonfinancial "
        "firms that are larger than the NYSE 10th percentile (measured by net assets at the beginning of the year)."
    )
    plt.figtext(0.5, 0.85, description, wrap=True, ha='center', va='top', fontsize=8)
    
    # Create a 1x3 grid of subplots
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # Panel A: Mean
    ax1 = plt.subplot(gs[0])
    ax1.plot(yearly_stats.index, yearly_stats[('cf', 'mean')], 'k-', linewidth=1.5, label='CF')
    ax1.plot(yearly_stats.index, yearly_stats[('prof_plus_depr', 'mean')], 'gray', linewidth=1.5, label='PROF+DEPR')
    ax1.set_xlim(yearly_stats.index.min(), yearly_stats.index.max())
    # ax1.set_ylim(0, 0.18)
    ax1.set_xticks([1971, 1980, 1989, 1998, 2007])
    # ax1.set_yticks([0.00, 0.04, 0.08, 0.12, 0.16])
    ax1.set_title('Graph A. Mean', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Std
    ax2 = plt.subplot(gs[1])
    ax2.plot(yearly_stats.index, yearly_stats[('cf', 'std')], 'k-', linewidth=1.5, label='CF')
    ax2.plot(yearly_stats.index, yearly_stats[('prof_plus_depr', 'std')], 'gray', linewidth=1.5, label='PROF+DEPR')
    ax2.set_xlim(yearly_stats.index.min(), yearly_stats.index.max())
    # ax2.set_ylim(0.04, 0.24)
    ax2.set_xticks([1971, 1980, 1989, 1998, 2007])
    # ax2.set_yticks([0.04, 0.09, 0.14, 0.19, 0.24])
    ax2.set_title('Graph B. Std', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Correlation
    ax3 = plt.subplot(gs[2])
    ax3.plot(correlation_df['fyear'], correlation_df['correlation'], 'k-', linewidth=1.5)
    ax3.set_xlim(yearly_stats.index.min(), yearly_stats.index.max())
    # ax3.set_ylim(0.6, 1.0)
    ax3.set_xticks([1971, 1980, 1989, 1998, 2007])
    # ax3.set_yticks([0.60, 0.70, 0.80, 0.90, 1.00])
    ax3.set_title('Graph C. Correlation', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.83])
    
    # Save figure
    figure_path = os.path.join(plots_dir, 'paper_figure1_cash_flow_distribution.png')
    plt.savefig(figure_path, dpi=300)
    plt.close()
    
    print(f"Paper-style Figure 1 saved to {figure_path}")
    
    # Save the data as CSV for reference
    yearly_stats.to_csv(os.path.join(plots_dir, 'paper_figure1_data.csv'))
    correlation_df.to_csv(os.path.join(plots_dir, 'paper_figure1_correlation_data.csv'), index=False)
    
    print("Data for Paper Figure 1 saved as CSV files")
    
    return regression_sample

if __name__ == "__main__":
    generate_paper_figure1()