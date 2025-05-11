#!/usr/bin/env python
"""
Script to replicate Figure 1 from Lewellen and Lewellen (2016).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules with correct paths
from replicate_table1 import generate_table1

def generate_figure1(regression_sample=None):
    """
    Generate Figure 1: Investment and Cash Flow by Year.
    
    Args:
        regression_sample: Optional pre-loaded regression sample. If None, it will be generated.
    """
    # Load regression sample if not provided
    if regression_sample is None:
        print("Generating regression sample...")
        regression_sample = generate_table1()
    
    # Create directory for plots
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Generating Figure 1: Investment and Cash Flow by Year")
    
    # Group by year and compute median values
    median_by_year = regression_sample.groupby('fyear').agg({
        'cash_flow_scaled': 'median',
        'capx1_scaled': 'median',
        'capx2_scaled': 'median',
        'capx3_scaled': 'median'
    }).reset_index()
    
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Plot time series
    plt.plot(median_by_year['fyear'], median_by_year['cash_flow_scaled'], 'b-', 
             linewidth=2, label='Cash Flow')
    plt.plot(median_by_year['fyear'], median_by_year['capx1_scaled'], 'r--', 
             linewidth=2, label='CAPX1 (Capital Expenditures)')
    plt.plot(median_by_year['fyear'], median_by_year['capx2_scaled'], 'g-.', 
             linewidth=2, label='CAPX2 (CAPX + Acquisitions)')
    plt.plot(median_by_year['fyear'], median_by_year['capx3_scaled'], 'm:', 
             linewidth=2, label='CAPX3 (Total Investment)')
    
    # Add labels and title
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Median Value (Scaled by Net Assets)', fontsize=12)
    plt.title('Figure 1: Median Investment and Cash Flow by Year', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Adjust x-axis to show years at reasonable intervals
    plt.xticks(
        np.arange(
            median_by_year['fyear'].min(),
            median_by_year['fyear'].max() + 5,
            5
        ),
        rotation=45
    )
    
    # Save figure
    plt.tight_layout()
    figure_path = os.path.join(plots_dir, 'figure1_investment_cash_flow_by_year.png')
    plt.savefig(figure_path, dpi=300)
    plt.close()
    
    print(f"Figure 1 saved to {figure_path}")
    
    # Also create a separate figure showing mean values
    print("Generating Figure 1 (alternative): Mean Investment and Cash Flow by Year")
    
    # Group by year and compute mean values
    mean_by_year = regression_sample.groupby('fyear').agg({
        'cash_flow_scaled': 'mean',
        'capx1_scaled': 'mean',
        'capx2_scaled': 'mean',
        'capx3_scaled': 'mean'
    }).reset_index()
    
    # Plot time series with mean values
    plt.figure(figsize=(12, 8))
    plt.plot(mean_by_year['fyear'], mean_by_year['cash_flow_scaled'], 'b-', 
             linewidth=2, label='Cash Flow')
    plt.plot(mean_by_year['fyear'], mean_by_year['capx1_scaled'], 'r--', 
             linewidth=2, label='CAPX1 (Capital Expenditures)')
    plt.plot(mean_by_year['fyear'], mean_by_year['capx2_scaled'], 'g-.', 
             linewidth=2, label='CAPX2 (CAPX + Acquisitions)')
    plt.plot(mean_by_year['fyear'], mean_by_year['capx3_scaled'], 'm:', 
             linewidth=2, label='CAPX3 (Total Investment)')
    
    # Add labels and title
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Mean Value (Scaled by Net Assets)', fontsize=12)
    plt.title('Figure 1 (Alternative): Mean Investment and Cash Flow by Year', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Adjust x-axis to show years at reasonable intervals
    plt.xticks(
        np.arange(
            mean_by_year['fyear'].min(),
            mean_by_year['fyear'].max() + 5,
            5
        ),
        rotation=45
    )
    
    # Save alternative figure
    plt.tight_layout()
    alt_figure_path = os.path.join(plots_dir, 'figure1_alt_mean_investment_cash_flow_by_year.png')
    plt.savefig(alt_figure_path, dpi=300)
    plt.close()
    
    print(f"Alternative Figure 1 saved to {alt_figure_path}")
    
    # Save the data as CSV for reference
    median_by_year.to_csv(os.path.join(plots_dir, 'figure1_data_median.csv'), index=False)
    mean_by_year.to_csv(os.path.join(plots_dir, 'figure1_data_mean.csv'), index=False)
    
    print("Data for Figure 1 saved as CSV files")
    
    return regression_sample

if __name__ == "__main__":
    generate_figure1()