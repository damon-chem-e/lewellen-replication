#!/usr/bin/env python
"""
Script to replicate Table 1 from Lewellen and Lewellen (2016).
"""

import os
import sys
import pandas as pd
import numpy as np
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

def generate_table1():
    """
    Generate Table 1: Sample characteristics and summary statistics.
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
    
    # Create directory for tables
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    # Panel A: Sample Characteristics by Year
    print("Generating Panel A: Sample Characteristics by Year")
    
    # Group by year and compute statistics
    year_stats = regression_sample.groupby('fyear').agg(
        num_firms=('gvkey', 'nunique'),
        total_assets=('net_assets', 'sum'),
        avg_assets=('net_assets', 'mean'),
        median_assets=('net_assets', 'median'),
        total_capx=('capx1', 'sum'),
        avg_capx=('capx1', 'mean'),
        median_capx=('capx1', 'median')
    )
    
    # Convert to millions for easier reading
    for col in ['total_assets', 'avg_assets', 'median_assets', 'total_capx', 'avg_capx', 'median_capx']:
        year_stats[col] = year_stats[col] / 1e6
    
    # Select decades for display (similar to the paper)
    decades = sorted([year for year in year_stats.index if year % 10 == 0])
    if 2010 not in decades and 2010 in year_stats.index:
        decades.append(2010)
    if year_stats.index.max() not in decades:
        decades.append(year_stats.index.max())
    
    panel_a = year_stats.loc[decades].copy()
    panel_a.index.name = 'Year'
    panel_a.columns = ['Number of Firms', 'Total Assets ($M)', 'Average Assets ($M)', 
                       'Median Assets ($M)', 'Total CAPX ($M)', 'Average CAPX ($M)', 
                       'Median CAPX ($M)']
    
    # Save Panel A
    panel_a.to_csv(os.path.join(tables_dir, 'table1_panel_a.csv'))
    
    # Panel B: Summary Statistics for Key Variables
    print("Generating Panel B: Summary Statistics for Key Variables")
    
    # Select key variables
    key_vars = [
        'cash_flow_scaled', 'capx1_scaled', 'capx2_scaled', 'capx3_scaled',
        'delta_cash_scaled', 'delta_nwc_scaled', 'delta_debt_scaled', 
        'issues_scaled', 'div_scaled', 'mb_lag',
        'cash_lag', 'debt_lag'
    ]
    
    # Compute mean, median, standard deviation, min, max, and quantiles
    panel_b = pd.DataFrame({
        'Mean': regression_sample[key_vars].mean(),
        'Median': regression_sample[key_vars].median(),
        'Std Dev': regression_sample[key_vars].std(),
        'Min': regression_sample[key_vars].min(),
        'P25': regression_sample[key_vars].quantile(0.25),
        'P75': regression_sample[key_vars].quantile(0.75),
        'Max': regression_sample[key_vars].max()
    })
    
    # Rename rows for better readability
    panel_b.index = [
        'Cash Flow', 'CAPX1 (Capital Expenditures)', 'CAPX2 (CAPX + Acquisitions)', 
        'CAPX3 (Total Investment)', 'Change in Cash', 'Change in Net Working Capital',
        'Change in Debt', 'Equity Issuance', 'Dividends', 'Market-to-Book',
        'Cash Holdings (t-1)', 'Debt (t-1)'
    ]
    
    # Save Panel B
    panel_b.to_csv(os.path.join(tables_dir, 'table1_panel_b.csv'))
    
    # Panel C: Correlation Matrix
    print("Generating Panel C: Correlation Matrix")
    
    # Compute correlation matrix for key variables
    panel_c = regression_sample[key_vars].corr()
    
    # Use same variable names as Panel B
    panel_c.index = panel_b.index
    panel_c.columns = panel_b.index
    
    # Save Panel C
    panel_c.to_csv(os.path.join(tables_dir, 'table1_panel_c.csv'))
    
    # Create full Table 1 with all panels
    print("Creating full Table 1")
    
    # Create HTML file with all panels
    with open(os.path.join(tables_dir, 'table1_full.html'), 'w') as f:
        f.write("<html><head><title>Table 1: Sample Characteristics and Summary Statistics</title>")
        f.write("<style>table {border-collapse: collapse; margin-bottom: 30px;} th, td {border: 1px solid black; padding: 8px;}</style>")
        f.write("</head><body>")
        f.write("<h1>Table 1: Sample Characteristics and Summary Statistics</h1>")
        
        f.write("<h2>Panel A: Sample Characteristics by Year</h2>")
        f.write(panel_a.to_html())
        
        f.write("<h2>Panel B: Summary Statistics for Key Variables</h2>")
        f.write(panel_b.to_html())
        
        f.write("<h2>Panel C: Correlation Matrix</h2>")
        f.write(panel_c.to_html())
        
        f.write("</body></html>")
    
    print(f"Table 1 saved to {tables_dir}")
    return regression_sample

if __name__ == "__main__":
    generate_table1()