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
    
    # Create LaTeX version in professional directory
    generate_latex_table1(tables_dir, panel_a, panel_b, panel_c)
    
    print(f"Table 1 saved to {tables_dir}")
    return regression_sample

def generate_latex_table1(tables_dir, panel_a, panel_b, panel_c):
    """
    Generate LaTeX version of Table 1 with professional formatting.
    
    Parameters
    ----------
    tables_dir : str
        Directory path where tables are stored
    panel_a : pandas.DataFrame
        Panel A data (sample characteristics by year)
    panel_b : pandas.DataFrame
        Panel B data (summary statistics)
    panel_c : pandas.DataFrame
        Panel C data (correlation matrix)
    """
    professional_dir = os.path.join(tables_dir, 'professional')
    os.makedirs(professional_dir, exist_ok=True)
    
    latex_file = os.path.join(professional_dir, 'table1.tex')
    
    # Start LaTeX document
    latex_content = [
        "\\documentclass[12pt]{article}",
        "\\usepackage{booktabs}",
        "\\usepackage{array}",
        "\\usepackage{caption}",
        "\\usepackage{float}",
        "\\usepackage{geometry}",
        "\\usepackage{siunitx}",
        "\\usepackage{lscape}",
        "\\usepackage{pdflscape}",
        "",
        "\\geometry{margin=1in}",
        "",
        "\\begin{document}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Sample Characteristics and Summary Statistics}",
        "\\label{tab:table1}",
        "",
        "\\vspace{0.5cm}",
        ""
    ]
    
    # Panel A: Sample characteristics by year
    latex_content.extend([
        "\\begin{minipage}{\\textwidth}",
        "\\textbf{Panel A: Sample Characteristics by Year}",
        "\\vspace{0.3cm}",
        "",
        "\\begin{tabular}{l S[table-format=4.0] S[table-format=6.0] S[table-format=4.0] S[table-format=3.0] S[table-format=5.0] S[table-format=3.0] S[table-format=2.0]}",
        "\\toprule",
        "{Year} & {Number of Firms} & {Total Assets (\\$M)} & {Average Assets (\\$M)} & {Median Assets (\\$M)} & {Total CAPX (\\$M)} & {Average CAPX (\\$M)} & {Median CAPX (\\$M)} \\\\",
        "\\midrule"
    ])
    
    # Add Panel A data rows
    for year, row in panel_a.iterrows():
        latex_content.append(f"{year} & {row['Number of Firms']:.0f} & {row['Total Assets ($M)']:.0f} & {row['Average Assets ($M)']:.0f} & {row['Median Assets ($M)']:.0f} & {row['Total CAPX ($M)']:.0f} & {row['Average CAPX ($M)']:.0f} & {row['Median CAPX ($M)']:.0f} \\\\")
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{minipage}",
        "",
        "\\vspace{1cm}",
        ""
    ])
    
    # Panel B: Summary statistics for key variables
    latex_content.extend([
        "\\begin{minipage}{\\textwidth}",
        "\\textbf{Panel B: Summary Statistics for Key Variables}",
        "\\vspace{0.3cm}",
        "",
        "\\begin{tabular}{l S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] S[table-format=1.3]}",
        "\\toprule",
        "{Variable} & {Mean} & {Median} & {Std Dev} & {Min} & {P25} & {P75} & {Max} \\\\",
        "\\midrule"
    ])
    
    # Add Panel B data rows
    for var, row in panel_b.iterrows():
        latex_content.append(f"{var} & {row['Mean']:.3f} & {row['Median']:.3f} & {row['Std Dev']:.3f} & {row['Min']:.3f} & {row['P25']:.3f} & {row['P75']:.3f} & {row['Max']:.3f} \\\\")
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{minipage}",
        "",
        "\\vspace{1cm}",
        ""
    ])
    
    # Panel C: Correlation matrix
    latex_content.extend([
        "\\begin{minipage}{\\textwidth}",
        "\\textbf{Panel C: Correlation Matrix}",
        "\\vspace{0.3cm}",
        "",
        "\\begin{tabular}{l *{12}{c}}",
        "\\toprule",
        "{} & {CF} & {CAPX1} & {CAPX2} & {CAPX3} & {$\\Delta$Cash} & {$\\Delta$NWC} & {$\\Delta$Debt} & {Issues} & {Div} & {M/B} & {Cash$_{t-1}$} & {Debt$_{t-1}$} \\\\",
        "\\midrule"
    ])
    
    # Short names for columns (used in correlation matrix)
    short_names = [
        "CF", "CAPX1", "CAPX2", "CAPX3", "$\\Delta$Cash", "$\\Delta$NWC", 
        "$\\Delta$Debt", "Issues", "Div", "M/B", "Cash$_{t-1}$", "Debt$_{t-1}$"
    ]
    
    # Add Panel C data rows
    for i, var in enumerate(panel_c.index):
        row_values = []
        for j in range(len(short_names)):
            if i == j:
                val = "1.00"
            else:
                val = f"{panel_c.iloc[i, j]:.2f}"
            row_values.append(val)
        
        latex_content.append(f"{var} & {' & '.join(row_values)} \\\\")
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{minipage}",
        "",
        "\\begin{minipage}{\\textwidth}",
        "\\vspace{0.5cm}",
        "\\small",
        "\\textit{Notes:} This table presents sample characteristics and summary statistics for the firm-years in our sample. Panel A reports the number of firms and their total, average, and median assets and capital expenditures by selected years. Panel B presents summary statistics for the key variables used in our analysis. All flow variables are scaled by beginning-of-year net assets. Panel C presents the correlation matrix for these variables.",
        "\\end{minipage}",
        "",
        "\\end{table}",
        "",
        "\\end{document}"
    ])
    
    # Write to file
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX Table 1 saved to {latex_file}")

if __name__ == "__main__":
    generate_table1()