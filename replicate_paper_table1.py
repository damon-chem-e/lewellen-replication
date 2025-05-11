#!/usr/bin/env python
"""
Script to replicate Table 1 from Lewellen and Lewellen (2016) in the exact format shown in the paper.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules with correct paths
try:
    from utils import winsorize
except ModuleNotFoundError:
    from src.utils import winsorize
from src.variable_construction import construct_variables
from src.sample_preparation import filter_by_nyse_percentile

def generate_paper_table1():
    """
    Generate Table 1: Descriptive Statistics exactly as in the paper.
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
    
    # ------------------------------------------------------------------
    # Table-1 sample: apply NYSE size filter only, no balanced-panel filter
    # and restrict fiscal years to 1971-2009 inclusive (paper sample).
    # ------------------------------------------------------------------

    # Add lagged net assets for NYSE size screen
    constructed_data = constructed_data.sort_values(['gvkey', 'datadate'])
    constructed_data['net_assets_lag'] = constructed_data.groupby('gvkey')['net_assets'].shift(1)

    sample = filter_by_nyse_percentile(constructed_data, cutoff=0.1, assets_col='net_assets_lag')

    # Restrict years
    sample = sample[(sample['fyear'] >= 1971) & (sample['fyear'] <= 2009)]

    print(f'Table-1 sample after NYSE filter and year restriction: {len(sample)} rows')
    del constructed_data
    
    # Create directory for tables
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    print("Generating Paper-Style Table 1: Descriptive Statistics")
    
    # Print available columns for debugging
    print("Available columns in sample DataFrame:")
    for col in sorted(sample.columns):
        print(f"  - {col}")
    
    # Calculate end year for title
    end_year = int(sample['fyear'].max())
    
    # Define variable names and descriptions to match the paper (comprehensive list)
    variables = {
        # Income and cash-flow measures
        'op_prof_scaled':        {'name': 'OP_PROF',   'desc': 'Operating income'},
        'prof_scaled':           {'name': 'PROF',      'desc': 'Income before extraordinary items'},
        'ni_scaled':             {'name': 'NI',        'desc': 'Net income'},
        'depr_scaled':           {'name': 'DEPR',      'desc': 'Depreciation'},
        'othcf_scaled':          {'name': 'OTHCF',     'desc': 'Other operating cash flows'},
        'cash_flow_scaled':      {'name': 'CF',        'desc': 'PROF + DEPR + OTHCF'},
        'trad_cash_flow_scaled': {'name': 'PROF+DEPR', 'desc': 'Income before extraordinary items + Depreciation'},

        # Balance-sheet items
        'che_lag':               {'name': 'CASH',      'desc': 'Cash holdings'},
        'nwc_lag':               {'name': 'NWC',       'desc': 'Noncash net working capital'},
        'plant_lag':             {'name': 'PLANT',     'desc': 'Property, plant, and equipment'},
        'fa_lag':                {'name': 'FA',        'desc': 'Fixed assets'},
        'debt1_lag':             {'name': 'DEBT1',     'desc': 'Short-term debt+Long-term debt'},
        'debt2_lag':             {'name': 'DEBT2',     'desc': 'Total nonoperating liabilities'},
        'toteq_lag':             {'name': 'TOTEQ',     'desc': 'Shareholders\' equity'},
        
        # Changes in balance sheet items
        'delta_na_scaled':       {'name': 'ΔNA',       'desc': 'Change in net assets'},
        'delta_cash_scaled':     {'name': 'ΔCASH',     'desc': 'Change in cash holdings'},
        'delta_debt2_scaled':    {'name': 'ΔDEBT2',    'desc': 'Change in DEBT2'},
        'delta_toteq_scaled':    {'name': 'ΔTOTEQ',    'desc': 'Change in TOTEQ'},
        'delta_nwc_scaled':      {'name': 'ΔNWC',      'desc': 'Change in net working capital'},
        'delta_debt_scaled':     {'name': 'ΔDEBT',     'desc': 'Change in debt'},
        
        # Uses of cash-flow / investment
        'capx1_scaled':          {'name': 'CAPX1',     'desc': 'Capital expenditures (net)'},
        'capx2_scaled':          {'name': 'CAPX2',     'desc': 'CAPX1 + Other investments'},
        'capx3_scaled':          {'name': 'CAPX3',     'desc': 'Total investment in fixed assets'},
        'capx4_scaled':          {'name': 'CAPX4',     'desc': 'Total investment'},
        'issues_scaled':         {'name': 'ISSUES',    'desc': 'Share issuance'},
        'div_scaled':            {'name': 'DIV',       'desc': 'Dividends'},
        'inteq_scaled':          {'name': 'INTEQ',     'desc': 'Internal equity (NI-DIV)'},

        # Free cash-flow
        'fcf1_scaled':           {'name': 'FCF1',      'desc': 'CF - CAPX1'},
        'fcf3_scaled':           {'name': 'FCF3',      'desc': 'CF - CAPX3'},
        'fcf4_scaled':           {'name': 'FCF4',      'desc': 'CF - CAPX4'},

        # Other measures
        'sales_scaled':          {'name': 'SALES',     'desc': 'Revenues'},
        'mb_lag':                {'name': 'MB',        'desc': 'Market-to-book asset ratio'},
        'return':                {'name': 'RETURN',    'desc': 'Annual stock return'}
    }
    
    # Calculate additional variables if they don't exist
    if 'fcf1_scaled' not in sample.columns:
        sample['fcf1_scaled'] = sample['cash_flow_scaled'] - sample['capx1_scaled']
    if 'fcf3_scaled' not in sample.columns:
        sample['fcf3_scaled'] = sample['cash_flow_scaled'] - sample['capx3_scaled']
    if 'fcf4_scaled' not in sample.columns and 'capx4_scaled' in sample.columns:
        sample['fcf4_scaled'] = sample['cash_flow_scaled'] - sample['capx4_scaled']
    if 'inteq_scaled' not in sample.columns and 'ni_scaled' in sample.columns and 'div_scaled' in sample.columns:
        sample['inteq_scaled'] = sample['ni_scaled'] - sample['div_scaled']
    
    # Create summary statistics
    stats_data = []
    for var, info in variables.items():
        if var in sample.columns:
            # ----  annual cross-sectional stats then time-series average ----
            stats_by_year = sample.groupby('fyear')[var].agg(
                mean='mean',
                median='median',
                std='std',
                min=lambda x: x.quantile(0.01),
                max=lambda x: x.quantile(0.99),
                N='count'
            )

            # Time-series average across years
            mean     = stats_by_year['mean'].mean()
            median   = stats_by_year['median'].mean()
            std      = stats_by_year['std'].mean()
            min_val  = stats_by_year['min'].mean()
            max_val  = stats_by_year['max'].mean()
            n        = stats_by_year['N'].mean()
            
            # Add to the table
            stats_data.append({
                'Variable': info['name'],
                'Description': info['desc'],
                'Mean': mean,
                'Median': median,
                'Std': std,
                'Min': min_val,
                'Max': max_val,
                'N': int(round(n))
            })
        else:
            print(f"Warning: Variable '{var}' not found in the dataset. Skipping this variable.")
    
    # Create DataFrame with statistics
    stats_df = pd.DataFrame(stats_data)
    
    # Format DataFrame to match the paper's table
    for col in ['Mean', 'Median', 'Std', 'Min', 'Max']:
        stats_df[col] = stats_df[col].round(3)
    
    # Save statistics as CSV
    stats_df.to_csv(os.path.join(tables_dir, 'paper_table1_stats.csv'), index=False)
    
    # Create Table 1 as a figure to match the paper's format
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Add title
    plt.suptitle(f"TABLE 1", fontsize=14, y=0.98)
    plt.figtext(0.5, 0.95, f"Descriptive Statistics (1971–{end_year})", ha='center', fontsize=12)
    
    # Add the description text from the paper
    description = (
        "Table 1 reports the time-series average of the annual cross-sectional mean, median, standard deviation (Std), 1st per-"
        "centile (Min), 99th percentile (Max), and sample size (N) for the variables listed. All flow variables other than stock returns "
        "are scaled by average net assets during the year, whereas all level variables are scaled by ending net assets (net assets "
        "equal total assets minus nondebt current liabilities). Variables are winsorized annually at their 1st and 99th percentiles. "
        "Accounting data come from Compustat, and returns come from CRSP. The sample consists of all nonfinancial firms that "
        "are larger than the NYSE 10th percentile of NYSE firms (measured by net assets at the beginning of the year) and that have data "
        "for net assets and stock returns."
    )
    plt.figtext(0.5, 0.88, description, wrap=True, ha='center', va='top', fontsize=8)
    
    # Create the table
    table = ax.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc='center',
        loc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Add horizontal lines
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_linewidth(1)
        
        # Add bottom border to all cells
        cell.set_linewidth(0.5)
    
    # Save the table as an image
    plt.tight_layout(rect=[0, 0, 1, 0.87])
    
    table_path = os.path.join(tables_dir, 'paper_table1_descriptive_statistics.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create an HTML version for better readability
    html_content = f"""
    <html>
    <head>
        <title>Table 1: Descriptive Statistics (1971–{end_year})</title>
        <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        h1, h2 {{
            text-align: center;
        }}
        table {{
            border-collapse: collapse;
            margin: 20px auto;
            width: 90%;
        }}
        th, td {{
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .description {{
            text-align: justify;
            margin: 20px auto;
            width: 80%;
            font-size: 0.9em;
        }}
        </style>
    </head>
    <body>
        <h1>TABLE 1</h1>
        <h2>Descriptive Statistics (1971–{end_year})</h2>
        
        <div class="description">
            {description}
        </div>
        
        <table>
            <tr>
                <th>Variable</th>
                <th>Description</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
                <th>N</th>
            </tr>
    """
    
    for _, row in stats_df.iterrows():
        # Replace Greek delta with HTML entity
        var_name = row['Variable'].replace('Δ', '&Delta;')
        
        html_content += f"""
            <tr>
                <td>{var_name}</td>
                <td>{row['Description']}</td>
                <td>{row['Mean']:.3f}</td>
                <td>{row['Median']:.3f}</td>
                <td>{row['Std']:.3f}</td>
                <td>{row['Min']:.3f}</td>
                <td>{row['Max']:.3f}</td>
                <td>{row['N']}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(os.path.join(tables_dir, 'paper_table1.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Create LaTeX version in professional directory
    generate_latex_paper_table1(tables_dir, stats_df, end_year, description)
    
    print(f"Paper Table 1 saved to {tables_dir}")
    return sample

def generate_latex_paper_table1(tables_dir, stats_df, end_year, description):
    """
    Generate LaTeX version of the paper's Table 1 with professional formatting.
    
    Parameters
    ----------
    tables_dir : str
        Directory path where tables are stored
    stats_df : pandas.DataFrame
        DataFrame containing the statistics for the table
    end_year : int
        End year for the table title
    description : str
        Description text for the table
    """
    professional_dir = os.path.join(tables_dir, 'professional')
    os.makedirs(professional_dir, exist_ok=True)
    
    latex_file = os.path.join(professional_dir, 'paper_table1.tex')
    
    # Start LaTeX document
    latex_content = [
        "\\documentclass[12pt]{article}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage{booktabs}",
        "\\usepackage{array}",
        "\\usepackage{caption}",
        "\\usepackage{float}",
        "\\usepackage{geometry}",
        "\\usepackage{siunitx}",
        "\\usepackage{multirow}",
        "\\usepackage{amsmath}",
        "\\usepackage[table]{xcolor}",
        "\\usepackage{threeparttable}",
        "",
        "\\geometry{margin=1in}",
        "",
        "\\begin{document}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Descriptive Statistics (1971--" + str(end_year) + ")}",
        "\\label{tab:paper_table1}",
        "",
        "\\begin{threeparttable}",
        "\\begin{tabular}{l p{5cm} *{6}{S[table-format=1.3]}}",
        "\\toprule",
        "{Variable} & {Description} & {Mean} & {Median} & {Std} & {Min} & {Max} & {N} \\\\",
        "\\midrule"
    ]
    
    # Add data rows
    for _, row in stats_df.iterrows():
        # Replace potentially problematic characters in variable names
        var_name = row['Variable']
        var_name = var_name.replace('Δ', '$\\Delta$')
        
        # Format N as integer
        n_value = f"{int(row['N'])}" if not pd.isna(row['N']) else "{--}"
        
        # Add LaTeX row
        latex_content.append(
            f"{var_name} & {row['Description']} & "
            f"{row['Mean']:.3f} & {row['Median']:.3f} & {row['Std']:.3f} & "
            f"{row['Min']:.3f} & {row['Max']:.3f} & {n_value} \\\\"
        )
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "\\begin{tablenotes}[flushleft]",
        "\\small",
        "\\item \\textit{Notes:} " + description,
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
        "",
        "\\end{document}"
    ])
    
    # Write to file with UTF-8 encoding
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX Paper Table 1 saved to {latex_file}")

if __name__ == "__main__":
    generate_paper_table1()