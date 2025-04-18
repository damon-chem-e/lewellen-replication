"""
Sample Preparation Module for Lewellen and Lewellen (2016) Replication.

This module contains functions to filter and prepare the sample dataset
according to the criteria specified in the original paper.
"""

import pandas as pd
import numpy as np
import os
from utils import compute_nyse_size_percentile


def filter_by_nyse_percentile(data, cutoff=0.1, year_col='fyear', assets_col='net_assets'):
    """
    Filter firms by NYSE size percentile, excluding small firms below the specified cutoff.
    
    Args:
        data (pd.DataFrame): Input data frame
        cutoff (float): NYSE size percentile cutoff (default: 0.1 for 10th percentile)
        year_col (str): Column name for fiscal year
        assets_col (str): Column name for asset measure
        
    Returns:
        pd.DataFrame: Filtered data
    """
    # Compute NYSE size percentiles
    data = compute_nyse_size_percentile(data, year_col, assets_col)
    
    # Filter out firms below cutoff
    filtered_data = data[data['nyse_size_percentile'] >= cutoff].copy()
    
    # Print statistics
    total_assets = data[assets_col].sum()
    filtered_assets = filtered_data[assets_col].sum()
    pct_assets = (filtered_assets / total_assets) * 100
    
    print(f"Filtered out firms below NYSE {cutoff*100:.0f}th percentile")
    print(f"Original observations: {len(data)}")
    print(f"Filtered observations: {len(filtered_data)}")
    print(f"Retained {pct_assets:.1f}% of total asset value")
    
    return filtered_data


def filter_missing_data(data, required_vars):
    """
    Filter observations with missing values in required variables.
    
    Args:
        data (pd.DataFrame): Input data frame
        required_vars (list): List of column names for required variables
        
    Returns:
        pd.DataFrame: Filtered data without missing values
    """
    before_count = len(data)
    filtered_data = data.dropna(subset=required_vars)
    after_count = len(filtered_data)
    
    print(f"Filtered out {before_count - after_count} observations with missing values")
    print(f"Retained {after_count} out of {before_count} observations ({after_count/before_count*100:.1f}%)")
    
    return filtered_data


def ensure_balanced_panel(data, firm_id='gvkey', year='fyear', min_years=3):
    """
    Filter to keep only firms with at least min_years consecutive years of data.
    
    Args:
        data (pd.DataFrame): Input data frame
        firm_id (str): Column name for firm identifier
        year (str): Column name for year
        min_years (int): Minimum number of consecutive years required
        
    Returns:
        pd.DataFrame: Filtered data with balanced panel
    """
    # Sort data by firm and year
    data = data.sort_values([firm_id, year])
    
    # Count consecutive years for each firm
    data['year_diff'] = data.groupby(firm_id)[year].diff()
    data['consecutive'] = (data['year_diff'] == 1) | (data['year_diff'].isna())
    data['streak'] = data.groupby(firm_id)['consecutive'].cumsum()
    
    # Identify firms with sufficient consecutive years
    firm_max_streak = data.groupby(firm_id)['streak'].max()
    firms_to_keep = firm_max_streak[firm_max_streak >= min_years].index
    
    # Filter data
    before_count = len(data)
    filtered_data = data[data[firm_id].isin(firms_to_keep)].copy()
    after_count = len(filtered_data)
    
    print(f"Filtered to firms with at least {min_years} consecutive years of data")
    print(f"Retained {len(firms_to_keep)} firms with {after_count} firm-years")
    print(f"Removed {before_count - after_count} observations ({(before_count - after_count)/before_count*100:.1f}%)")
    
    # Drop temporary columns
    filtered_data = filtered_data.drop(['year_diff', 'consecutive', 'streak'], axis=1)
    
    return filtered_data


def classify_financial_constraints(data, year_col='fyear', n_groups=3):
    """
    Classify firms into financial constraint groups based on forecasted free cash flow.
    
    Args:
        data (pd.DataFrame): Input data frame
        year_col (str): Column name for fiscal year
        n_groups (int): Number of groups to classify (default: 3 for terciles)
        
    Returns:
        pd.DataFrame: Data with financial constraint classification
    """
    # Ensure we have necessary constraint variables
    required_vars = ['fcf1', 'fcf2', 'fcf3', 'cash_to_assets', 'debt_to_assets',
                    'sales_growth', 'cash_flow_scaled', 'cf_to_debt']
    
    for var in required_vars:
        if var not in data.columns:
            print(f"Warning: Variable {var} needed for constraint classification is missing")
            return data
    
    # Copy the dataframe to avoid modifying the original
    result = data.copy()
    
    # For each year, classify firms into constraint groups
    for year in result[year_col].unique():
        year_data = result[result[year_col] == year]
        
        # Calculate percentiles for FCF1 (simple approach - in practice, would use forecasted FCF)
        fcf1_percentiles = np.percentile(year_data['fcf1'].dropna(), 
                                         [i * 100/n_groups for i in range(1, n_groups)])
        
        # Assign constraint groups (1 = most constrained, n_groups = least constrained)
        result.loc[result[year_col] == year, 'constraint_group'] = n_groups
        for i, p in enumerate(fcf1_percentiles):
            mask = (result[year_col] == year) & (result['fcf1'] <= p)
            result.loc[mask, 'constraint_group'] = i + 1
    
    # Create dummy variables for constrained (bottom tercile) and unconstrained (top tercile)
    result['constrained'] = (result['constraint_group'] == 1).astype(int)
    result['unconstrained'] = (result['constraint_group'] == n_groups).astype(int)
    
    print(f"Classified firms into {n_groups} constraint groups based on free cash flow")
    print(f"Constrained firms: {result['constrained'].sum()} firm-years")
    print(f"Unconstrained firms: {result['unconstrained'].sum()} firm-years")
    
    return result


def prepare_regression_sample(data, min_years=3):
    """
    Prepare final regression sample with all necessary filters applied.
    
    Args:
        data (pd.DataFrame): Input data with constructed variables
        min_years (int): Minimum consecutive years required
        
    Returns:
        pd.DataFrame: Final regression sample
    """
    # Required variables for regressions
    required_vars = [
        'gvkey', 'fyear', 'cash_flow_scaled', 'mb_lag',
        'capx1_scaled', 'capx2_scaled', 'capx3_scaled',
        'delta_cash_scaled', 'delta_nwc_scaled', 'delta_debt_scaled',
        'issues_scaled', 'div_scaled'
    ]
    
    # Filter missing data on required variables
    sample = filter_missing_data(data, required_vars)
    
    # Filter by NYSE size percentile
    sample = filter_by_nyse_percentile(sample, cutoff=0.1)
    
    # Ensure we have enough consecutive years for lagged variables
    sample = ensure_balanced_panel(sample, min_years=min_years)
    
    # Classify financial constraints
    sample = classify_financial_constraints(sample)
    
    # Create firm and year indicators for fixed effects
    sample['firm_id'] = pd.Categorical(sample['gvkey']).codes
    sample['year_id'] = pd.Categorical(sample['fyear']).codes
    
    # Add lagged cash flow
    sample = sample.sort_values(['gvkey', 'fyear'])
    sample['cash_flow_scaled_lag'] = sample.groupby('gvkey')['cash_flow_scaled'].shift(1)
    
    # Add control variables
    sample['cash_lag'] = sample.groupby('gvkey')['cash_to_assets'].shift(1)
    sample['debt_lag'] = sample.groupby('gvkey')['debt_to_assets'].shift(1)
    
    return sample


def save_regression_sample(data, file_name='regression_sample.csv'):
    """
    Save final regression sample to a CSV file.
    
    Args:
        data (pd.DataFrame): Regression sample data
        file_name (str): Name of the output file
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    file_path = os.path.join(data_dir, file_name)
    
    data.to_csv(file_path, index=False)
    print(f"Regression sample saved to {file_path}")


if __name__ == '__main__':
    # Example usage
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    constructed_data_path = os.path.join(data_dir, 'constructed_variables.csv')
    
    if os.path.exists(constructed_data_path):
        constructed_data = pd.read_csv(constructed_data_path)
        regression_sample = prepare_regression_sample(constructed_data)
        save_regression_sample(regression_sample)
    else:
        print(f"Constructed variables file not found at {constructed_data_path}")
        print("Please run variable_construction.py first to generate the variables file.") 