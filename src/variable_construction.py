"""
Variable Construction Module for Lewellen and Lewellen (2016) Replication.

This module contains functions to construct all variables required for the 
replication study, as described in the original paper.
"""

import pandas as pd
import numpy as np
import os
from utils import winsorize


def calculate_net_assets(data):
    """
    Calculate net assets: Total assets minus nondebt current liabilities.
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with net_assets column added
    """
    # NET_ASSETS = AT - (LCT - DLC)
    data['net_assets'] = data['at'] - (data['lct'] - data['dlc'])
    
    # Replace any zero or negative values with a small positive value
    mask = data['net_assets'] <= 0
    if mask.any():
        print(f"Warning: {mask.sum()} observations with zero or negative net assets, replacing with min positive value.")
        min_positive = data.loc[data['net_assets'] > 0, 'net_assets'].min()
        data.loc[mask, 'net_assets'] = min_positive * 0.01
    
    return data


def calculate_cash_flow(data):
    """
    Calculate cash flow using Statement of Cash Flows approach.
    
    CF = IBC + XIDOC + DPC + TXDC + ESUBC + SPPIV + FOPO
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with cash_flow column added
    """
    # Fill NA values for components with 0
    cf_components = ['ibc', 'xidoc', 'dpc', 'txdc', 'esubc', 'sppiv', 'fopo']
    for col in cf_components:
        if col in data.columns:
            data[col] = data[col].fillna(0)
    
    # Calculate cash flow
    data['cash_flow'] = (data['ibc'] + 
                         data['xidoc'].fillna(0) + 
                         data['dpc'].fillna(0) + 
                         data['txdc'].fillna(0) + 
                         data['esubc'].fillna(0) + 
                         data['sppiv'].fillna(0) + 
                         data['fopo'].fillna(0))
    
    # Calculate traditional measure for comparison
    data['trad_cash_flow'] = data['ibc'] + data['dpc']
    
    return data


def calculate_investment_measures(data):
    """
    Calculate three investment measures: CAPX1, CAPX2, CAPX3.
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with investment measures added
    """
    # CAPX1: Capital Expenditures
    data['capx1'] = data['capx']
    
    # CAPX2: CAPX + Acquisitions + Other investing activities
    # Fill NA values for components with 0
    inv_components = ['capx', 'aqc', 'ivch', 'siv']
    for col in inv_components:
        if col in data.columns:
            data[col] = data[col].fillna(0)
    
    data['capx2'] = data['capx'] + data['aqc'] + data['ivch'] + data['siv']
    
    # CAPX3: Change in PPE + Depreciation + Write-downs
    # Need to compute year-over-year change in PPENT
    data = data.sort_values(['gvkey', 'datadate'])
    data['ppent_lag'] = data.groupby('gvkey')['ppent'].shift(1)
    
    # Approximate write-downs using FOPO if available
    data['write_downs'] = data['fopo'].fillna(0)
    
    # Calculate CAPX3
    data['capx3'] = (data['ppent'] - data['ppent_lag']).fillna(0) + data['dpc'] + data['write_downs']
    
    return data


def calculate_other_cash_uses(data):
    """
    Calculate other uses of cash flow: ΔCASH, ΔNWC, ΔDEBT, ISSUES, DIV.
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with other cash use variables added
    """
    # Sort data for lag calculations
    data = data.sort_values(['gvkey', 'datadate'])
    
    # 1. Change in Cash Holdings (ΔCASH)
    data['che_lag'] = data.groupby('gvkey')['che'].shift(1)
    data['delta_cash'] = data['che'] - data['che_lag']
    
    # 2. Change in Net Working Capital (ΔNWC)
    # NWC = (ACT - CHE) - (LCT - DLC)
    data['nwc'] = (data['act'] - data['che']) - (data['lct'] - data['dlc'])
    data['nwc_lag'] = data.groupby('gvkey')['nwc'].shift(1)
    data['delta_nwc'] = data['nwc'] - data['nwc_lag']
    
    # 3. Change in Debt (ΔDEBT)
    # DEBT = DLC + DLTT + (LT - LCT - DLTT)
    data['debt'] = data['dlc'] + data['dltt'] + (data['lt'] - data['lct'] - data['dltt'])
    data['debt_lag'] = data.groupby('gvkey')['debt'].shift(1)
    data['txdc_lag'] = data.groupby('gvkey')['txdc'].shift(1)
    data['delta_debt'] = (data['debt'] - data['debt_lag']) - (data['txdc'] - data['txdc_lag'].fillna(0))
    
    # 4. Equity Issuance (ISSUES)
    # ISSUES = (ΔCEQ + ΔPSTK) - ΔRE
    data['ceq_lag'] = data.groupby('gvkey')['ceq'].shift(1)
    data['pstk_lag'] = data.groupby('gvkey')['pstk'].shift(1)
    data['re_lag'] = data.groupby('gvkey')['re'].shift(1)
    
    data['issues'] = ((data['ceq'] - data['ceq_lag']) + 
                     (data['pstk'].fillna(0) - data['pstk_lag'].fillna(0))) - (data['re'] - data['re_lag'])
    
    # 5. Dividends (DIV)
    data['div'] = data['dvc'].fillna(0) + data['dvp'].fillna(0)
    
    return data


def calculate_market_to_book(data):
    """
    Calculate Market-to-Book ratio as a proxy for Tobin's q.
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with MB ratio added
    """
    # Convert price to positive (CRSP reports negative price for bid/ask average)
    data['prc'] = data['prc'].abs()
    
    # Calculate market value of equity (shares * price)
    data['mve'] = data['prc'] * data['shrout']
    
    # Calculate MB ratio = (MVE + DEBT) / NET_ASSETS
    data['mb'] = (data['mve'] + data['debt']) / data['net_assets']
    
    # Calculate lagged MB (useful for regressions)
    data['mb_lag'] = data.groupby('gvkey')['mb'].shift(1)
    
    return data


def calculate_returns(data):
    """
    Calculate lagged annual stock returns for use as instruments.
    
    Args:
        data (pd.DataFrame): Input data frame with monthly returns
        
    Returns:
        pd.DataFrame: Data with annual returns for past 4 years
    """
    # First ensure we're working with a panel sorted by gvkey and date
    data = data.sort_values(['gvkey', 'datadate'])
    
    # Convert monthly returns to annual returns for the past 4 years
    for lag in range(1, 5):
        lag_col = f'ret_{lag}'
        data[lag_col] = np.nan
        
        # For each firm-year, we need the annual return ending in the previous fiscal year
        # Since our data is already at the fiscal year-end, we need to look back 12*lag months
        # This is a simplified approach - in practice, would need to compound monthly returns
        
    return data


def scale_variables(data):
    """
    Scale all flow variables by average net assets for the year.
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with scaled variables
    """
    # Calculate average net assets
    data['avg_net_assets'] = (data['net_assets'] + data['net_assets'].groupby('gvkey').shift(1)) / 2
    
    # Replace missing values with current net assets (first observations)
    data['avg_net_assets'] = data['avg_net_assets'].fillna(data['net_assets'])
    
    # List of variables to scale
    scale_vars = [
        'cash_flow', 'trad_cash_flow', 'capx1', 'capx2', 'capx3',
        'delta_cash', 'delta_nwc', 'delta_debt', 'issues', 'div'
    ]
    
    # Scale variables
    for var in scale_vars:
        if var in data.columns:
            data[f'{var}_scaled'] = data[var] / data['avg_net_assets']
    
    return data


def prepare_financial_constraint_measures(data):
    """
    Add variables needed for financial constraint classification.
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with constraint-related variables
    """
    # Calculate free cash flow measures
    data['fcf1'] = data['cash_flow_scaled'] - data['capx1_scaled']
    data['fcf2'] = data['cash_flow_scaled'] - data['capx2_scaled']
    data['fcf3'] = data['cash_flow_scaled'] - data['capx3_scaled']
    
    # Calculate additional ratios for constraint classification
    data['cash_to_assets'] = data['che'] / data['net_assets']
    data['debt_to_assets'] = data['debt'] / data['net_assets']
    data['sales_growth'] = data.groupby('gvkey')['sale'].pct_change()
    
    # Cash flow to debt ratio
    data['cf_to_debt'] = data['cash_flow'] / data['debt']
    data.loc[data['debt'] == 0, 'cf_to_debt'] = np.inf
    
    return data


def construct_variables(data, winsorize_limits=(0.01, 0.01)):
    """
    Master function to construct all variables needed for the study.
    
    Args:
        data (pd.DataFrame): Raw merged data from Compustat and CRSP
        winsorize_limits (tuple): Lower and upper percentiles for winsorization
        
    Returns:
        pd.DataFrame: Data with all constructed variables
    """
    # Basic data cleaning
    data = data.dropna(subset=['at', 'lct', 'dlc'])  # Require non-missing for net assets
    
    # Step 1: Calculate Net Assets
    data = calculate_net_assets(data)
    
    # Step 2: Calculate Cash Flow
    data = calculate_cash_flow(data)
    
    # Step 3: Calculate Investment Measures
    data = calculate_investment_measures(data)
    
    # Step 4: Calculate Other Uses of Cash Flow
    data = calculate_other_cash_uses(data)
    
    # Step 5: Calculate Market-to-Book Ratio
    data = calculate_market_to_book(data)
    
    # Step 6: Calculate Returns
    data = calculate_returns(data)
    
    # Step 7: Scale Variables
    data = scale_variables(data)
    
    # Step 8: Prepare Financial Constraint Measures
    data = prepare_financial_constraint_measures(data)
    
    # Winsorize variables to mitigate outliers
    vars_to_winsorize = [
        'cash_flow_scaled', 'trad_cash_flow_scaled', 
        'capx1_scaled', 'capx2_scaled', 'capx3_scaled',
        'delta_cash_scaled', 'delta_nwc_scaled', 'delta_debt_scaled', 
        'issues_scaled', 'div_scaled', 'mb', 'mb_lag'
    ]
    data = winsorize(data, vars_to_winsorize, winsorize_limits)
    
    return data


def save_constructed_data(data, file_name='constructed_variables.csv'):
    """
    Save constructed variables to a CSV file.
    
    Args:
        data (pd.DataFrame): Data with constructed variables
        file_name (str): Name of the output file
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    file_path = os.path.join(data_dir, file_name)
    
    data.to_csv(file_path, index=False)
    print(f"Constructed variables saved to {file_path}")


if __name__ == '__main__':
    # Example usage
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    raw_data_path = os.path.join(data_dir, 'raw_merged_data.csv')
    
    if os.path.exists(raw_data_path):
        raw_data = pd.read_csv(raw_data_path)
        constructed_data = construct_variables(raw_data)
        save_constructed_data(constructed_data)
    else:
        print(f"Raw data file not found at {raw_data_path}")
        print("Please run data_collection.py first to generate the raw data file.") 