"""
Variable Construction Module for Lewellen and Lewellen (2016) Replication.

This module contains functions to construct all variables required for the 
replication study, as described in the original paper.
"""

import pandas as pd
import numpy as np
import os

# Fix import path
try:
    from utils import winsorize
except ModuleNotFoundError:
    from src.utils import winsorize


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
    
    # Drop observations with non-positive net assets – they are not in the paper's sample
    neg_count = (data['net_assets'] <= 0).sum()
    if neg_count > 0:
        print(f"Dropping {neg_count} observations with non-positive net assets")
        data = data[data['net_assets'] > 0].copy()
    
    return data


def calculate_income_variables(data):
    """
    Calculate various income variables required for the paper.
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with income variables added
    """
    # Operating Income (OP_PROF) - oiadp in Compustat
    data['op_prof'] = data['oiadp']
    
    # Income before extraordinary items (PROF) - ib in Compustat
    inc_col = 'ib' if 'ib' in data.columns else 'ibc'
    data['prof'] = data[inc_col]
    
    # Net income (NI) - ni in Compustat
    data['ni'] = data['ni']
    
    # Depreciation (DEPR) - dp in Compustat
    dep_col = 'dp' if 'dp' in data.columns else 'dpc'
    data['depr'] = data[dep_col]
    
    return data


def calculate_cash_flow(data):
    """
    Calculate cash flow using Statement of Cash Flows approach.
    
    CF = PROF (ib) + DEPR (dp) + OTHCF  →  equals Compustat OANCF when available.
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with cash_flow column added
    """
    # Choose appropriate income and depreciation variables
    inc_col = 'ib' if 'ib' in data.columns else 'ibc'  # fallback to ibc
    dep_col = 'dp' if 'dp' in data.columns else 'dpc'

    # Ensure required cols exist
    for col in [inc_col, dep_col]:
        if col not in data.columns:
            data[col] = 0.0

    # Calculate other cash flows (OTHCF)
    # If Compustat provides OANCF, compute OTHCF as the residual
    if 'oancf' in data.columns:
        data['cash_flow'] = data['oancf']
        data['othcf'] = data['oancf'] - (data[inc_col].fillna(0) + data[dep_col].fillna(0))
    else:
        # Otherwise reconstruct using other components available (close to paper definition)
        cf_components = ['xidoc', 'txdc', 'esubc', 'sppiv', 'fopo']
        for col in cf_components:
            if col not in data.columns:
                data[col] = 0.0
            data[col] = data[col].fillna(0)

        # Define OTHCF as the sum of these components
        data['othcf'] = data['xidoc'] + data['txdc'] + data['esubc'] + data['sppiv'] + data['fopo']
        data['cash_flow'] = data[inc_col].fillna(0) + data[dep_col].fillna(0) + data['othcf']

    # Traditional measure: PROF + DEPR (ib + dp)
    data['trad_cash_flow'] = data[inc_col] + data[dep_col]
    
    return data


def calculate_balance_sheet_items(data):
    """
    Calculate balance sheet items required for the paper.
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with balance sheet items added
    """
    # Sort data for lag calculations
    data = data.sort_values(['gvkey', 'datadate'])
    
    # PLANT - Property, Plant, and Equipment (ppent in Compustat)
    data['plant'] = data['ppent']
    
    # FA - Fixed Assets (ppent in Compustat)
    data['fa'] = data['ppent']
    
    # NWC - Noncash net working capital ((ACT - CHE) - (LCT - DLC))
    data['nwc'] = (data['act'] - data['che']) - (data['lct'] - data['dlc'])
    
    # DEBT1 - Short-term debt + Long-term debt (dlc + dltt)
    data['debt1'] = data['dlc'] + data['dltt']
    
    # DEBT2 - Total nonoperating liabilities (lt - (lct - dlc))
    data['debt2'] = data['lt'] - (data['lct'] - data['dlc'])
    
    # TOTEQ - Shareholders' equity (ceq + pstk)
    data['toteq'] = data['ceq'] + data['pstk'].fillna(0)
    
    # Add lagged versions (useful for some calculations and referencing in the paper)
    for var in ['plant', 'fa', 'nwc', 'debt1', 'debt2', 'toteq']:
        data[f'{var}_lag'] = data.groupby('gvkey')[var].shift(1)
    
    return data


def calculate_investment_measures(data):
    """
    Calculate investment measures: CAPX1, CAPX2, CAPX3, CAPX4.
    
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
    
    # CAPX4: Total investment (CAPX3 + ΔNWC)
    # Calculate ΔNWC if not already calculated
    if 'delta_nwc' not in data.columns and 'nwc' in data.columns:
        data['nwc_lag'] = data.groupby('gvkey')['nwc'].shift(1)
        data['delta_nwc'] = data['nwc'] - data['nwc_lag']
    
    data['capx4'] = data['capx3'] + data['delta_nwc']
    
    return data


def calculate_other_cash_uses(data):
    """
    Calculate other uses of cash flow: ΔCASH, ΔNWC, ΔDEBT, ΔDEBT2, ΔTOTEQ, ISSUES, DIV, ΔNA.
    
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
    if 'nwc' not in data.columns:
        data['nwc'] = (data['act'] - data['che']) - (data['lct'] - data['dlc'])
        data['nwc_lag'] = data.groupby('gvkey')['nwc'].shift(1)
    data['delta_nwc'] = data['nwc'] - data['nwc_lag']
    
    # 3. Change in Debt (ΔDEBT)
    # DEBT = DLC + DLTT + (LT - LCT - DLTT)
    data['debt'] = data['dlc'] + data['dltt'] + (data['lt'] - data['lct'] - data['dltt'])
    data['debt_lag'] = data.groupby('gvkey')['debt'].shift(1)
    data['txdc_lag'] = data.groupby('gvkey')['txdc'].shift(1)
    data['delta_debt'] = (data['debt'] - data['debt_lag']) - (data['txdc'] - data['txdc_lag'].fillna(0))
    
    # 4. Change in DEBT2 (ΔDEBT2)
    if 'debt2' not in data.columns:
        data['debt2'] = data['lt'] - (data['lct'] - data['dlc'])
        data['debt2_lag'] = data.groupby('gvkey')['debt2'].shift(1)
    data['delta_debt2'] = data['debt2'] - data['debt2_lag']
    
    # 5. Change in TOTEQ (ΔTOTEQ)
    if 'toteq' not in data.columns:
        data['toteq'] = data['ceq'] + data['pstk'].fillna(0)
        data['toteq_lag'] = data.groupby('gvkey')['toteq'].shift(1)
    data['delta_toteq'] = data['toteq'] - data['toteq_lag']
    
    # 6. Change in Net Assets (ΔNA)
    data['net_assets_lag'] = data.groupby('gvkey')['net_assets'].shift(1)
    data['delta_na'] = data['net_assets'] - data['net_assets_lag']
    
    # 7. Equity Issuance (ISSUES)
    # ISSUES = (ΔCEQ + ΔPSTK) - ΔRE
    data['ceq_lag'] = data.groupby('gvkey')['ceq'].shift(1)
    data['pstk_lag'] = data.groupby('gvkey')['pstk'].shift(1)
    data['re_lag'] = data.groupby('gvkey')['re'].shift(1)
    
    data['issues'] = ((data['ceq'] - data['ceq_lag']) + 
                     (data['pstk'].fillna(0) - data['pstk_lag'].fillna(0))) - (data['re'] - data['re_lag'])
    
    # 8. Dividends (DIV)
    data['div'] = data['dvc'].fillna(0) + data['dvp'].fillna(0)
    
    # 9. Internal Equity (INTEQ = NI - DIV)
    if 'ni' in data.columns:
        data['inteq'] = data['ni'] - data['div']
    
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
    Calculate stock returns for the paper.
    
    Args:
        data (pd.DataFrame): Input data frame with returns
        
    Returns:
        pd.DataFrame: Data with annual stock returns
    """
    # Annual stock return (RETURN)
    # If 'ret' is monthly, we need to compound it to get annual
    # For now, we'll assume 'ret' is already the annual return
    data['return'] = data['ret']
    
    # Calculate lagged annual returns for the past 4 years
    data = data.sort_values(['gvkey', 'datadate'])
    for lag in range(1, 5):
        lag_col = f'ret_{lag}'
        data[lag_col] = data.groupby('gvkey')['return'].shift(lag)
    
    return data


def calculate_sales(data):
    """
    Add sales/revenue variable.
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with sales variable added
    """
    # SALES - Revenues (sale in Compustat)
    data['sales'] = data['sale']
    
    return data


def scale_variables(data):
    """
    Scale all flow variables by average net assets for the year.
    Level variables are scaled by ending net assets.
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with scaled variables
    """
    # Make a copy to avoid SettingWithCopyWarning
    result = data.copy()
    
    # Sort by gvkey and date for correct lagged calculations
    if 'gvkey' in result.columns and 'datadate' in result.columns:
        result = result.sort_values(['gvkey', 'datadate'])
        
        # Calculate average net assets
        result['net_assets_lag'] = result.groupby('gvkey')['net_assets'].shift(1)
        result['avg_net_assets'] = (result['net_assets'] + result['net_assets_lag'].fillna(0)) / 2
        
        # Replace zero averages with current net assets
        mask = result['avg_net_assets'] <= 0
        result.loc[mask, 'avg_net_assets'] = result.loc[mask, 'net_assets']
    else:
        # If missing required columns, use net_assets as fallback
        print("Warning: Missing gvkey or datadate columns for proper scaling.")
        result['avg_net_assets'] = result['net_assets']
    
    # List of flow variables to scale by average net assets
    flow_vars = [
        'cash_flow', 'trad_cash_flow', 'op_prof', 'prof', 'ni', 'depr', 'othcf',
        'capx1', 'capx2', 'capx3', 'capx4', 'inteq',
        'delta_cash', 'delta_nwc', 'delta_debt', 'delta_debt2', 'delta_toteq', 'delta_na',
        'issues', 'div', 'sales'
    ]
    
    # Scale flow variables by average net assets
    for var in flow_vars:
        if var in result.columns:
            result[f'{var}_scaled'] = result[var] / result['avg_net_assets']
    
    # No need to scale 'return' as it's already a percentage
    
    return result


def prepare_financial_constraint_measures(data):
    """
    Add variables needed for financial constraint classification.
    
    Args:
        data (pd.DataFrame): Input data frame
        
    Returns:
        pd.DataFrame: Data with constraint-related variables
    """
    # Calculate free cash flow measures
    data['fcf1_scaled'] = data['cash_flow_scaled'] - data['capx1_scaled']
    data['fcf3_scaled'] = data['cash_flow_scaled'] - data['capx3_scaled']
    
    # Add FCF4 (CF - CAPX4)
    if 'capx4_scaled' in data.columns:
        data['fcf4_scaled'] = data['cash_flow_scaled'] - data['capx4_scaled']
    
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
    # Make a copy of the data
    result = data.copy()
    
    # Basic data cleaning
    result = result.dropna(subset=['at', 'lct', 'dlc'])  # Require non-missing for net assets
    
    # Exclude regulated utilities (SIC 4900–4999) and financials (SIC 6000–6999)
    if 'sic' in result.columns:
        before_sic = len(result)
        result = result[((result['sic'] < 4900) | (result['sic'] > 4999)) &
                        ((result['sic'] < 6000) | (result['sic'] > 6999) | (result['sic'].isna()))]
        print(f"Removed {before_sic - len(result)} utility/financial firm-year observations based on SIC")
    
    # Print initial dataset size
    print(f"Starting with {len(result)} observations after removing missing values")
    
    # Step 1: Calculate Net Assets
    result = calculate_net_assets(result)
    
    # Step 2: Calculate Income Variables
    result = calculate_income_variables(result)
    
    # Step 3: Calculate Cash Flow
    result = calculate_cash_flow(result)
    
    # Step 4: Calculate Balance Sheet Items
    result = calculate_balance_sheet_items(result)
    
    # Step 5: Calculate Investment Measures
    result = calculate_investment_measures(result)
    
    # Step 6: Calculate Other Uses of Cash Flow
    result = calculate_other_cash_uses(result)
    
    # Step 7: Calculate Market-to-Book Ratio
    result = calculate_market_to_book(result)
    
    # Step 8: Calculate Sales/Revenues
    result = calculate_sales(result)
    
    # Step 9: Calculate Returns
    result = calculate_returns(result)
    
    # Step 10: Scale Variables
    result = scale_variables(result)
    
    # Step 11: Prepare Financial Constraint Measures
    result = prepare_financial_constraint_measures(result)
    
    # Check availability of variables to winsorize
    vars_to_winsorize = [
        'op_prof_scaled', 'prof_scaled', 'ni_scaled', 'depr_scaled', 'othcf_scaled',
        'cash_flow_scaled', 'trad_cash_flow_scaled', 
        'capx1_scaled', 'capx2_scaled', 'capx3_scaled', 'capx4_scaled',
        'delta_cash_scaled', 'delta_nwc_scaled', 'delta_debt_scaled', 
        'delta_debt2_scaled', 'delta_toteq_scaled', 'delta_na_scaled',
        'issues_scaled', 'div_scaled', 'inteq_scaled', 'sales_scaled',
        'return', 'mb', 'mb_lag',
        'fcf1_scaled'
    ]
    
    # Filter the list to only include variables that exist in the data
    vars_exist = [var for var in vars_to_winsorize if var in result.columns]
    if len(vars_exist) < len(vars_to_winsorize):
        missing_vars = set(vars_to_winsorize) - set(vars_exist)
        print(f"Warning: The following variables are missing and won't be winsorized: {missing_vars}")
    
    # Annual winsorise variables within each fiscal year
    if vars_exist:
        from src.utils import winsorize_by_year  # local import to avoid circular
        result = winsorize_by_year(result, vars_exist, winsorize_limits, year_col='fyear')
    
    print(f"Completed variable construction with {len(result)} observations")
    return result


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