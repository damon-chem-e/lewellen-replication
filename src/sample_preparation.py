"""
Sample Preparation Module for Lewellen and Lewellen (2016) Replication.

This module contains functions to filter and prepare the sample dataset
according to the criteria specified in the original paper.
"""

import pandas as pd
import numpy as np
import os
import statsmodels.api as sm # Added for OLS regression

# Fix import path
try:
    from utils import compute_nyse_size_percentile, winsorize_by_year
except ModuleNotFoundError:
    from src.utils import compute_nyse_size_percentile, winsorize_by_year


def forecast_fcf(data):
    """
    Forecast free cash flow (FCF1_scaled) using annual cross-sectional regressions.
    FCF_scaled_it = a_t + b1_t * SalesGrowthScaled_i,t-1 + b2_t * CFScaled_i,t-1 + 
                    b3_t * LeverageScaled_i,t-1 + b4_t * CashScaled_i,t-1 + e_it
    All predictors are lagged and scaled by lagged average net assets.
    """
    print("Forecasting Free Cash Flow (FCF)...")
    data_copy = data.copy()
    data_copy = data_copy.sort_values(['gvkey', 'fyear'])

    # Calculate lagged average net assets for scaling predictors
    # avg_net_assets is (net_assets_t + net_assets_t-1)/2
    data_copy['avg_net_assets_lag1'] = data_copy.groupby('gvkey')['avg_net_assets'].shift(1)

    # Prepare predictors (X_t-1 variables scaled by AvgNetAssets_t-1)
    # 1. Sales Growth Scaled (t-1)
    sale_lag1 = data_copy.groupby('gvkey')['sale'].shift(1)
    sale_lag2 = data_copy.groupby('gvkey')['sale'].shift(2)
    data_copy['sg_pred'] = (sale_lag1 - sale_lag2) / data_copy['avg_net_assets_lag1']

    # 2. Cash Flow Scaled (t-1) - cash_flow_scaled is CF_t / AvgNetAssets_t
    data_copy['cf_pred'] = data_copy.groupby('gvkey')['cash_flow_scaled'].shift(1)
    
    # 3. Leverage Scaled (t-1) - debt is (dlc + dltt)
    # Ensure 'debt' column exists from variable_construction if not already present
    if 'debt' not in data_copy.columns and 'dlc' in data_copy.columns and 'dltt' in data_copy.columns:
        data_copy['debt'] = data_copy['dlc'].fillna(0) + data_copy['dltt'].fillna(0)
    elif 'debt' not in data_copy.columns: # Fallback if dlc/dltt also missing, though unlikely
        data_copy['debt'] = 0 
        print("Warning: 'debt' column and its components (dlc, dltt) not found for FCF forecast. Using 0 for debt.")


    debt_lag1 = data_copy.groupby('gvkey')['debt'].shift(1)
    data_copy['lev_pred'] = debt_lag1 / data_copy['avg_net_assets_lag1']
    
    # 4. Cash Scaled (t-1)
    che_lag1 = data_copy.groupby('gvkey')['che'].shift(1)
    data_copy['cash_pred'] = che_lag1 / data_copy['avg_net_assets_lag1']

    # Dependent variable: fcf1_scaled (FCF_t / AvgNetAssets_t)
    # fcf1_scaled should already be in 'data' from construct_variables module
    if 'fcf1_scaled' not in data_copy.columns:
        print("Error: 'fcf1_scaled' is missing, cannot forecast FCF.")
        data['expected_fcf_scaled'] = np.nan
        return data
        
    data_copy['expected_fcf_scaled'] = np.nan
    
    predictors_base = ['sg_pred', 'cf_pred', 'lev_pred', 'cash_pred']
    
    # Winsorize the predictors annually before using them in regressions
    # Ensure winsorize_by_year is available (it should be via 'from src.utils import ...')
    # The dependent variable (fcf1_scaled) is assumed to be winsorized by variable_construction.py
    print("Winsorizing predictors for FCF forecast...")
    # Create a temporary DataFrame with only the necessary columns for winsorizing by year
    cols_for_winsorizing = ['gvkey', 'fyear'] + predictors_base
    temp_winsorize_df = data_copy[cols_for_winsorizing].copy()
    
    # Filter to existing columns in temp_winsorize_df before passing to winsorize_by_year
    existing_predictors_for_winsorize = [p for p in predictors_base if p in temp_winsorize_df.columns]
    if existing_predictors_for_winsorize:
        temp_winsorize_df = winsorize_by_year(temp_winsorize_df, existing_predictors_for_winsorize, limits=(0.01, 0.01), year_col='fyear')
        # Merge winsorized predictors back into data_copy
        # Drop original predictors and then merge to avoid duplicate columns if names are the same
        data_copy = data_copy.drop(columns=existing_predictors_for_winsorize)
        data_copy = data_copy.merge(temp_winsorize_df[['gvkey', 'fyear'] + existing_predictors_for_winsorize], on=['gvkey', 'fyear'], how='left')
    else:
        print("No predictor variables found to winsorize for FCF forecast.")

    print("Predictor winsorization complete.")

    for year in data_copy['fyear'].unique():
        year_data = data_copy[data_copy['fyear'] == year].copy()
        
        # Prepare data for regression
        Y = year_data['fcf1_scaled']
        X = year_data[existing_predictors_for_winsorize] # Use the possibly reduced list of existing predictors
        X = sm.add_constant(X) # Add intercept
        
        # Drop rows with any NaNs in Y or X for this year's regression
        valid_idx = Y.notna() & X.notna().all(axis=1)
        Y_clean = Y[valid_idx]
        X_clean = X[valid_idx]
        
        if len(X_clean) < (len(existing_predictors_for_winsorize) + 1 + 5): # Check for sufficient observations
            print(f"Skipping FCF forecast for year {year} due to insufficient data: {len(X_clean)} obs.")
            continue
            
        try:
            model = sm.OLS(Y_clean, X_clean).fit()
            predictions = model.predict(X) # Predict on original X for the year to keep NaNs where predictors were NaN
            data_copy.loc[year_data.index, 'expected_fcf_scaled'] = predictions
        except Exception as e:
            print(f"Error during OLS for FCF forecast in year {year}: {e}")

    print("FCF forecasting complete.")
    # Merge expected_fcf_scaled back to the original data DataFrame to ensure only this new column is added
    # from the potentially modified data_copy (which had temp columns for predictors)
    data = data.merge(data_copy[['gvkey', 'fyear', 'expected_fcf_scaled']], on=['gvkey', 'fyear'], how='left')
    return data


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
    # Compute NYSE size percentiles using specified asset base (beginning-of-year)
    data = compute_nyse_size_percentile(data, year_col, assets_col)
    
    # Filter out firms below cutoff
    filtered_data = data[data['nyse_size_percentile'] >= cutoff].copy()
    
    # Print statistics
    total_assets = data[assets_col].sum()
    filtered_assets = filtered_data[assets_col].sum()
    pct_assets = (filtered_assets / total_assets) * 100
    
    print(f"Filtered out firms below NYSE {cutoff*100:.0f}th percentile (based on {assets_col})")
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


def classify_financial_constraints(data, year_col='fyear', n_groups=3, sort_variable='expected_fcf_scaled'):
    """
    Classify firms into financial constraint groups based on a specified sort variable.
    Default sort_variable is 'expected_fcf_scaled'.
    
    Args:
        data (pd.DataFrame): Input data frame
        year_col (str): Column name for fiscal year
        n_groups (int): Number of groups to classify (default: 3 for terciles)
        sort_variable (str): Column name of the variable to sort on for classification.
        
    Returns:
        pd.DataFrame: Data with financial constraint classification
    """
    if sort_variable not in data.columns:
        print(f"Error: Sort variable '{sort_variable}' for constraint classification is missing.")
        # Add placeholder columns if sorting fails, to prevent downstream errors
        data['constraint_group'] = np.nan
        data['constrained'] = 0
        data['unconstrained'] = 0
        return data

    # Copy the dataframe to avoid modifying the original
    result = data.copy()
    
    # For each year, classify firms into constraint groups
    result['constraint_group'] = np.nan # Initialize column

    for year_val in result[year_col].unique():
        year_data_mask = result[year_col] == year_val
        current_year_firms = result[year_data_mask]
        sort_values_for_year = current_year_firms[sort_variable]

        if sort_values_for_year.dropna().empty:
            print(f"Warning: No valid data for sorting in year {year_val} using {sort_variable}. Constraint groups will be NaN.")
            # NaNs are already in result['constraint_group'] for these from initialization
            continue

        try:
            # Use pd.qcut for robust quantile-based group assignment.
            # labels=False returns 0-indexed groups (0, 1, ..., n_groups-1).
            # Add 1 to make them 1-indexed (1, 2, ..., n_groups).
            # duplicates='drop' will create fewer than n_groups if values are concentrated,
            # which is a standard way to handle this.
            
            # Ensure we only pass non-NaN values to qcut if it doesn't handle them well by default for labels=False
            # pd.qcut assigns NaN to corresponding input NaNs, which is desired.
            if len(sort_values_for_year.dropna()) < n_groups:
                print(f"Warning: Not enough unique data points ({len(sort_values_for_year.dropna().unique())} unique of {len(sort_values_for_year.dropna())} non-NaN) to form {n_groups} distinct groups in year {year_val} using {sort_variable}. Constraint groups may be affected or NaN.")
                # Let qcut attempt and handle it, or assign NaNs if it fails

            group_assignments = pd.qcut(sort_values_for_year, n_groups, labels=False, duplicates='drop') + 1
            result.loc[year_data_mask, 'constraint_group'] = group_assignments
            
            # Verify if all groups were formed if necessary (optional debug)
            # counts = result.loc[year_data_mask, 'constraint_group'].value_counts()
            # if len(counts) < n_groups and not sort_values_for_year.dropna().empty:
            #    print(f"Debug: Year {year_val} formed {len(counts)} groups instead of {n_groups}. Counts: {counts.to_dict()}")

        except ValueError as e: # pd.qcut can raise ValueError if bins cannot be formed
            print(f"Warning: pd.qcut could not form {n_groups} quantiles for year {year_val} (e.g., all values identical or too few unique values): {e}. Assigning NaN to constraint_group for this year.")
            result.loc[year_data_mask, 'constraint_group'] = np.nan # Ensure NaNs if qcut fails completely
        except Exception as e: # Catch other potential errors during qcut
            print(f"Error during pd.qcut for constraint classification in year {year_val}: {e}. Assigning NaN.")
            result.loc[year_data_mask, 'constraint_group'] = np.nan

    # Create dummy variables for constrained (bottom group 1) and unconstrained (top group n_groups)
    result['constrained'] = (result['constraint_group'] == 1).astype(int)
    result['unconstrained'] = (result['constraint_group'] == n_groups).astype(int)
    
    print(f"Classified firms into up to {n_groups} constraint groups based on {sort_variable}")
    print(f"Constrained firms (Group 1): {result['constrained'].sum()} firm-years")
    print(f"Unconstrained firms (Group {n_groups}): {result['unconstrained'].sum()} firm-years")
    
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
        'issues_scaled', 'div_scaled',
        # Variables needed for FCF forecast and other controls
        'sale', 'avg_net_assets', 'debt', 'che', 'fcf1_scaled' 
    ]
    
    # Filter missing data on required variables (initial pass)
    # Note: FCF forecast predictors might have their own NaNs due to lags.
    sample = filter_missing_data(data, required_vars)

    # Step 1: Forecast Free Cash Flow
    # This function adds 'expected_fcf_scaled' and necessary predictors
    sample = forecast_fcf(sample) 
    
    # Filter by NYSE size percentile (10th) using beginning-of-year net assets
    if 'net_assets_lag' not in sample.columns and 'net_assets' in sample.columns and 'gvkey' in sample.columns:
        sample['net_assets_lag'] = sample.groupby('gvkey')['net_assets'].shift(1)
    sample = filter_by_nyse_percentile(sample, cutoff=0.1, assets_col='net_assets_lag')
    
    # Ensure we have enough consecutive years for lagged variables
    sample = ensure_balanced_panel(sample, min_years=min_years)
    
    # Classify financial constraints using forecasted FCF
    # The forecast_fcf function should have added 'expected_fcf_scaled'
    sample = classify_financial_constraints(sample, sort_variable='expected_fcf_scaled')
    
    # Create firm and year indicators for fixed effects
    sample['firm_id'] = pd.Categorical(sample['gvkey']).codes
    sample['year_id'] = pd.Categorical(sample['fyear']).codes
    
    # Add lagged cash flow
    sample = sample.sort_values(['gvkey', 'fyear'])
    sample['cash_flow_scaled_lag'] = sample.groupby('gvkey')['cash_flow_scaled'].shift(1)
    
    # Add control variables
    sample['cash_lag'] = sample.groupby('gvkey')['cash_to_assets'].shift(1)
    sample['debt_lag'] = sample.groupby('gvkey')['debt_to_assets'].shift(1)
    
    # Final check for missingness in core regression variables after all steps
    final_regression_vars = [
        'cash_flow_scaled', 'mb_lag', 'capx1_scaled', 'capx3_scaled',
        'delta_cash_scaled', 'delta_nwc_scaled', 'delta_debt_scaled',
        'issues_scaled', 'div_scaled', 'cash_flow_scaled_lag',
        'constrained', 'unconstrained', # from classification
        'firm_id', 'year_id' # for fixed effects
    ]
    # Add other controls if they are used in regressions directly from this sample
    control_vars_for_final_check = ['cash_lag', 'debt_lag'] 
    
    missing_check_vars = final_regression_vars + control_vars_for_final_check
    missing_before_final_dropna = len(sample)
    sample = sample.dropna(subset=[var for var in missing_check_vars if var in sample.columns])
    print(f"Dropped {missing_before_final_dropna - len(sample)} additional rows due to NaNs in final regression/control variables.")

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
    # Ensure this example reflects how data would be available after variable_construction.py
    # This part typically requires 'constructed_variables.csv'
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    constructed_data_path = os.path.join(data_dir, 'constructed_variables.csv')
    
    if os.path.exists(constructed_data_path):
        print("Loading constructed_variables.csv for example run...")
        # It's important that constructed_data has all columns produced by variable_construction.py
        # including 'avg_net_assets', 'fcf1_scaled', 'sale', 'debt', 'che', 'cash_flow_scaled' etc.
        try:
            # Specify dtype for gvkey if it's read as int/float and needs to be string for merge
            constructed_data = pd.read_csv(constructed_data_path, dtype={'gvkey': str})
             # Ensure fyear is int if it's used as int later
            if 'fyear' in constructed_data.columns:
                constructed_data['fyear'] = pd.to_numeric(constructed_data['fyear'], errors='coerce').astype('Int64')

        except Exception as e:
            print(f"Error reading {constructed_data_path}: {e}")
            constructed_data = None

        if constructed_data is not None:
            print(f"Loaded {len(constructed_data)} observations from constructed_variables.csv")
            print("Columns in loaded data:", constructed_data.columns.tolist())
            
            # Minimal check for required columns for forecast_fcf input
            required_for_forecast = ['gvkey', 'fyear', 'avg_net_assets', 'sale', 'cash_flow_scaled', 'debt', 'che', 'fcf1_scaled']
            missing_cols = [col for col in required_for_forecast if col not in constructed_data.columns]
            if missing_cols:
                print(f"ERROR: Missing columns required for FCF forecast in example: {missing_cols}")
            else:
                regression_sample = prepare_regression_sample(constructed_data)
                if regression_sample is not None and not regression_sample.empty:
                    save_regression_sample(regression_sample)
                    print("\nExample run completed. Regression sample prepared and saved.")
                    print(f"Final sample size: {len(regression_sample)}")
                    print(f"Constrained firms: {regression_sample['constrained'].sum()}")
                    print(f"Unconstrained firms: {regression_sample['unconstrained'].sum()}")
                else:
                    print("Example run failed to produce a regression sample.")
        else:
            print("Failed to load constructed_data for example run.")

    else:
        print(f"Constructed variables file not found at {constructed_data_path}")
        print("Please run variable_construction.py first to generate the variables file.")
        print("Then, ensure 'avg_net_assets', 'fcf1_scaled', 'sale', 'debt', 'che', 'cash_flow_scaled' are present.") 