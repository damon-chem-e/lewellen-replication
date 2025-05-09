"""
Utility functions for the Lewellen and Lewellen (2016) replication study.

This module provides helper functions for connecting to WRDS, handling data,
and implementing common operations used throughout the replication.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import wrds conditionally - we don't need it for analysis-only mode
try:
    import wrds
    # Try to import credentials, but don't fail if they're not available
    try:
        from config import wrds_credentials
        HAS_WRDS_CREDENTIALS = True
    except ImportError:
        HAS_WRDS_CREDENTIALS = False
except ImportError:
    HAS_WRDS_CREDENTIALS = False


def connect_to_wrds():
    """
    Establish a connection to WRDS database.
    
    Returns:
        wrds.Connection: A connection object to the WRDS database, or None if connection fails
    """
    if not HAS_WRDS_CREDENTIALS:
        print("WRDS credentials not available. Skipping connection.")
        return None
    
    try:
        conn = wrds.Connection(wrds_username=wrds_credentials.wrds_username,
                              wrds_password=wrds_credentials.wrds_password)
        print(f"Successfully connected to WRDS as {wrds_credentials.wrds_username}")
        return conn
    except Exception as e:
        print(f"Failed to connect to WRDS: {e}")
        return None


def winsorize(data, columns, limits=(0.01, 0.01)):
    """
    Winsorize data to mitigate outliers.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        columns (list): List of column names to winsorize
        limits (tuple): Lower and upper percentiles for winsorization
        
    Returns:
        pd.DataFrame: DataFrame with winsorized values
    """
    result = data.copy()
    for col in columns:
        if col in result.columns:
            lower_limit = result[col].quantile(limits[0])
            upper_limit = result[col].quantile(1 - limits[1])
            result[col] = result[col].clip(lower=lower_limit, upper=upper_limit)
    return result


# -----------------------------------------------------------------------------
# NEW: Annual (within-year) winsorisation to match Lewellen & Lewellen (2016)
# -----------------------------------------------------------------------------

def winsorize_by_year(data, columns, limits=(0.01, 0.01), year_col='fyear'):
    """Winsorize columns year-by-year.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        columns (list): Column names to winsorize.
        limits (tuple): (lower, upper) percentiles to clip at, e.g. (0.01,0.01).
        year_col (str): Column identifying fiscal year.

    Returns:
        pd.DataFrame: Copy with winsorized columns.
    """
    if year_col not in data.columns:
        raise ValueError(f"'{year_col}' column not found for annual winsorisation")

    result = data.copy()
    for col in columns:
        if col not in result.columns:
            continue

        def _clip(s):
            lower = s.quantile(limits[0])
            upper = s.quantile(1 - limits[1])
            return s.clip(lower, upper)

        result[col] = result.groupby(year_col)[col].transform(_clip)

    return result


# Updated to allow arbitrary asset column (default unchanged)
def compute_nyse_size_percentile(data, year_col='fyear', assets_col='net_assets'):
    """
    Compute NYSE size percentiles for each year in the data.
    
    Args:
        data (pd.DataFrame): DataFrame with firm data
        year_col (str): Column name for fiscal year
        assets_col (str): Column name for asset measure
        
    Returns:
        pd.DataFrame: DataFrame with original data and percentile column
    """
    result = data.copy()
    
    # Identify NYSE firms (exchg == 1 in Compustat/CRSP merge)
    nyse_firms = result[result['exchg'] == 1].copy()
    
    # Calculate percentiles by year for NYSE firms
    percentiles = {}
    for year in nyse_firms[year_col].unique():
        year_data = nyse_firms[nyse_firms[year_col] == year][assets_col].dropna()
        if year_data.empty:
            continue
        percentiles[year] = {
            p/100: np.percentile(year_data, p) 
            for p in range(1, 100)
        }
    
    # Add percentile column
    result['nyse_size_percentile'] = 0.0
    for year in result[year_col].unique():
        if year in percentiles:
            year_data = result[result[year_col] == year]
            for p in sorted(percentiles[year].keys()):
                mask = (result[year_col] == year) & (result[assets_col] <= percentiles[year][p])
                result.loc[mask, 'nyse_size_percentile'] = p
    
    return result 