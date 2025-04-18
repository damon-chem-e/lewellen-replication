"""
Data Collection Module for Lewellen and Lewellen (2016) Replication.

This module contains functions to query and retrieve data from WRDS databases
(Compustat and CRSP) needed for the replication study.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from utils import connect_to_wrds


def query_compustat_annual(conn, start_year=1971, end_year=2023):
    """
    Query Compustat Annual data.
    
    Args:
        conn: WRDS connection object
        start_year (int): Start year for data collection
        end_year (int): End year for data collection
        
    Returns:
        pd.DataFrame: Compustat annual data
    """
    print(f"Querying Compustat annual data from {start_year} to {end_year}...")
    
    # SQL query to fetch required Compustat variables
    query = f"""
    SELECT 
        gvkey, datadate, fyear, indfmt, consol, popsrc, datafmt, tic, cusip, sic,
        at, lct, dlc, ibc, xidoc, dpc, txdc, esubc, sppiv, fopo, ppent, capx, aqc,
        ivch, siv, che, act, dltt, lt, ceq, pstk, re, dvc, dvp, sale, exchg
    FROM comp.funda
    WHERE 
        indfmt = 'INDL' AND
        datafmt = 'STD' AND
        consol = 'C' AND
        popsrc = 'D' AND
        fyear BETWEEN {start_year} AND {end_year}
    """
    compustat = conn.raw_sql(query)
    
    # Convert date column to datetime
    compustat['datadate'] = pd.to_datetime(compustat['datadate'])
    
    # Filter out financial firms (SIC codes 6000-6999)
    compustat = compustat[(compustat['sic'] < 6000) | (compustat['sic'] > 6999)]
    
    print(f"Retrieved {len(compustat)} firm-year observations from Compustat")
    return compustat


def query_crsp_monthly(conn, start_year=1971, end_year=2023):
    """
    Query CRSP monthly stock return and price data.
    
    Args:
        conn: WRDS connection object
        start_year (int): Start year for data collection
        end_year (int): End year for data collection
        
    Returns:
        pd.DataFrame: CRSP monthly data
    """
    print(f"Querying CRSP monthly data from {start_year} to {end_year}...")
    
    # Adjust date range to include previous years for return calculations
    adj_start_year = start_year - 5  # Need lagged returns up to 5 years
    
    # SQL query for CRSP
    query = f"""
    SELECT 
        a.permno, a.permco, a.date, a.ret, a.prc, a.shrout, a.exchcd,
        a.shrcd, b.gvkey
    FROM crsp.msf a
    LEFT JOIN crsp.ccmxpf_lnkhist b
    ON a.permno = b.lpermno
    WHERE 
        a.date BETWEEN '{adj_start_year}-01-01' AND '{end_year}-12-31' AND
        b.linktype IN ('LU', 'LC', 'LS') AND
        b.linkprim IN ('P', 'C')
    """
    crsp = conn.raw_sql(query)
    
    # Convert date column to datetime
    crsp['date'] = pd.to_datetime(crsp['date'])
    
    # Add year column for easier merging
    crsp['year'] = crsp['date'].dt.year
    
    print(f"Retrieved {len(crsp)} monthly observations from CRSP")
    return crsp


def merge_compustat_crsp(compustat, crsp):
    """
    Merge Compustat and CRSP data.
    
    Args:
        compustat (pd.DataFrame): Compustat annual data
        crsp (pd.DataFrame): CRSP monthly data
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    print("Merging Compustat and CRSP datasets...")
    
    # Prepare Compustat data for merge
    compustat['year'] = compustat['datadate'].dt.year
    compustat['month'] = compustat['datadate'].dt.month
    
    # Match CRSP monthly data to fiscal year-end
    merged_data = pd.merge(
        compustat,
        crsp,
        on=['gvkey', 'year'],
        how='inner'
    )
    
    # Keep only observations where the CRSP date is in the same month as the fiscal year-end
    merged_data = merged_data[merged_data['datadate'].dt.month == merged_data['date'].dt.month]
    
    print(f"Merged dataset contains {len(merged_data)} firm-year observations")
    return merged_data


def save_data(data, file_name):
    """
    Save data to a csv file.
    
    Args:
        data (pd.DataFrame): Data to save
        file_name (str): Name of the output file
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    file_path = os.path.join(data_dir, file_name)
    
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


def collect_raw_data(start_year=1971, end_year=2023):
    """
    Collect and merge all raw data required for the study.
    
    Args:
        start_year (int): Start year for data collection
        end_year (int): End year for data collection
        
    Returns:
        pd.DataFrame: Merged dataset with Compustat and CRSP data
    """
    # Connect to WRDS
    conn = connect_to_wrds()
    if conn is None:
        print("Failed to connect to WRDS. Cannot proceed with data collection.")
        return None
    
    # Query Compustat and CRSP data
    compustat_data = query_compustat_annual(conn, start_year, end_year)
    crsp_data = query_crsp_monthly(conn, start_year, end_year)
    
    # Merge datasets
    merged_data = merge_compustat_crsp(compustat_data, crsp_data)
    
    # Save raw data
    save_data(merged_data, 'raw_merged_data.csv')
    
    # Close WRDS connection
    conn.close()
    
    return merged_data


if __name__ == '__main__':
    # Example usage
    data = collect_raw_data(1971, 2023) 