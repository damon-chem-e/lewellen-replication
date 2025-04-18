"""
Main script for Lewellen and Lewellen (2016) Replication Study.

This script orchestrates the entire data preparation pipeline by calling
modules for data collection, variable construction, and sample preparation.
"""

import os
import sys
import pandas as pd
import time
from datetime import datetime

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import modules
from src.data_collection import collect_raw_data
from src.variable_construction import construct_variables, save_constructed_data
from src.sample_preparation import prepare_regression_sample, save_regression_sample


def check_data_directory():
    """
    Check if data directory exists, create if not.
    
    Returns:
        str: Path to data directory
    """
    project_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_dir, 'data')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")
    else:
        print(f"Data directory exists: {data_dir}")
    
    return data_dir


def run_pipeline(start_year=1971, end_year=2023, force_refresh=False):
    """
    Run the full data preparation pipeline.
    
    Args:
        start_year (int): Start year for data collection
        end_year (int): End year for data collection
        force_refresh (bool): If True, rerun all steps even if files exist
        
    Returns:
        pd.DataFrame: Final regression sample
    """
    # Setup
    start_time = time.time()
    print(f"Starting Lewellen and Lewellen (2016) replication pipeline")
    print(f"Time period: {start_year} to {end_year}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    # Check data directory
    data_dir = check_data_directory()
    
    # Step 1: Data Collection
    raw_data_path = os.path.join(data_dir, 'raw_merged_data.csv')
    if os.path.exists(raw_data_path) and not force_refresh:
        print(f"Raw data file found at {raw_data_path}")
        print("Loading existing raw data...")
        raw_data = pd.read_csv(raw_data_path)
        print(f"Loaded {len(raw_data)} observations from raw data file")
    else:
        print("Collecting raw data from WRDS...")
        raw_data = collect_raw_data(start_year, end_year)
        if raw_data is None:
            print("Failed to collect raw data. Exiting pipeline.")
            return None
    
    print("-" * 80)
    step1_time = time.time()
    print(f"Step 1 (Data Collection) completed in {(step1_time - start_time) / 60:.2f} minutes")
    
    # Step 2: Variable Construction
    constructed_data_path = os.path.join(data_dir, 'constructed_variables.csv')
    if os.path.exists(constructed_data_path) and not force_refresh:
        print(f"Constructed variables file found at {constructed_data_path}")
        print("Loading existing constructed variables...")
        constructed_data = pd.read_csv(constructed_data_path)
        print(f"Loaded {len(constructed_data)} observations with constructed variables")
    else:
        print("Constructing variables...")
        constructed_data = construct_variables(raw_data)
        save_constructed_data(constructed_data)
    
    print("-" * 80)
    step2_time = time.time()
    print(f"Step 2 (Variable Construction) completed in {(step2_time - step1_time) / 60:.2f} minutes")
    
    # Step 3: Sample Preparation
    regression_sample_path = os.path.join(data_dir, 'regression_sample.csv')
    if os.path.exists(regression_sample_path) and not force_refresh:
        print(f"Regression sample file found at {regression_sample_path}")
        print("Loading existing regression sample...")
        regression_sample = pd.read_csv(regression_sample_path)
        print(f"Loaded {len(regression_sample)} observations in regression sample")
    else:
        print("Preparing regression sample...")
        regression_sample = prepare_regression_sample(constructed_data)
        save_regression_sample(regression_sample)
    
    print("-" * 80)
    step3_time = time.time()
    print(f"Step 3 (Sample Preparation) completed in {(step3_time - step2_time) / 60:.2f} minutes")
    
    # Pipeline Summary
    print("\nPipeline Summary:")
    print(f"Total runtime: {(step3_time - start_time) / 60:.2f} minutes")
    print(f"Raw observations: {len(raw_data)}")
    print(f"Observations with constructed variables: {len(constructed_data)}")
    print(f"Final regression sample observations: {len(regression_sample)}")
    print(f"Number of unique firms in final sample: {regression_sample['gvkey'].nunique()}")
    print(f"Sample period: {regression_sample['fyear'].min()} to {regression_sample['fyear'].max()}")
    
    return regression_sample


if __name__ == '__main__':
    # Run the pipeline with default parameters
    regression_data = run_pipeline(start_year=1971, end_year=2023, force_refresh=False)
    
    # If you want to rerun with different parameters, uncomment and modify:
    # regression_data = run_pipeline(start_year=1980, end_year=2020, force_refresh=True) 