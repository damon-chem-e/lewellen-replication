"""
Main script for Lewellen and Lewellen (2016) Replication Study.

This script orchestrates the entire data preparation pipeline by calling
modules for data collection, variable construction, and sample preparation.
It can also run the analysis after preparing the data.

Usage:
    python main.py [--start_year YEAR] [--end_year YEAR] [--force_refresh] [--run_analysis]
"""

import os
import sys
import pandas as pd
import time
import argparse
from datetime import datetime
try:
    from src.data_collection import collect_raw_data
    from src.variable_construction import construct_variables, save_constructed_data
    from src.sample_preparation import prepare_regression_sample, save_regression_sample
except ImportError:
    from data_collection import collect_raw_data
    from variable_construction import construct_variables, save_constructed_data
    from sample_preparation import prepare_regression_sample, save_regression_sample

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import analysis modules only - we'll conditionally import data modules later
from src.analyze_results import (
    create_results_directory,
    analyze_sample_characteristics,
    analyze_constraint_groups,
    run_baseline_analysis,
    analyze_cash_flow_uses,
    analyze_investment_financing,
    compare_with_original_paper,
    run_full_regression_analysis
)


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


def run_pipeline(start_year=1971, end_year=2023, force_refresh=False, skip_collection=False):
    """
    Run the full data preparation pipeline.
    
    Args:
        start_year (int): Start year for data collection
        end_year (int): End year for data collection
        force_refresh (bool): If True, rerun all steps even if files exist
        skip_collection (bool): If True, skip raw data collection and use existing file
        
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
    
    # Step 1: Data Collection / Loading
    raw_data_parquet_path = os.path.join(data_dir, 'raw_merged_data.parquet')
    raw_data_csv_path = os.path.join(data_dir, 'raw_merged_data.csv')
    raw_data = None

    if skip_collection:
        print("Skipping data collection step as per --skip_collection flag.")
        if os.path.exists(raw_data_parquet_path):
            print(f"Loading existing raw data from {raw_data_parquet_path}...")
            raw_data = pd.read_parquet(raw_data_parquet_path)
            print(f"Loaded {len(raw_data)} observations from Parquet file.")
        elif os.path.exists(raw_data_csv_path):
            print(f"Parquet file not found. Loading existing raw data from {raw_data_csv_path}...")
            raw_data = pd.read_csv(raw_data_csv_path)
            print(f"Loaded {len(raw_data)} observations from CSV file.")
        else:
            print(f"Error: --skip_collection specified, but no raw data file found at {raw_data_parquet_path} or {raw_data_csv_path}.")
            print("Please ensure one of these files exists in the 'data' directory or run without --skip_collection.")
            return None
    else:
        # Original logic: try to load CSV if not force_refresh, otherwise collect.
        # The project seems to be moving towards Parquet, so eventually collect_raw_data should save to Parquet.
        # For now, matching existing behavior for the collection path.
        if os.path.exists(raw_data_csv_path) and not force_refresh:
            print(f"Raw data file found at {raw_data_csv_path}")
            print("Loading existing raw data...")
            raw_data = pd.read_csv(raw_data_csv_path)
            print(f"Loaded {len(raw_data)} observations from raw data file")
        else:
            if force_refresh and os.path.exists(raw_data_csv_path):
                print(f"Force refresh is True. Will re-collect raw data, ignoring existing {raw_data_csv_path}.")
            elif force_refresh and os.path.exists(raw_data_parquet_path):
                 print(f"Force refresh is True. Will re-collect raw data, ignoring existing {raw_data_parquet_path}.")
            print("Collecting raw data from WRDS...")
            # collect_raw_data is imported in main() before run_pipeline is called
            raw_data = collect_raw_data(start_year, end_year)
            if raw_data is None:
                print("Failed to collect raw data. Exiting pipeline.")
                return None
            # Assuming collect_raw_data saves its output if it runs, e.g. as raw_merged_data.csv or .parquet
            # For now, the script implies it saves to CSV if it collects new data.
            # Consider standardizing collect_raw_data to save to parquet and then this logic can simplify.

    if raw_data is None: # Should have been caught earlier if skip_collection failed, but as a safeguard.
        print("Raw data is not available. Exiting pipeline.")
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


def run_analysis(data_path=None, output_dir=None):
    """
    Run analysis on the prepared regression sample.
    
    Args:
        data_path (str): Path to regression sample data
        output_dir (str): Directory to save results
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Set default paths if not provided
    if data_path is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        data_path = os.path.join(data_dir, 'raw_merged_data.parquet')
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please check that the raw data file exists.")
        return False
    
    # Import required modules for data processing
    from src.variable_construction import construct_variables
    from src.sample_preparation import prepare_regression_sample
    
    # Create results directories
    tables_dir, plots_dir = create_results_directory(output_dir)
    
    # Load data with optimized memory usage - select only required columns
    print(f"\nLoading raw data from {data_path}")
    
    # First, let's check what columns are available
    columns_preview = pd.read_parquet(data_path, columns=None)
    print(f"Available columns: {columns_preview.columns.tolist()}")
    
    # Define required columns based on the analysis
    required_columns = [
        'gvkey', 'datadate', 'fyear', 'at', 'lct', 'dlc', 'ibc', 'xidoc', 'dpc', 
        'txdc', 'esubc', 'sppiv', 'fopo', 'capx', 'aqc', 'ivch', 'siv', 'ppent',
        'che', 'act', 'dltt', 'lt', 'ceq', 'pstk', 're', 'dvc', 'dvp', 'sale',
        'exchg', 'prc', 'shrout', 'sic', 'permno', 'oiadp', 'oancf', 'ni','ret'
    ]
    
    # Filter to only use available columns
    available_columns = [col for col in required_columns if col in columns_preview.columns]
    
    try:
        # Read only the required columns
        data = pd.read_parquet(data_path, columns=available_columns)
        print(f"Loaded {len(data)} observations with {len(available_columns)} columns")
    except Exception as e:
        print(f"Error loading specific columns, falling back to loading all columns: {e}")
        data = pd.read_parquet(data_path)
        print(f"Loaded {len(data)} observations with all columns")
    
    # Process data
    print("Constructing variables...")
    constructed_data = construct_variables(data)
    print(f"Constructed variables for {len(constructed_data)} observations")
    
    # Clear original data to free memory
    del data
    
    print("Preparing regression sample...")
    regression_sample = prepare_regression_sample(constructed_data)
    print(f"Final regression sample has {len(regression_sample)} observations")
    
    # Clear constructed data to free memory
    del constructed_data
    
    # Run analyses
    start_time = datetime.now()
    print(f"Analysis started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    # Run analyses individually to manage memory better
    try:
        print("Analyzing sample characteristics...")
        analyze_sample_characteristics(regression_sample, plots_dir)
    except Exception as e:
        print(f"Error in sample characteristics analysis: {e}")
    
    try:
        print("Analyzing constraint groups...")
        analyze_constraint_groups(regression_sample, plots_dir)
    except Exception as e:
        print(f"Error in constraint groups analysis: {e}")
    
    try:
        print("Running baseline analysis...")
        run_baseline_analysis(regression_sample, plots_dir)
    except Exception as e:
        print(f"Error in baseline analysis: {e}")
    
    try:
        print("Analyzing cash flow uses...")
        analyze_cash_flow_uses(regression_sample, plots_dir)
    except Exception as e:
        print(f"Error in cash flow uses analysis: {e}")
    
    try:
        print("Analyzing investment financing...")
        analyze_investment_financing(regression_sample, plots_dir)
    except Exception as e:
        print(f"Error in investment financing analysis: {e}")
    
    try:
        print("Comparing with original paper...")
        compare_with_original_paper(regression_sample, tables_dir)
    except Exception as e:
        print(f"Error in comparison with original paper: {e}")
    
    try:
        print("Running full regression analysis...")
        run_full_regression_analysis(regression_sample, tables_dir, plots_dir)
    except Exception as e:
        print(f"Error in full regression analysis: {e}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    print("-" * 80)
    print(f"Analysis completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {duration:.2f} minutes")
    print(f"Results saved to {output_dir}")
    
    return True


def main():
    """
    Parse command line arguments and run the appropriate pipeline steps.
    """
    parser = argparse.ArgumentParser(
        description='Run Lewellen and Lewellen (2016) replication study'
    )
    parser.add_argument(
        '--start-year', 
        type=int, 
        default=1971,
        help='Start year for data collection (default: 1971)'
    )
    parser.add_argument(
        '--end-year', 
        type=int, 
        default=2023,
        help='End year for data collection (default: 2023)'
    )
    parser.add_argument(
        '--force-refresh', 
        action='store_true',
        help='Force refresh of all data files even if they exist'
    )
    parser.add_argument(
        '--run-analysis', 
        action='store_true',
        help='Run analysis after preparing the data'
    )
    parser.add_argument(
        '--analysis-only', 
        action='store_true',
        help='Skip data preparation and only run analysis. Assumes raw_merged_data.parquet exists.'
    )
    parser.add_argument(
        '--skip-collection',
        action='store_true',
        help='Skip raw data collection. Assumes raw_merged_data.parquet (or .csv) exists and proceeds with variable construction and sample prep.'
    )
    
    args = parser.parse_args()
    
    # Run analysis only if requested
    if args.analysis_only:
        print("Skipping data preparation and running analysis only")
        run_analysis()
        return
    
    # Import data modules only when needed
    from src.data_collection import collect_raw_data
    from src.variable_construction import construct_variables, save_constructed_data
    from src.sample_preparation import prepare_regression_sample, save_regression_sample
    
    # Run data preparation pipeline
    regression_data = run_pipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        force_refresh=args.force_refresh,
        skip_collection=args.skip_collection
    )
    
    # Run analysis if requested
    if args.run_analysis and regression_data is not None:
        run_analysis()


if __name__ == '__main__':
    main() 