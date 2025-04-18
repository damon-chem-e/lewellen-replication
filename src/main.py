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

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import modules
from src.data_collection import collect_raw_data
from src.variable_construction import construct_variables, save_constructed_data
from src.sample_preparation import prepare_regression_sample, save_regression_sample
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
        data_path = os.path.join(data_dir, 'regression_sample.csv')
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run main.py first to prepare the data.")
        return False
    
    # Create results directories
    tables_dir, plots_dir = create_results_directory(output_dir)
    
    # Load data
    print(f"\nLoading regression sample from {data_path}")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} observations")
    
    # Run analyses
    start_time = datetime.now()
    print(f"Analysis started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    analyze_sample_characteristics(data, plots_dir)
    analyze_constraint_groups(data, plots_dir)
    run_baseline_analysis(data, plots_dir)
    analyze_cash_flow_uses(data, plots_dir)
    analyze_investment_financing(data, plots_dir)
    compare_with_original_paper(data, tables_dir)
    run_full_regression_analysis(data, tables_dir, plots_dir)
    
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
        '--start_year', 
        type=int, 
        default=1971,
        help='Start year for data collection (default: 1971)'
    )
    parser.add_argument(
        '--end_year', 
        type=int, 
        default=2023,
        help='End year for data collection (default: 2023)'
    )
    parser.add_argument(
        '--force_refresh', 
        action='store_true',
        help='Force refresh of all data files even if they exist'
    )
    parser.add_argument(
        '--run_analysis', 
        action='store_true',
        help='Run analysis after preparing the data'
    )
    parser.add_argument(
        '--analysis_only', 
        action='store_true',
        help='Skip data preparation and only run analysis'
    )
    
    args = parser.parse_args()
    
    # Run analysis only if requested
    if args.analysis_only:
        print("Skipping data preparation and running analysis only")
        run_analysis()
        return
    
    # Run data preparation pipeline
    regression_data = run_pipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        force_refresh=args.force_refresh
    )
    
    # Run analysis if requested
    if args.run_analysis and regression_data is not None:
        run_analysis()


if __name__ == '__main__':
    main() 