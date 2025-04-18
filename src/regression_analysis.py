"""
Regression Analysis Module for Lewellen and Lewellen (2016) Replication.

This module implements the OLS and IV regressions specified in the original study,
focusing on investment-cash flow sensitivities and measurement error correction.
"""

import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns


def run_ols_regression(data, dependent_var, independent_vars, fe_firm=True, fe_year=True):
    """
    Run OLS regression with firm and year fixed effects.
    
    Args:
        data (pd.DataFrame): Regression sample data
        dependent_var (str): Name of dependent variable
        independent_vars (list): List of independent variable names
        fe_firm (bool): Include firm fixed effects
        fe_year (bool): Include year fixed effects
        
    Returns:
        sm.regression.linear_model.RegressionResultsWrapper: Regression results
    """
    # Create formula for regression
    X = independent_vars.copy()
    
    # Add fixed effects if requested
    if fe_firm:
        # Create firm dummies (but exclude from formula - will be added as dummy variables)
        firm_dummies = pd.get_dummies(data['firm_id'], prefix='firm', drop_first=True)
        X_with_fe = pd.concat([data[X], firm_dummies], axis=1)
    else:
        X_with_fe = data[X].copy()
    
    if fe_year:
        # Create year dummies
        year_dummies = pd.get_dummies(data['year_id'], prefix='year', drop_first=True)
        X_with_fe = pd.concat([X_with_fe, year_dummies], axis=1)
    
    # Add constant
    X_with_fe = sm.add_constant(X_with_fe)
    
    # Run regression
    model = sm.OLS(data[dependent_var], X_with_fe)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': data['gvkey']})
    
    # Print summary of key coefficients (not all the FE)
    print(f"\nOLS Regression Results for {dependent_var}")
    print("=" * 50)
    print(f"R-squared: {results.rsquared:.4f}")
    print(f"Observations: {results.nobs}")
    
    # Print coefficients and t-stats for main variables
    print("\nCoefficients:")
    for var in independent_vars:
        if var in results.params.index:
            coef = results.params[var]
            t_stat = results.tvalues[var]
            print(f"{var}: {coef:.4f} (t={t_stat:.2f})")
    
    return results


def run_first_stage_regression(data, y_var, instruments, controls=None):
    """
    Run first-stage regression for IV estimation.
    
    Args:
        data (pd.DataFrame): Regression sample data
        y_var (str): Endogenous variable (MB ratio)
        instruments (list): List of instruments (cash flow, returns)
        controls (list): Additional control variables
        
    Returns:
        sm.regression.linear_model.RegressionResultsWrapper: First-stage results
        pd.Series: Fitted values for use in second stage
    """
    # Combine instruments and controls
    X_vars = instruments.copy()
    if controls is not None:
        X_vars.extend(controls)
    
    # Create firm and year fixed effects
    firm_dummies = pd.get_dummies(data['firm_id'], prefix='firm', drop_first=True)
    year_dummies = pd.get_dummies(data['year_id'], prefix='year', drop_first=True)
    
    # Combine all X variables
    X = pd.concat([data[X_vars], firm_dummies, year_dummies], axis=1)
    X = sm.add_constant(X)
    
    # Run regression
    model = sm.OLS(data[y_var], X)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': data['gvkey']})
    
    # Generate fitted values
    fitted_values = results.predict(X)
    
    # Print summary of key coefficients
    print(f"\nFirst-Stage Regression Results for {y_var}")
    print("=" * 50)
    print(f"R-squared: {results.rsquared:.4f}")
    print(f"F-statistic (instruments): {results.fvalue:.2f}")
    print(f"Observations: {results.nobs}")
    
    # Print coefficients and t-stats for instruments
    print("\nInstrument Coefficients:")
    for var in instruments:
        if var in results.params.index:
            coef = results.params[var]
            t_stat = results.tvalues[var]
            print(f"{var}: {coef:.4f} (t={t_stat:.2f})")
    
    return results, fitted_values


def run_iv_regression(data, dependent_var, endogenous_var, instruments, controls=None):
    """
    Run IV 2SLS regression with firm and year fixed effects.
    
    Args:
        data (pd.DataFrame): Regression sample data
        dependent_var (str): Dependent variable (e.g., investment measure)
        endogenous_var (str): Endogenous variable (MB ratio)
        instruments (list): List of instruments (cash flow, returns)
        controls (list): Additional control variables
        
    Returns:
        tuple: (First-stage results, Second-stage results)
    """
    # Step 1: First-stage regression
    first_stage, mb_fitted = run_first_stage_regression(
        data, endogenous_var, instruments, controls
    )
    
    # Add fitted values to data
    data_copy = data.copy()
    data_copy[f"{endogenous_var}_fitted"] = mb_fitted
    
    # Step 2: Second-stage regression
    # Replace endogenous variable with fitted values
    second_stage_vars = []
    if controls is not None:
        second_stage_vars.extend(controls)
    
    # Add fitted value instead of original endogenous variable
    second_stage_vars.append(f"{endogenous_var}_fitted")
    
    # Run second stage regression
    second_stage = run_ols_regression(
        data_copy, dependent_var, second_stage_vars, fe_firm=True, fe_year=True
    )
    
    return first_stage, second_stage


def run_table3_regressions(data):
    """
    Replicate Table 3 from the original paper: OLS investment-cash flow regressions.
    
    Args:
        data (pd.DataFrame): Regression sample data
        
    Returns:
        dict: Dictionary of regression results
    """
    print("\nReplicating Table 3: OLS Investment-Cash Flow Regressions")
    print("=" * 70)
    
    # Dependent variables (uses of cash flow)
    dependent_vars = [
        'delta_cash_scaled', 'delta_nwc_scaled', 
        'capx1_scaled', 'capx2_scaled', 'capx3_scaled',
        'delta_debt_scaled', 'issues_scaled', 'div_scaled'
    ]
    
    # Store results
    table3_results = {}
    
    # Model 1: CF_t and MB_t-1
    print("\nModel 1: CF_t and MB_t-1")
    model1_results = {}
    for dep_var in dependent_vars:
        result = run_ols_regression(
            data, dep_var, ['cash_flow_scaled', 'mb_lag']
        )
        model1_results[dep_var] = result
    
    table3_results['model1'] = model1_results
    
    # Model 2: Add CF_t-1
    print("\nModel 2: Add CF_t-1")
    model2_results = {}
    for dep_var in dependent_vars:
        result = run_ols_regression(
            data, dep_var, ['cash_flow_scaled', 'cash_flow_scaled_lag', 'mb_lag']
        )
        model2_results[dep_var] = result
    
    table3_results['model2'] = model2_results
    
    # Model 3: Add CASH_t-1 and DEBT_t-1
    print("\nModel 3: Add CASH_t-1 and DEBT_t-1")
    model3_results = {}
    for dep_var in dependent_vars:
        result = run_ols_regression(
            data, dep_var, 
            ['cash_flow_scaled', 'cash_flow_scaled_lag', 'mb_lag', 'cash_lag', 'debt_lag']
        )
        model3_results[dep_var] = result
    
    table3_results['model3'] = model3_results
    
    return table3_results


def run_table4_regressions(data):
    """
    Replicate Table 4 from the original paper: OLS regressions for constrained vs. unconstrained firms.
    
    Args:
        data (pd.DataFrame): Regression sample data
        
    Returns:
        dict: Dictionary of regression results for constrained and unconstrained subsamples
    """
    print("\nReplicating Table 4: OLS Regressions by Financial Constraint")
    print("=" * 70)
    
    # Dependent variables (uses of cash flow)
    dependent_vars = [
        'delta_cash_scaled', 'delta_nwc_scaled', 
        'capx1_scaled', 'capx2_scaled', 'capx3_scaled',
        'delta_debt_scaled', 'issues_scaled', 'div_scaled'
    ]
    
    # Create constrained and unconstrained subsamples
    constrained_data = data[data['constrained'] == 1].copy()
    unconstrained_data = data[data['unconstrained'] == 1].copy()
    
    print(f"Constrained subsample: {len(constrained_data)} observations")
    print(f"Unconstrained subsample: {len(unconstrained_data)} observations")
    
    # Store results
    table4_results = {'constrained': {}, 'unconstrained': {}}
    
    # Model 1 for constrained firms
    print("\nConstrained Firms: CF_t and MB_t-1")
    for dep_var in dependent_vars:
        result = run_ols_regression(
            constrained_data, dep_var, ['cash_flow_scaled', 'mb_lag']
        )
        table4_results['constrained'][dep_var] = result
    
    # Model 1 for unconstrained firms
    print("\nUnconstrained Firms: CF_t and MB_t-1")
    for dep_var in dependent_vars:
        result = run_ols_regression(
            unconstrained_data, dep_var, ['cash_flow_scaled', 'mb_lag']
        )
        table4_results['unconstrained'][dep_var] = result
    
    return table4_results


def run_table6_regressions(data):
    """
    Replicate Table 6 from the original paper: IV investment-cash flow regressions.
    
    Args:
        data (pd.DataFrame): Regression sample data
        
    Returns:
        dict: Dictionary of IV regression results
    """
    print("\nReplicating Table 6: IV Investment-Cash Flow Regressions")
    print("=" * 70)
    
    # Dependent variables (uses of cash flow)
    dependent_vars = [
        'delta_cash_scaled', 'delta_nwc_scaled', 
        'capx1_scaled', 'capx2_scaled', 'capx3_scaled',
        'delta_debt_scaled', 'issues_scaled', 'div_scaled'
    ]
    
    # Store results
    table6_results = {}
    
    # Model 1: CF_t and MB_t-1 (instrumented by CF_t and return lags)
    print("\nModel 1: CF_t and MB_t-1 (IV)")
    model1_results = {}
    for dep_var in dependent_vars:
        # Run IV regression using returns as instruments for MB
        instruments = ['cash_flow_scaled', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
        first_stage, second_stage = run_iv_regression(
            data, dep_var, 'mb_lag', instruments, ['cash_flow_scaled']
        )
        model1_results[dep_var] = (first_stage, second_stage)
    
    table6_results['model1'] = model1_results
    
    # Model 2: Add CF_t-1
    print("\nModel 2: Add CF_t-1 (IV)")
    model2_results = {}
    for dep_var in dependent_vars:
        instruments = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
        first_stage, second_stage = run_iv_regression(
            data, dep_var, 'mb_lag', instruments, 
            ['cash_flow_scaled', 'cash_flow_scaled_lag']
        )
        model2_results[dep_var] = (first_stage, second_stage)
    
    table6_results['model2'] = model2_results
    
    # Model 3: Add CASH_t-1 and DEBT_t-1
    print("\nModel 3: Add CASH_t-1 and DEBT_t-1 (IV)")
    model3_results = {}
    for dep_var in dependent_vars:
        instruments = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
        controls = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'cash_lag', 'debt_lag']
        first_stage, second_stage = run_iv_regression(
            data, dep_var, 'mb_lag', instruments, controls
        )
        model3_results[dep_var] = (first_stage, second_stage)
    
    table6_results['model3'] = model3_results
    
    return table6_results


def run_table7_regressions(data):
    """
    Replicate Table 7 from the original paper: IV regressions for constrained vs. unconstrained firms.
    
    Args:
        data (pd.DataFrame): Regression sample data
        
    Returns:
        dict: Dictionary of IV regression results for constrained and unconstrained subsamples
    """
    print("\nReplicating Table 7: IV Regressions by Financial Constraint")
    print("=" * 70)
    
    # Focus on CAPX3 (fixed investment) as the key dependent variable
    dep_var = 'capx3_scaled'
    
    # Create constrained and unconstrained subsamples
    constrained_data = data[data['constrained'] == 1].copy()
    unconstrained_data = data[data['unconstrained'] == 1].copy()
    
    print(f"Constrained subsample: {len(constrained_data)} observations")
    print(f"Unconstrained subsample: {len(unconstrained_data)} observations")
    
    # Store results
    table7_results = {'constrained': {}, 'unconstrained': {}}
    
    # Model 1: CF_t and MB_t-1 (IV)
    print("\nModel 1: CF_t and MB_t-1 (IV)")
    
    # Constrained firms
    instruments = ['cash_flow_scaled', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
    first_stage, second_stage = run_iv_regression(
        constrained_data, dep_var, 'mb_lag', instruments, ['cash_flow_scaled']
    )
    table7_results['constrained']['model1'] = (first_stage, second_stage)
    
    # Unconstrained firms
    first_stage, second_stage = run_iv_regression(
        unconstrained_data, dep_var, 'mb_lag', instruments, ['cash_flow_scaled']
    )
    table7_results['unconstrained']['model1'] = (first_stage, second_stage)
    
    # Model 2: Add CF_t-1 (IV)
    print("\nModel 2: Add CF_t-1 (IV)")
    
    # Constrained firms
    instruments = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
    first_stage, second_stage = run_iv_regression(
        constrained_data, dep_var, 'mb_lag', instruments, 
        ['cash_flow_scaled', 'cash_flow_scaled_lag']
    )
    table7_results['constrained']['model2'] = (first_stage, second_stage)
    
    # Unconstrained firms
    first_stage, second_stage = run_iv_regression(
        unconstrained_data, dep_var, 'mb_lag', instruments, 
        ['cash_flow_scaled', 'cash_flow_scaled_lag']
    )
    table7_results['unconstrained']['model2'] = (first_stage, second_stage)
    
    # Model 3: Add CASH_t-1 and DEBT_t-1 (IV)
    print("\nModel 3: Add CASH_t-1 and DEBT_t-1 (IV)")
    
    # Constrained firms
    instruments = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
    controls = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'cash_lag', 'debt_lag']
    first_stage, second_stage = run_iv_regression(
        constrained_data, dep_var, 'mb_lag', instruments, controls
    )
    table7_results['constrained']['model3'] = (first_stage, second_stage)
    
    # Unconstrained firms
    first_stage, second_stage = run_iv_regression(
        unconstrained_data, dep_var, 'mb_lag', instruments, controls
    )
    table7_results['unconstrained']['model3'] = (first_stage, second_stage)
    
    return table7_results


def create_summary_table(results_dict, model_name, key_vars):
    """
    Create a summary table of regression results.
    
    Args:
        results_dict (dict): Dictionary of regression results
        model_name (str): Name of model to summarize
        key_vars (list): List of key variables to include in summary
        
    Returns:
        pd.DataFrame: Summary table
    """
    # Extract results for the specified model
    model_results = results_dict[model_name]
    
    # Initialize dictionaries to store coefficients and t-stats
    coefs = {}
    t_stats = {}
    r2_values = {}
    n_obs = {}
    
    # Extract results for each dependent variable
    for dep_var, result in model_results.items():
        coefs[dep_var] = {}
        t_stats[dep_var] = {}
        
        # Store R-squared and observation count
        r2_values[dep_var] = result.rsquared
        n_obs[dep_var] = result.nobs
        
        # Store coefficients and t-stats for key variables
        for var in key_vars:
            if var in result.params.index:
                coefs[dep_var][var] = result.params[var]
                t_stats[dep_var][var] = result.tvalues[var]
    
    # Create summary dataframes
    coef_df = pd.DataFrame(coefs).T
    tstat_df = pd.DataFrame(t_stats).T
    
    # Add R-squared and observation count
    coef_df['R-squared'] = pd.Series(r2_values)
    coef_df['N'] = pd.Series(n_obs)
    
    return coef_df, tstat_df


def save_regression_results(results_dict, output_dir):
    """
    Save regression results to CSV files.
    
    Args:
        results_dict (dict): Dictionary of regression results
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract and save each table's results
    for table_name, table_results in results_dict.items():
        # Create a descriptive filename
        filename = f"{table_name}_results.csv"
        file_path = os.path.join(output_dir, filename)
        
        # Convert results to DataFrame and save
        if isinstance(table_results, dict):
            # Handle nested dictionaries (tables with multiple models)
            summary_data = []
            for model_name, model_results in table_results.items():
                # Extract key information from each model
                model_summary = {
                    'model': model_name,
                    'r2': model_results.rsquared,
                    'nobs': model_results.nobs
                }
                # Add coefficients and t-stats for key variables
                for var in model_results.params.index:
                    if var != 'const' and not var.startswith(('firm', 'year')):
                        model_summary[f"{var}_coef"] = model_results.params[var]
                        model_summary[f"{var}_tstat"] = model_results.tvalues[var]
                
                summary_data.append(model_summary)
            
            # Create and save DataFrame
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(file_path, index=False)
            print(f"Saved {filename}")
    
    print(f"All regression results saved to {output_dir}")


def plot_investment_cash_flow_sensitivity(data, output_dir):
    """
    Create plots of investment-cash flow sensitivity.
    
    Args:
        data (pd.DataFrame): Regression sample data
        output_dir (str): Directory to save plots
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Plot for constrained vs unconstrained firms
    plt.subplot(2, 1, 1)
    sns.regplot(x='cash_flow_scaled', y='capx3_scaled', 
                data=data[data['constrained'] == 1], 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'},
                label='Constrained')
    sns.regplot(x='cash_flow_scaled', y='capx3_scaled', 
                data=data[data['unconstrained'] == 1], 
                scatter_kws={'alpha':0.3}, line_kws={'color':'blue'},
                label='Unconstrained')
    plt.title('Investment-Cash Flow Sensitivity: Constrained vs. Unconstrained Firms')
    plt.xlabel('Cash Flow / Net Assets')
    plt.ylabel('Investment (CAPX3) / Net Assets')
    plt.legend()
    
    # Plot for high vs low Tobin's q firms
    plt.subplot(2, 1, 2)
    high_q = data['mb_lag'] > data['mb_lag'].median()
    sns.regplot(x='cash_flow_scaled', y='capx3_scaled', 
                data=data[high_q], 
                scatter_kws={'alpha':0.3}, line_kws={'color':'green'},
                label='High Q')
    sns.regplot(x='cash_flow_scaled', y='capx3_scaled', 
                data=data[~high_q], 
                scatter_kws={'alpha':0.3}, line_kws={'color':'purple'},
                label='Low Q')
    plt.title('Investment-Cash Flow Sensitivity: High vs. Low Q Firms')
    plt.xlabel('Cash Flow / Net Assets')
    plt.ylabel('Investment (CAPX3) / Net Assets')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'investment_cash_flow_sensitivity.png'), dpi=300)
    plt.close()
    
    print("Investment-cash flow sensitivity plots saved")


def run_all_regressions(data_path, output_dir='../results'):
    """
    Run all regression analyses for the Lewellen and Lewellen replication.
    
    Args:
        data_path (str): Path to regression sample data
        output_dir (str): Directory to save results
        
    Returns:
        dict: Dictionary of all regression results
    """
    # Load regression sample
    print(f"Loading regression sample from {data_path}")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} observations")
    
    # Run regressions from Tables 3, 4, 6, and 7
    results = {}
    
    # Table 3: OLS investment-cash flow regressions
    results['table3'] = run_table3_regressions(data)
    
    # Table 4: OLS regressions for constrained vs. unconstrained firms
    results['table4'] = run_table4_regressions(data)
    
    # Table 6: IV investment-cash flow regressions
    results['table6'] = run_table6_regressions(data)
    
    # Table 7: IV regressions for constrained vs. unconstrained firms
    results['table7'] = run_table7_regressions(data)
    
    # Create plots
    plot_investment_cash_flow_sensitivity(data, output_dir)
    
    # Save results
    save_regression_results(results, output_dir)
    
    return results


if __name__ == '__main__':
    # Set up paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    data_path = os.path.join(data_dir, 'regression_sample.csv')
    
    # Run regressions
    if os.path.exists(data_path):
        results = run_all_regressions(data_path, output_dir)
    else:
        print(f"Regression sample not found at {data_path}")
        print("Please run src/main.py first to prepare the data.") 