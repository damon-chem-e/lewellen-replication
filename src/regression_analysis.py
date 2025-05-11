"""
Regression Analysis Module for Lewellen and Lewellen (2016) Replication.

This module implements the OLS and IV regressions specified in the original study,
focusing on investment-cash flow sensitivities and measurement error correction,
now primarily using R's felm via the octopus wrapper for efficiency with fixed effects.
"""

import pandas as pd
import numpy as np
import os
# No longer using statsmodels directly for FE regressions here
# import statsmodels.api as sm
# from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns
from octopus.octopus import octopus # Import octopus

# Initialize Octopus instance for use in this module
# Assumes regression_analysis.py is in 'src/' and octo_reg.R is in 'octopus/' at project root
try:
    OCTOPUS_INSTANCE = octopus()
    OCTOPUS_INSTANCE.r_script_path = os.path.join(os.path.dirname(__file__), '..', 'octopus', 'octo_reg.R')
    # OCTOPUS_INSTANCE.workdir will use its default (None), leading to temp dirs per call.
except Exception as e:
    print(f"Error initializing Octopus: {e}")
    OCTOPUS_INSTANCE = None

# The old run_ols_regression, run_first_stage_regression, and run_iv_regression
# are no longer needed as their functionality with high-cardinality FEs
# is replaced by calls to R's felm via OCTOPUS_INSTANCE.

def run_simple_ols_regression(data, dependent_var, independent_vars, cluster_col='gvkey'):
    """
    Run a simple OLS regression using statsmodels.
    This version does NOT handle fixed effects automatically.
    It supports clustering of standard errors.
    
    Args:
        data (pd.DataFrame): Regression sample data.
        dependent_var (str): Name of the dependent variable.
        independent_vars (list): List of independent variable names.
        cluster_col (str, optional): Column name to cluster standard errors by. Defaults to 'gvkey'.
                                     If None, standard errors are not clustered.
        
    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: Regression results.
    """
    import statsmodels.api as sm # Import locally as it's only for this function now

    X_df = data[independent_vars].copy()
    y_df = data[dependent_var].copy()

    # Convert boolean columns to int (0 or 1)
    for col in X_df.columns:
        if X_df[col].dtype == 'bool':
            X_df[col] = X_df[col].astype(int)

    # Add constant
    X_with_const = sm.add_constant(X_df)
    
    # Drop rows with NaNs in X or y, as sm.OLS requires this.
    # Ensure alignment between X and y after dropping NaNs.
    combined_df = pd.concat([y_df, X_with_const], axis=1).dropna()
    y_clean = combined_df[dependent_var]
    X_clean = combined_df[X_with_const.columns]

    if y_clean.empty or X_clean.empty:
        print(f"Warning: Not enough data for {dependent_var} ~ {' + '.join(independent_vars)} after NaN removal.")
        return None # Or raise an error

    model = sm.OLS(y_clean, X_clean)
    
    if cluster_col and cluster_col in data.columns:
        # Align cluster groups with the cleaned data
        cluster_groups = data.loc[X_clean.index, cluster_col]
        results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_groups})
    else:
        if cluster_col:
            print(f"Warning: Cluster column '{cluster_col}' not found or not specified. Fitting with non-robust SEs.")
        results = model.fit() # Standard errors
    
    # Optional: Print a brief summary
    print(f"\nSimple OLS Results for {dependent_var} ~ {' + '.join(independent_vars)}")
    print(f"R-squared: {results.rsquared:.4f}, N.obs: {results.nobs}")
    # for var in independent_vars:
    #     if var in results.params.index:
    #         print(f"  {var}: {results.params[var]:.4f} (t={results.tvalues[var]:.2f})")
            
    return results

def run_table3_regressions(data):
    """
    Replicate Table 3: OLS investment-cash flow regressions using felm.
    """
    if OCTOPUS_INSTANCE is None:
        raise RuntimeError("Octopus instance not initialized. Cannot run R regressions.")

    print("\nReplicating Table 3: OLS Investment-Cash Flow Regressions (via R felm)")
    print("=" * 70)
    
    dependent_vars = [
        'delta_cash_scaled', 'delta_nwc_scaled', 
        'capx1_scaled', 'capx2_scaled', 'capx3_scaled',
        'delta_debt_scaled', 'issues_scaled', 'div_scaled'
    ]
    
    table3_results = {}
    fe_str = "| factor(gvkey) + factor(fyear) | 0 | gvkey + fyear"

    # Model 1: CF_t and MB_t-1
    print("\nModel 1: CF_t and MB_t-1")
    model1_vars = ['cash_flow_scaled', 'mb_lag']
    model1_rhs = " + ".join(model1_vars)
    model1_results_dict = {}
    for dep_var in dependent_vars:
        formula = f"{dep_var} ~ {model1_rhs} {fe_str}"
        try:
            result_df = OCTOPUS_INSTANCE.run_regressions(
                df=data, regression_commands=['felm'],
                regression_specifications=[formula], stargazer_specs=[]
            )
            model1_results_dict[dep_var] = result_df
        except Exception as e:
            print(f"Error running R regression for {dep_var} ~ {model1_rhs}: {e}")
            model1_results_dict[dep_var] = pd.DataFrame() # Store empty df on error
    table3_results['model1'] = model1_results_dict
    
    # Model 2: Add CF_t-1
    print("\nModel 2: Add CF_t-1")
    model2_vars = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'mb_lag']
    model2_rhs = " + ".join(model2_vars)
    model2_results_dict = {}
    for dep_var in dependent_vars:
        formula = f"{dep_var} ~ {model2_rhs} {fe_str}"
        try:
            result_df = OCTOPUS_INSTANCE.run_regressions(
                df=data, regression_commands=['felm'],
                regression_specifications=[formula], stargazer_specs=[]
            )
            model2_results_dict[dep_var] = result_df
        except Exception as e:
            print(f"Error running R regression for {dep_var} ~ {model2_rhs}: {e}")
            model2_results_dict[dep_var] = pd.DataFrame()
    table3_results['model2'] = model2_results_dict
    
    # Model 3: Add CASH_t-1 and DEBT_t-1
    print("\nModel 3: Add CASH_t-1 and DEBT_t-1")
    model3_vars = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'mb_lag', 'cash_lag', 'debt_lag']
    model3_rhs = " + ".join(model3_vars)
    model3_results_dict = {}
    for dep_var in dependent_vars:
        formula = f"{dep_var} ~ {model3_rhs} {fe_str}"
        try:
            result_df = OCTOPUS_INSTANCE.run_regressions(
                df=data, regression_commands=['felm'],
                regression_specifications=[formula], stargazer_specs=[]
            )
            model3_results_dict[dep_var] = result_df
        except Exception as e:
            print(f"Error running R regression for {dep_var} ~ {model3_rhs}: {e}")
            model3_results_dict[dep_var] = pd.DataFrame()
    table3_results['model3'] = model3_results_dict
    
    return table3_results


def run_table4_regressions(data):
    """
    Replicate Table 4: OLS by constraint using felm.
    """
    if OCTOPUS_INSTANCE is None:
        raise RuntimeError("Octopus instance not initialized. Cannot run R regressions.")

    print("\nReplicating Table 4: OLS Regressions by Financial Constraint (via R felm)")
    print("=" * 70)
    
    dependent_vars = [
        'delta_cash_scaled', 'delta_nwc_scaled', 
        'capx1_scaled', 'capx2_scaled', 'capx3_scaled',
        'delta_debt_scaled', 'issues_scaled', 'div_scaled'
    ]
    
    constrained_data = data[data['constrained'] == 1].copy()
    unconstrained_data = data[data['unconstrained'] == 1].copy()
    
    print(f"Constrained subsample: {len(constrained_data)} observations")
    print(f"Unconstrained subsample: {len(unconstrained_data)} observations")
    
    table4_results = {'constrained': {}, 'unconstrained': {}}
    fe_str = "| factor(gvkey) + factor(fyear) | 0 | gvkey + fyear"
    model_vars = ['cash_flow_scaled', 'mb_lag']
    model_rhs = " + ".join(model_vars)

    # Constrained firms
    print("\nConstrained Firms: CF_t and MB_t-1")
    constrained_results_dict = {}
    if not constrained_data.empty:
        for dep_var in dependent_vars:
            formula = f"{dep_var} ~ {model_rhs} {fe_str}"
            try:
                result_df = OCTOPUS_INSTANCE.run_regressions(
                    df=constrained_data, regression_commands=['felm'],
                    regression_specifications=[formula], stargazer_specs=[]
                )
                constrained_results_dict[dep_var] = result_df
            except Exception as e:
                print(f"Error running R regression for constrained {dep_var} ~ {model_rhs}: {e}")
                constrained_results_dict[dep_var] = pd.DataFrame()
    table4_results['constrained'] = constrained_results_dict
    
    # Unconstrained firms
    print("\nUnconstrained Firms: CF_t and MB_t-1")
    unconstrained_results_dict = {}
    if not unconstrained_data.empty:
        for dep_var in dependent_vars:
            formula = f"{dep_var} ~ {model_rhs} {fe_str}"
            try:
                result_df = OCTOPUS_INSTANCE.run_regressions(
                    df=unconstrained_data, regression_commands=['felm'],
                    regression_specifications=[formula], stargazer_specs=[]
                )
                unconstrained_results_dict[dep_var] = result_df
            except Exception as e:
                print(f"Error running R regression for unconstrained {dep_var} ~ {model_rhs}: {e}")
                unconstrained_results_dict[dep_var] = pd.DataFrame()
    table4_results['unconstrained'] = unconstrained_results_dict
        
    return table4_results


def run_table6_regressions(data):
    """
    Replicate Table 6: IV investment-cash flow regressions using felm.
    Endogenous variable is 'mb_lag'.
    """
    if OCTOPUS_INSTANCE is None:
        raise RuntimeError("Octopus instance not initialized. Cannot run R regressions.")

    print("\nReplicating Table 6: IV Investment-Cash Flow Regressions (via R felm)")
    print("=" * 70)
    
    dependent_vars = [
        'delta_cash_scaled', 'delta_nwc_scaled', 
        'capx1_scaled', 'capx2_scaled', 'capx3_scaled',
        'delta_debt_scaled', 'issues_scaled', 'div_scaled'
    ]
    table6_results = {}
    fe_and_cluster_str = "| factor(gvkey) + factor(fyear) | {iv_spec} | gvkey + fyear"

    # Model 1: Controls: CF_t. Endog: MB_t-1. Instruments for MB_t-1: CF_t, Returns
    print("\nModel 1: Controls: CF_t. Endog: MB_t-1. IV: CF_t, Returns")
    m1_controls = ['cash_flow_scaled']
    m1_endog = 'mb_lag'
    m1_instruments = ['cash_flow_scaled', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
    m1_controls_rhs = " + ".join(m1_controls)
    m1_iv_spec = f"({m1_endog} ~ {' + '.join(m1_instruments)})"
    m1_formula_template = f"{{dep_var}} ~ {m1_controls_rhs} {fe_and_cluster_str.format(iv_spec=m1_iv_spec)}"
    
    model1_results_dict = {}
    for dep_var in dependent_vars:
        formula = m1_formula_template.format(dep_var=dep_var)
        try:
            result_df = OCTOPUS_INSTANCE.run_regressions(
                df=data, regression_commands=['felm'],
                regression_specifications=[formula], stargazer_specs=[]
            )
            # Note: felm names the endogenous variable like `fit_mb_lag` or similar in output if IV is used.
            # The coefficient for the instrumented variable will be for 'mb_lag'.
            model1_results_dict[dep_var] = result_df
        except Exception as e:
            print(f"Error running R IV regression for {dep_var} (Model 1): {e}")
            model1_results_dict[dep_var] = pd.DataFrame()
    table6_results['model1'] = model1_results_dict

    # Model 2: Controls: CF_t, CF_t-1. Endog: MB_t-1. Instruments for MB_t-1: CF_t, CF_t-1, Returns
    print("\nModel 2: Controls: CF_t, CF_t-1. Endog: MB_t-1. IV: CF_t, CF_t-1, Returns")
    m2_controls = ['cash_flow_scaled', 'cash_flow_scaled_lag']
    m2_endog = 'mb_lag'
    m2_instruments = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
    m2_controls_rhs = " + ".join(m2_controls)
    m2_iv_spec = f"({m2_endog} ~ {' + '.join(m2_instruments)})"
    m2_formula_template = f"{{dep_var}} ~ {m2_controls_rhs} {fe_and_cluster_str.format(iv_spec=m2_iv_spec)}"

    model2_results_dict = {}
    for dep_var in dependent_vars:
        formula = m2_formula_template.format(dep_var=dep_var)
        try:
            result_df = OCTOPUS_INSTANCE.run_regressions(
                df=data, regression_commands=['felm'],
                regression_specifications=[formula], stargazer_specs=[]
            )
            model2_results_dict[dep_var] = result_df
        except Exception as e:
            print(f"Error running R IV regression for {dep_var} (Model 2): {e}")
            model2_results_dict[dep_var] = pd.DataFrame()
    table6_results['model2'] = model2_results_dict

    # Model 3: Controls: CF_t, CF_t-1, CASH_t-1, DEBT_t-1. Endog: MB_t-1. Instruments for MB_t-1: CF_t, CF_t-1, Returns
    print("\nModel 3: Controls: CF_t, CF_t-1, CASH_t-1, DEBT_t-1. Endog: MB_t-1. IV: CF_t, CF_t-1, Returns")
    m3_controls = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'cash_lag', 'debt_lag']
    m3_endog = 'mb_lag'
    # Instruments for MB_t-1 are same as Model 2 for this specific table structure in original paper
    m3_instruments = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'ret_1', 'ret_2', 'ret_3', 'ret_4'] 
    m3_controls_rhs = " + ".join(m3_controls)
    m3_iv_spec = f"({m3_endog} ~ {' + '.join(m3_instruments)})"
    m3_formula_template = f"{{dep_var}} ~ {m3_controls_rhs} {fe_and_cluster_str.format(iv_spec=m3_iv_spec)}"
    
    model3_results_dict = {}
    for dep_var in dependent_vars:
        formula = m3_formula_template.format(dep_var=dep_var)
        try:
            result_df = OCTOPUS_INSTANCE.run_regressions(
                df=data, regression_commands=['felm'],
                regression_specifications=[formula], stargazer_specs=[]
            )
            model3_results_dict[dep_var] = result_df
        except Exception as e:
            print(f"Error running R IV regression for {dep_var} (Model 3): {e}")
            model3_results_dict[dep_var] = pd.DataFrame()
    table6_results['model3'] = model3_results_dict
    
    return table6_results


def run_table7_regressions(data):
    """
    Replicate Table 7: IV by constraint using felm.
    Dep var is 'capx3_scaled'. Endogenous var is 'mb_lag'.
    """
    if OCTOPUS_INSTANCE is None:
        raise RuntimeError("Octopus instance not initialized. Cannot run R regressions.")

    print("\nReplicating Table 7: IV Regressions by Financial Constraint (via R felm)")
    print("=" * 70)
    
    dep_var = 'capx3_scaled' # Fixed dependent variable for this table
    
    constrained_data = data[data['constrained'] == 1].copy()
    unconstrained_data = data[data['unconstrained'] == 1].copy()
    
    print(f"Constrained subsample: {len(constrained_data)} observations")
    print(f"Unconstrained subsample: {len(unconstrained_data)} observations")
    
    table7_results = {'constrained': {}, 'unconstrained': {}}
    fe_and_cluster_str = "| factor(gvkey) + factor(fyear) | {iv_spec} | gvkey + fyear"

    # IV Model specifications (same as Table 6)
    # Model 1
    m1_controls = ['cash_flow_scaled']
    m1_endog = 'mb_lag'
    m1_instruments = ['cash_flow_scaled', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
    m1_controls_rhs = " + ".join(m1_controls)
    m1_iv_spec = f"({m1_endog} ~ {' + '.join(m1_instruments)})"
    m1_formula = f"{dep_var} ~ {m1_controls_rhs} {fe_and_cluster_str.format(iv_spec=m1_iv_spec)}"

    # Model 2
    m2_controls = ['cash_flow_scaled', 'cash_flow_scaled_lag']
    m2_endog = 'mb_lag'
    m2_instruments = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
    m2_controls_rhs = " + ".join(m2_controls)
    m2_iv_spec = f"({m2_endog} ~ {' + '.join(m2_instruments)})"
    m2_formula = f"{dep_var} ~ {m2_controls_rhs} {fe_and_cluster_str.format(iv_spec=m2_iv_spec)}"

    # Model 3
    m3_controls = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'cash_lag', 'debt_lag']
    m3_endog = 'mb_lag'
    m3_instruments = ['cash_flow_scaled', 'cash_flow_scaled_lag', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
    m3_controls_rhs = " + ".join(m3_controls)
    m3_iv_spec = f"({m3_endog} ~ {' + '.join(m3_instruments)})"
    m3_formula = f"{dep_var} ~ {m3_controls_rhs} {fe_and_cluster_str.format(iv_spec=m3_iv_spec)}"

    models_specs = {
        'model1': m1_formula,
        'model2': m2_formula,
        'model3': m3_formula
    }

    for group_name, group_data in [('constrained', constrained_data), ('unconstrained', unconstrained_data)]:
        print(f"\n{group_name.capitalize()} Firms (Table 7 IV models)")
        group_results_dict = {}
        if not group_data.empty:
            for model_key, formula_str in models_specs.items():
                print(f"Running {model_key} for {group_name} firms...")
                try:
                    result_df = OCTOPUS_INSTANCE.run_regressions(
                        df=group_data, regression_commands=['felm'],
                        regression_specifications=[formula_str], stargazer_specs=[]
                    )
                    group_results_dict[model_key] = result_df
                except Exception as e:
                    print(f"Error running R IV regression for {group_name} {dep_var} ({model_key}): {e}")
                    group_results_dict[model_key] = pd.DataFrame()
        table7_results[group_name] = group_results_dict
            
    return table7_results


def create_summary_table(results_dict_for_table_model, model_name, key_vars):
    """
    Create a summary table of regression results from octopus DataFrames.
    results_dict_for_table_model is the full dictionary for a table (e.g., table3_results).
    model_name is 'model1', 'model2', etc.
    """
    dep_var_results_dict = results_dict_for_table_model.get(model_name, {})
    
    coefs_data = {}
    t_stats_data = {}
    r2_values = {}
    n_obs_values = {}
    
    for dep_var, result_df in dep_var_results_dict.items():
        if result_df.empty:
            print(f"Skipping {dep_var} for model {model_name} due to empty result_df.")
            continue

        coefs_data[dep_var] = {}
        t_stats_data[dep_var] = {}
        
        # R-squared and N.obs are the same for all rows of a given regression, take from first row
        r2_values[dep_var] = result_df['r_squared'].iloc[0]
        n_obs_values[dep_var] = result_df['n_obs'].iloc[0]
        
        for var_to_extract in key_vars:
            coeff_row = result_df[result_df['variable'] == var_to_extract]
            if not coeff_row.empty:
                coef_val = coeff_row['coefficient'].iloc[0]
                std_err = coeff_row['std_error'].iloc[0]
                coefs_data[dep_var][var_to_extract] = coef_val
                t_stats_data[dep_var][var_to_extract] = coef_val / std_err if std_err != 0 and not pd.isna(std_err) else np.nan
            else:
                coefs_data[dep_var][var_to_extract] = np.nan
                t_stats_data[dep_var][var_to_extract] = np.nan
    
    coef_df = pd.DataFrame.from_dict(coefs_data, orient='index')
    tstat_df = pd.DataFrame.from_dict(t_stats_data, orient='index')
    
    # Add R-squared and observation count
    if coef_df.empty and not r2_values and not n_obs_values: # Handle case where no results were processed
         coef_df['R-squared'] = pd.Series(dtype='float64')
         coef_df['N'] = pd.Series(dtype='float64')
    else:
        coef_df['R-squared'] = pd.Series(r2_values)
        coef_df['N'] = pd.Series(n_obs_values)

    # Ensure key_vars are columns, even if all NaN
    for kvar in key_vars:
        if kvar not in coef_df.columns:
            coef_df[kvar] = np.nan
        if kvar not in tstat_df.columns:
            tstat_df[kvar] = np.nan
            
    return coef_df, tstat_df


def save_regression_results(results_dict, output_dir):
    """
    Save regression results to CSV files.
    results_dict contains table names as keys, and their specific result structures as values.
    Each table will be saved as a single CSV in a long format.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for table_key, content_for_table in results_dict.items():
        # table_key is 'table3', 'table4', etc.
        # content_for_table is the dictionary returned by run_tableX_regressions
        
        all_rows_for_csv = []
        
        if table_key == 'table3' or table_key == 'table6':
            # Structure: {'model1': {'dep_varA': df, 'dep_varB': df}, 'model2': ...}
            for model_name, dep_var_dict in content_for_table.items():
                for dep_var, result_df in dep_var_dict.items():
                    if result_df.empty: continue
                    row_base = {'table': table_key, 'model': model_name, 'dependent_variable': dep_var}
                    row_base['r_squared'] = result_df['r_squared'].iloc[0]
                    row_base['n_obs'] = result_df['n_obs'].iloc[0]
                    for _, coeff_data in result_df.iterrows():
                        var_name = coeff_data['variable']
                        # In felm output, instrumented variables might be prefixed e.g. `fit_mb_lag` for the structural param
                        # or just `mb_lag`. We use the 'variable' column as is.
                        row_base[f"{var_name}_coef"] = coeff_data['coefficient']
                        row_base[f"{var_name}_tstat"] = coeff_data['coefficient'] / coeff_data['std_error'] if coeff_data['std_error'] != 0 and not pd.isna(coeff_data['std_error']) else np.nan
                    all_rows_for_csv.append(row_base)

        elif table_key == 'table4':
            # Structure: {'constrained': {'dep_varA': df, ...}, 'unconstrained': {'dep_varA': df, ...}}
            for group_name, dep_var_dict in content_for_table.items():
                for dep_var, result_df in dep_var_dict.items():
                    if result_df.empty: continue
                    row_base = {'table': table_key, 'constraint_group': group_name, 'dependent_variable': dep_var}
                    # Table 4 has one model type implicitly
                    row_base['model'] = 'model1' # Or derive if more models are added to table4
                    row_base['r_squared'] = result_df['r_squared'].iloc[0]
                    row_base['n_obs'] = result_df['n_obs'].iloc[0]
                    for _, coeff_data in result_df.iterrows():
                        var_name = coeff_data['variable']
                        row_base[f"{var_name}_coef"] = coeff_data['coefficient']
                        row_base[f"{var_name}_tstat"] = coeff_data['coefficient'] / coeff_data['std_error'] if coeff_data['std_error'] != 0 and not pd.isna(coeff_data['std_error']) else np.nan
                    all_rows_for_csv.append(row_base)

        elif table_key == 'table7':
            # Structure: {'constrained': {'model1': df_capx3, ...}, 'unconstrained': ...}
            # Dependent variable is fixed for table 7 ('capx3_scaled')
            fixed_dep_var_table7 = 'capx3_scaled'
            for group_name, model_dict in content_for_table.items():
                for model_name, result_df in model_dict.items():
                    if result_df.empty: continue
                    row_base = {'table': table_key, 'constraint_group': group_name, 'model': model_name, 'dependent_variable': fixed_dep_var_table7}
                    row_base['r_squared'] = result_df['r_squared'].iloc[0]
                    row_base['n_obs'] = result_df['n_obs'].iloc[0]
                    for _, coeff_data in result_df.iterrows():
                        var_name = coeff_data['variable']
                        row_base[f"{var_name}_coef"] = coeff_data['coefficient']
                        row_base[f"{var_name}_tstat"] = coeff_data['coefficient'] / coeff_data['std_error'] if coeff_data['std_error'] != 0 and not pd.isna(coeff_data['std_error']) else np.nan
                    all_rows_for_csv.append(row_base)
        
        if all_rows_for_csv:
            summary_df = pd.DataFrame(all_rows_for_csv)
            # Standardize column order somewhat for consistent CSVs
            cols = list(summary_df.columns)
            id_cols = [c for c in ['table', 'constraint_group', 'model', 'dependent_variable'] if c in cols]
            stat_cols = [c for c in ['r_squared', 'n_obs'] if c in cols]
            coeff_cols = sorted([c for c in cols if c.endswith('_coef')])
            tstat_cols = sorted([c for c in cols if c.endswith('_tstat')])
            ordered_cols = id_cols + stat_cols + coeff_cols + tstat_cols
            # Add any missing columns that might have been generated (e.g. if some vars only appear in some regs)
            ordered_cols.extend([c for c in cols if c not in ordered_cols])
            summary_df = summary_df[ordered_cols]

            filename = f"{table_key}_summary_results.csv" # Changed filename to reflect new format
            file_path = os.path.join(output_dir, filename)
            summary_df.to_csv(file_path, index=False, float_format='%.4f')
            print(f"Saved {filename}")
        else:
            print(f"No results to save for {table_key}")

    print(f"All regression summary results saved to {output_dir}")


def plot_investment_cash_flow_sensitivity(data, output_dir):
    """
    Plot investment-cash flow sensitivity for constrained and unconstrained firms.
    This function remains largely unchanged as it uses the raw data, not regression objects.
    """
    print("\nPlotting Investment-Cash Flow Sensitivity")
    
    # Create output directory if it doesn't exist
    plots_output_dir = os.path.join(output_dir, 'plots') # Specific plots subdir
    if not os.path.exists(plots_output_dir):
        os.makedirs(plots_output_dir)

    # Ensure necessary columns exist
    required_cols = ['constrained', 'unconstrained', 'cash_flow_scaled', 'capx3_scaled', 'fyear']
    if not all(col in data.columns for col in required_cols):
        print("Warning: Missing required columns for sensitivity plot. Skipping.")
        return

    constrained = data[data['constrained'] == 1]
    unconstrained = data[data['unconstrained'] == 1]

    plt.figure(figsize=(12, 8))
    if not constrained.empty:
        sns.regplot(x='cash_flow_scaled', y='capx3_scaled', data=constrained, 
                    scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, label='Constrained')
    if not unconstrained.empty:
        sns.regplot(x='cash_flow_scaled', y='capx3_scaled', data=unconstrained, 
                    scatter_kws={'alpha':0.3}, line_kws={'color':'blue'}, label='Unconstrained')
    
    plt.title('Investment-Cash Flow Sensitivity: Constrained vs. Unconstrained Firms')
    plt.xlabel('Cash Flow / Net Assets')
    plt.ylabel('Investment (CAPX3) / Net Assets')
    if not constrained.empty or not unconstrained.empty:
        plt.legend()
    plt.savefig(os.path.join(plots_output_dir, 'investment_cf_sensitivity_scatter.png'), dpi=300)
    plt.close()
    print("Investment-cash flow sensitivity scatter plot saved.")

    # Note: The original plot also included sensitivity over time using OLS.
    # This part would need to be refactored to use octopus if it's still desired
    # or removed if this plot is just the scatter. For now, keeping it simple.
    # The `run_baseline_analysis` in `analyze_results.py` does a more detailed version of this.


def run_all_regressions(data_path, output_dir='../results/regression_tables'):
    """
    Run all regression analyses for the Lewellen and Lewellen replication.
    Saves detailed summary CSVs for each table's results.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Loading regression sample from {data_path}")
    # Assuming data_path is to a Parquet file as used elsewhere
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        raise ValueError("Data path must be for a .csv or .parquet file.")
    print(f"Loaded {len(data)} observations")
    
    # Ensure 'gvkey' and 'fyear' are present for felm
    if 'gvkey' not in data.columns or 'fyear' not in data.columns:
        raise ValueError("Input data must contain 'gvkey' and 'fyear' columns for felm fixed effects and clustering.")

    results = {}
    
    results['table3'] = run_table3_regressions(data.copy()) # Pass copy to be safe
    results['table4'] = run_table4_regressions(data.copy())
    results['table6'] = run_table6_regressions(data.copy())
    results['table7'] = run_table7_regressions(data.copy())
    
    # Save all results
    save_regression_results(results, output_dir)
    
    # Plotting (if any specific to this module, distinct from analyze_results.py)
    # plot_investment_cash_flow_sensitivity(data, output_dir) # This was a simplified plot
    
    print(f"All regression analyses completed. Summary CSVs saved to {output_dir}")
    return results

# Example of how to run (if this script were to be run directly)
if __name__ == '__main__':
    # This is an example. Typically, analyze_results.py would call these functions.
    # Ensure you have a prepared regression_sample.parquet file.
    # Example path, adjust as needed:
    example_data_path = '../../data/regression_sample.parquet' 
    example_output_dir = '../../results/regression_analysis_output'
    
    if os.path.exists(example_data_path):
        print(f"Running example with data: {example_data_path}")
        try:
            all_results = run_all_regressions(example_data_path, example_output_dir)
            print("Example run completed.")
            
            # Example: Create a specific summary table (like original Table 3, Model 1)
            # if 'table3' in all_results and 'model1' in all_results['table3']:
            #     key_vars_model1_t3 = ['cash_flow_scaled', 'mb_lag']
            #     coef_df, tstat_df = create_summary_table(all_results['table3'], 'model1', key_vars_model1_t3)
            #     print("\nTable 3, Model 1 Coefficients:")
            #     print(coef_df.to_string(float_format="%.4f"))
            #     print("\nTable 3, Model 1 T-stats:")
            #     print(tstat_df.to_string(float_format="%.2f"))

        except Exception as e:
            print(f"Error during example run: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Example data file not found at {example_data_path}. Skipping example run.") 