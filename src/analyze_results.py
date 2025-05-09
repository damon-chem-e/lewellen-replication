"""
Analysis Script for Lewellen and Lewellen (2016) Replication.

This script analyzes and visualizes the results of the replication study,
generating tables and plots that can be saved to disk.

Usage:
    python analyze_results.py [--data_path PATH] [--output_dir PATH]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import our modules
from src.regression_analysis import (
    run_ols_regression, 
    run_iv_regression, 
    run_table3_regressions,
    run_table4_regressions,
    run_table6_regressions,
    run_table7_regressions,
    plot_investment_cash_flow_sensitivity
)


def create_results_directory(output_dir):
    """
    Create results directory if it doesn't exist.
    
    Args:
        output_dir (str): Path to results directory
        
    Returns:
        str: Path to results directory
    """
    # Create directories for tables and plots
    tables_dir = os.path.join(output_dir, 'tables')
    plots_dir = os.path.join(output_dir, 'plots')
    
    for directory in [output_dir, tables_dir, plots_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    return tables_dir, plots_dir


def analyze_sample_characteristics(data, plots_dir):
    """
    Analyze and visualize sample characteristics.
    
    Args:
        data (pd.DataFrame): Regression sample data
        plots_dir (str): Directory to save plots
    """
    print("\n=== Sample Characteristics ===")
    print(f"Sample period: {data['fyear'].min()} to {data['fyear'].max()}")
    print(f"Number of observations: {len(data)}")
    print(f"Number of unique firms: {data['gvkey'].nunique()}")
    print(f"Number of years: {data['fyear'].nunique()}")
    
    # Summary statistics for key variables
    key_vars = [
        'cash_flow_scaled', 'cash_flow_scaled_lag', 'mb_lag',
        'capx1_scaled', 'capx2_scaled', 'capx3_scaled',
        'delta_cash_scaled', 'delta_nwc_scaled', 'delta_debt_scaled',
        'issues_scaled', 'div_scaled',
        'cash_lag', 'debt_lag'
    ]
    
    # Create descriptive statistics
    stats = data[key_vars].describe().T
    # Add additional statistics
    stats['skew'] = data[key_vars].skew()
    stats['kurtosis'] = data[key_vars].kurtosis()
    
    # Print key statistics
    print("\nKey Variable Statistics:")
    print(stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(4))
    
    # Save statistics
    stats.to_csv(os.path.join(plots_dir, 'sample_statistics.csv'))
    
    # Create distribution plots
    print("\nCreating variable distribution plots...")
    
    # Cash flow and investment measures
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Cash flow and investment measures
    sns.histplot(data['cash_flow_scaled'], kde=True, ax=axes[0])
    axes[0].set_title('Cash Flow Distribution')
    
    sns.histplot(data['capx1_scaled'], kde=True, ax=axes[1])
    axes[1].set_title('CAPX1 (Capital Expenditures) Distribution')
    
    sns.histplot(data['capx3_scaled'], kde=True, ax=axes[2])
    axes[2].set_title('CAPX3 (Total Investment) Distribution')
    
    # Market-to-book and other variables
    sns.histplot(data['mb_lag'], kde=True, ax=axes[3])
    axes[3].set_title('Market-to-Book Ratio Distribution')
    
    sns.histplot(data['delta_debt_scaled'], kde=True, ax=axes[4])
    axes[4].set_title('Change in Debt Distribution')
    
    sns.histplot(data['issues_scaled'], kde=True, ax=axes[5])
    axes[5].set_title('Equity Issuance Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'variable_distributions.png'), dpi=300)
    plt.close()
    
    print("Variable distribution plots saved.")


def analyze_constraint_groups(data, plots_dir):
    """
    Analyze characteristics of different financial constraint groups.
    
    Args:
        data (pd.DataFrame): Regression sample data
        plots_dir (str): Directory to save plots
    """
    print("\n=== Financial Constraint Analysis ===")
    
    # Create subsamples
    constrained = data[data['constrained'] == 1]
    unconstrained = data[data['unconstrained'] == 1]
    
    print(f"Constrained firms: {len(constrained)} observations")
    print(f"Unconstrained firms: {len(unconstrained)} observations")
    
    # Key variables to compare
    key_vars = [
        'cash_flow_scaled', 'mb_lag', 'capx1_scaled', 'capx3_scaled',
        'delta_cash_scaled', 'delta_debt_scaled', 'issues_scaled', 
        'cash_lag', 'debt_lag'
    ]
    
    # Compare key characteristics
    comparison = pd.DataFrame({
        'Constrained': constrained[key_vars].mean(),
        'Unconstrained': unconstrained[key_vars].mean(),
        'Difference': unconstrained[key_vars].mean() - constrained[key_vars].mean(),
        'Diff %': ((unconstrained[key_vars].mean() / constrained[key_vars].mean()) - 1) * 100
    })
    
    # Display comparison
    print("\nComparison of Mean Values by Constraint Status:")
    print(comparison.round(4))
    
    # Save comparison
    comparison.to_csv(os.path.join(plots_dir, 'constraint_group_comparison.csv'))
    
    # Visualize key differences
    print("\nCreating constraint group comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Cash flow
    sns.boxplot(x='constraint_group', y='cash_flow_scaled', data=data, ax=axes[0, 0])
    axes[0, 0].set_title('Cash Flow by Constraint Group')
    axes[0, 0].set_xlabel('Constraint Group (1=Most Constrained)')
    
    # Investment
    sns.boxplot(x='constraint_group', y='capx3_scaled', data=data, ax=axes[0, 1])
    axes[0, 1].set_title('Total Investment by Constraint Group')
    axes[0, 1].set_xlabel('Constraint Group (1=Most Constrained)')
    
    # Market-to-book
    sns.boxplot(x='constraint_group', y='mb_lag', data=data, ax=axes[1, 0])
    axes[1, 0].set_title('Market-to-Book by Constraint Group')
    axes[1, 0].set_xlabel('Constraint Group (1=Most Constrained)')
    
    # Debt
    sns.boxplot(x='constraint_group', y='debt_lag', data=data, ax=axes[1, 1])
    axes[1, 1].set_title('Debt by Constraint Group')
    axes[1, 1].set_xlabel('Constraint Group (1=Most Constrained)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'constraint_group_comparison.png'), dpi=300)
    plt.close()
    
    print("Constraint group comparison plots saved.")


def run_baseline_analysis(data, plots_dir):
    """
    Run baseline analysis of investment-cash flow sensitivity.
    
    Args:
        data (pd.DataFrame): Regression sample data
        plots_dir (str): Directory to save plots
    """
    print("\n=== Investment-Cash Flow Sensitivity Analysis ===")
    
    # Create subsamples
    constrained = data[data['constrained'] == 1]
    unconstrained = data[data['unconstrained'] == 1]
    
    # Scatterplot of investment vs cash flow
    print("Creating investment-cash flow sensitivity plots...")
    
    plt.figure(figsize=(12, 8))
    
    # Constrained firms
    sns.regplot(x='cash_flow_scaled', y='capx3_scaled', 
                data=constrained, 
                scatter_kws={'alpha':0.3}, 
                line_kws={'color':'red'},
                label='Constrained')
    
    # Unconstrained firms
    sns.regplot(x='cash_flow_scaled', y='capx3_scaled', 
                data=unconstrained, 
                scatter_kws={'alpha':0.3}, 
                line_kws={'color':'blue'},
                label='Unconstrained')
    
    plt.title('Investment-Cash Flow Sensitivity: Constrained vs. Unconstrained Firms')
    plt.xlabel('Cash Flow / Net Assets')
    plt.ylabel('Investment (CAPX3) / Net Assets')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'investment_cf_sensitivity.png'), dpi=300)
    plt.close()
    
    # Investment-cash flow sensitivity over time
    data['decade'] = (data['fyear'] // 10) * 10
    decades = sorted(data['decade'].unique())
    sensitivities = []
    
    for decade in decades:
        decade_data = data[data['decade'] == decade]
        
        # Run regression for constrained firms
        const_data = decade_data[decade_data['constrained'] == 1]
        if len(const_data) > 30:  # Ensure enough observations
            const_reg = run_ols_regression(
                const_data, 'capx3_scaled', ['cash_flow_scaled', 'mb_lag'],
                fe_firm=False, fe_year=False
            )
            const_sensitivity = const_reg.params['cash_flow_scaled']
        else:
            const_sensitivity = np.nan
        
        # Run regression for unconstrained firms
        unconst_data = decade_data[decade_data['unconstrained'] == 1]
        if len(unconst_data) > 30:  # Ensure enough observations
            unconst_reg = run_ols_regression(
                unconst_data, 'capx3_scaled', ['cash_flow_scaled', 'mb_lag'],
                fe_firm=False, fe_year=False
            )
            unconst_sensitivity = unconst_reg.params['cash_flow_scaled']
        else:
            unconst_sensitivity = np.nan
            
        sensitivities.append({
            'decade': decade,
            'constrained': const_sensitivity,
            'unconstrained': unconst_sensitivity
        })
    
    # Convert to DataFrame and plot
    sensitivity_df = pd.DataFrame(sensitivities)
    sensitivity_df.to_csv(os.path.join(plots_dir, 'sensitivity_over_time.csv'), index=False)
    
    plt.figure(figsize=(12, 6))
    plt.plot(sensitivity_df['decade'], sensitivity_df['constrained'], 'ro-', label='Constrained')
    plt.plot(sensitivity_df['decade'], sensitivity_df['unconstrained'], 'bo-', label='Unconstrained')
    plt.title('Investment-Cash Flow Sensitivity Over Time')
    plt.xlabel('Decade')
    plt.ylabel('Sensitivity (Regression Coefficient)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'sensitivity_over_time.png'), dpi=300)
    plt.close()
    
    print("Investment-cash flow sensitivity plots saved.")


def analyze_cash_flow_uses(data, plots_dir):
    """
    Analyze how firms allocate their cash flow across different uses.
    
    Args:
        data (pd.DataFrame): Regression sample data
        plots_dir (str): Directory to save plots
    """
    print("\n=== Cash Flow Allocation Analysis ===")
    
    # Create subsamples
    constrained = data[data['constrained'] == 1]
    unconstrained = data[data['unconstrained'] == 1]
    
    # Calculate average allocation of cash flow
    uses = ['delta_cash_scaled', 'delta_nwc_scaled', 'capx1_scaled', 
            'delta_debt_scaled', 'issues_scaled', 'div_scaled']
    
    # Use CAPX1 for this analysis to avoid double-counting
    averages = data[uses].mean()
    
    # Save allocation data
    allocation_df = pd.DataFrame({
        'All Firms': data[uses].mean(),
        'Constrained': constrained[uses].mean(),
        'Unconstrained': unconstrained[uses].mean()
    })
    allocation_df.to_csv(os.path.join(plots_dir, 'cash_flow_allocation.csv'))
    
    # Plot allocation
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("husl", len(uses))
    bars = plt.bar(averages.index, averages.values, color=colors)
    plt.title('Average Allocation of Cash Flow')
    plt.ylabel('Proportion of Net Assets')
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.3f}',
                 ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cash_flow_allocation.png'), dpi=300)
    plt.close()
    
    # Compare allocation between constrained and unconstrained firms
    allocation = pd.DataFrame({
        'Constrained': constrained[uses].mean(),
        'Unconstrained': unconstrained[uses].mean()
    })
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    allocation.plot(kind='bar', ax=ax)
    plt.title('Cash Flow Allocation: Constrained vs. Unconstrained Firms')
    plt.ylabel('Proportion of Net Assets')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Firm Type')
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'constraint_cf_allocation.png'), dpi=300)
    plt.close()
    
    print("Cash flow allocation plots saved.")


def analyze_investment_financing(data, plots_dir):
    """
    Analyze how firms finance their investment activities.
    
    Args:
        data (pd.DataFrame): Regression sample data
        plots_dir (str): Directory to save plots
    """
    print("\n=== Investment Financing Analysis ===")
    
    # Calculate financing proportions
    data['investment'] = data['capx1_scaled']
    data['internal_financing'] = data['cash_flow_scaled']
    data['debt_financing'] = data['delta_debt_scaled']
    data['equity_financing'] = data['issues_scaled']
    data['other_financing'] = -data['delta_cash_scaled'] - data['delta_nwc_scaled'] - data['div_scaled']
    
    # Calculate proportions for firms with positive investment
    investing_firms = data[data['investment'] > 0].copy()
    financing_cols = ['internal_financing', 'debt_financing', 'equity_financing', 'other_financing']
    
    for col in financing_cols:
        investing_firms[f'{col}_pct'] = investing_firms[col] / investing_firms['investment']
    
    # Calculate and save averages
    financing_props = investing_firms[[f'{col}_pct' for col in financing_cols]].mean()
    financing_props.to_csv(os.path.join(plots_dir, 'financing_proportions.csv'))
    
    # Calculate proportions by constraint group
    constrained_firms = investing_firms[investing_firms['constrained'] == 1]
    unconstrained_firms = investing_firms[investing_firms['unconstrained'] == 1]
    
    # Check if we have both groups represented
    has_constrained = len(constrained_firms) > 0
    has_unconstrained = len(unconstrained_firms) > 0
    
    print(f"Investing firms with constraint data: {len(investing_firms)} observations")
    print(f"Constrained investing firms: {len(constrained_firms)} observations")
    print(f"Unconstrained investing firms: {len(unconstrained_firms)} observations")
    
    # Create financing by group dataframe
    financing_by_group = pd.DataFrame({
        'All Firms': investing_firms[[f'{col}_pct' for col in financing_cols]].mean()
    })
    
    if has_constrained:
        financing_by_group['Constrained'] = constrained_firms[[f'{col}_pct' for col in financing_cols]].mean()
    
    if has_unconstrained:
        financing_by_group['Unconstrained'] = unconstrained_firms[[f'{col}_pct' for col in financing_cols]].mean()
    
    financing_by_group.to_csv(os.path.join(plots_dir, 'financing_by_constraint.csv'))
    
    # Plot financing sources for all firms
    plt.figure(figsize=(10, 6))
    plt.pie(financing_props.abs(), labels=financing_props.index, autopct='%1.1f%%', 
            startangle=90, colors=sns.color_palette("pastel", len(financing_props)))
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Average Investment Financing Sources')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'investment_financing.png'), dpi=300)
    plt.close()
    
    # Plot financing sources by constraint group
    if has_constrained and has_unconstrained:
        # Both groups present - create side-by-side plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Constrained firms
        ax1.pie(constrained_firms[[f'{col}_pct' for col in financing_cols]].mean().abs(), 
                labels=financing_cols, autopct='%1.1f%%', 
                startangle=90, colors=sns.color_palette("pastel", len(financing_cols)))
        ax1.set_title('Constrained Firms')
        
        # Unconstrained firms
        ax2.pie(unconstrained_firms[[f'{col}_pct' for col in financing_cols]].mean().abs(), 
                labels=financing_cols, autopct='%1.1f%%', 
                startangle=90, colors=sns.color_palette("pastel", len(financing_cols)))
        ax2.set_title('Unconstrained Firms')
        
        plt.suptitle('Investment Financing by Constraint Status', fontsize=16)
    elif has_unconstrained:
        # Only unconstrained firms present
        plt.figure(figsize=(10, 6))
        plt.pie(unconstrained_firms[[f'{col}_pct' for col in financing_cols]].mean().abs(), 
                labels=financing_cols, autopct='%1.1f%%', 
                startangle=90, colors=sns.color_palette("pastel", len(financing_cols)))
        plt.axis('equal')
        plt.title('Investment Financing: Unconstrained Firms')
    elif has_constrained:
        # Only constrained firms present
        plt.figure(figsize=(10, 6))
        plt.pie(constrained_firms[[f'{col}_pct' for col in financing_cols]].mean().abs(), 
                labels=financing_cols, autopct='%1.1f%%', 
                startangle=90, colors=sns.color_palette("pastel", len(financing_cols)))
        plt.axis('equal')
        plt.title('Investment Financing: Constrained Firms')
    else:
        # Neither group present - skip this plot
        print("No firms classified as constrained or unconstrained for financing analysis")
        return
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'financing_by_constraint.png'), dpi=300)
    plt.close()
    
    print("Investment financing plots saved.")


def compare_with_original_paper(data, tables_dir):
    """
    Compare key results with the original paper.
    
    Args:
        data (pd.DataFrame): Regression sample data
        tables_dir (str): Directory to save tables
    """
    print("\n=== Comparison with Original Paper ===")
    
    # Create subsamples
    constrained = data[data['constrained'] == 1]
    unconstrained = data[data['unconstrained'] == 1]
    
    # Check if we have both constraint groups
    has_constrained = len(constrained) > 0
    has_unconstrained = len(unconstrained) > 0
    
    print(f"Full sample size: {len(data)} observations")
    print(f"Constrained subsample: {len(constrained)} observations")
    print(f"Unconstrained subsample: {len(unconstrained)} observations")
    
    # Original paper values
    original_values = {
        'OLS_Full': 0.35,
        'OLS_Constrained': 0.72,
        'OLS_Unconstrained': 0.30,
        'IV_Full': 0.28,
        'IV_Constrained': 0.63,
        'IV_Unconstrained': 0.16
    }
    
    # Initialize replication values
    replication_values = {}
    
    # Run OLS regressions
    ols_results = run_ols_regression(
        data, 'capx3_scaled', ['cash_flow_scaled', 'mb_lag'],
        fe_firm=True, fe_year=True
    )
    replication_values['OLS_Full'] = ols_results.params['cash_flow_scaled']
    
    # Run IV regression for full sample
    instruments = ['cash_flow_scaled', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
    
    try:
        first_stage, second_stage = run_iv_regression(
            data, 'capx3_scaled', 'mb_lag', instruments, ['cash_flow_scaled']
        )
        replication_values['IV_Full'] = second_stage.params['cash_flow_scaled']
    except Exception as e:
        print(f"Warning: Could not run IV regression for full sample: {e}")
        replication_values['IV_Full'] = None
    
    # Run regressions for constrained firms if available
    if has_constrained:
        try:
            constrained_results = run_ols_regression(
                constrained, 'capx3_scaled', ['cash_flow_scaled', 'mb_lag'],
                fe_firm=True, fe_year=True
            )
            replication_values['OLS_Constrained'] = constrained_results.params['cash_flow_scaled']
            
            first_stage_c, second_stage_c = run_iv_regression(
                constrained, 'capx3_scaled', 'mb_lag', instruments, ['cash_flow_scaled']
            )
            replication_values['IV_Constrained'] = second_stage_c.params['cash_flow_scaled']
        except Exception as e:
            print(f"Warning: Could not run regressions for constrained firms: {e}")
            replication_values['OLS_Constrained'] = None
            replication_values['IV_Constrained'] = None
    else:
        print("No constrained firms in sample, skipping constrained regressions")
        replication_values['OLS_Constrained'] = None
        replication_values['IV_Constrained'] = None
    
    # Run regressions for unconstrained firms if available
    if has_unconstrained:
        try:
            unconstrained_results = run_ols_regression(
                unconstrained, 'capx3_scaled', ['cash_flow_scaled', 'mb_lag'],
                fe_firm=True, fe_year=True
            )
            replication_values['OLS_Unconstrained'] = unconstrained_results.params['cash_flow_scaled']
            
            first_stage_u, second_stage_u = run_iv_regression(
                unconstrained, 'capx3_scaled', 'mb_lag', instruments, ['cash_flow_scaled']
            )
            replication_values['IV_Unconstrained'] = second_stage_u.params['cash_flow_scaled']
        except Exception as e:
            print(f"Warning: Could not run regressions for unconstrained firms: {e}")
            replication_values['OLS_Unconstrained'] = None
            replication_values['IV_Unconstrained'] = None
    else:
        print("No unconstrained firms in sample, skipping unconstrained regressions")
        replication_values['OLS_Unconstrained'] = None
        replication_values['IV_Unconstrained'] = None
    
    # Create comparison dataframe
    comparison_data = {'Original': original_values, 'Replication': replication_values}
    
    # Calculate differences only for values that exist
    diff_data = {}
    pct_diff_data = {}
    
    for k in original_values.keys():
        if replication_values.get(k) is not None:
            diff_data[k] = replication_values[k] - original_values[k]
            pct_diff_data[k] = (replication_values[k] / original_values[k] - 1) * 100
        else:
            diff_data[k] = None
            pct_diff_data[k] = None
    
    comparison_data['Difference'] = diff_data
    comparison_data['Percent Diff'] = pct_diff_data
    
    comparison = pd.DataFrame(comparison_data)
    
    # Print comparison
    print("Comparison of key coefficients with original paper:")
    print(comparison.round(4))
    
    # Save comparison
    comparison.to_csv(os.path.join(tables_dir, 'paper_comparison.csv'))
    
    # Create visual comparison only for available results
    plt.figure(figsize=(10, 6))
    
    # Filter for results that have values
    available_keys = [k for k in replication_values.keys() if replication_values[k] is not None]
    
    if available_keys:
        # Create the bar chart
        x = np.arange(len(available_keys))
        width = 0.35
        
        original = [original_values[k] for k in available_keys]
        replication = [replication_values[k] for k in available_keys]
        
        plt.bar(x - width/2, original, width, label='Original Paper')
        plt.bar(x + width/2, replication, width, label='Replication')
        
        plt.xlabel('Model')
        plt.ylabel('Cash Flow Coefficient')
        plt.title('Comparison of Cash Flow Coefficients with Original Paper')
        plt.xticks(x, available_keys, rotation=45)
        plt.legend()
    else:
        # No results to plot
        plt.text(0.5, 0.5, "No comparison results available", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.title('Comparison with Original Paper')
    
    plt.tight_layout()
    plt.savefig(os.path.join(tables_dir, 'paper_comparison.png'), dpi=300)
    plt.close()
    
    print("Comparison with original paper saved.")


def run_full_regression_analysis(data, tables_dir, plots_dir):
    """
    Run full regression analysis and save results.
    
    Args:
        data (pd.DataFrame): Regression sample data
        tables_dir (str): Directory to save tables
        plots_dir (str): Directory to save plots
    """
    print("\n=== Running Full Regression Analysis ===")
    
    # Check if we have constrained and unconstrained groups
    constrained = data[data['constrained'] == 1]
    unconstrained = data[data['unconstrained'] == 1]
    has_constraint_groups = len(constrained) > 0 and len(unconstrained) > 0
    
    # Run Table 3 regressions (OLS investment-cash flow sensitivities)
    print("Running Table 3 regressions...")
    try:
        table3_results = run_table3_regressions(data)
    except Exception as e:
        print(f"Error running Table 3 regressions: {e}")
    
    # Run Table 4 regressions (OLS by constraint)
    if has_constraint_groups:
        print("Running Table 4 regressions...")
        try:
            table4_results = run_table4_regressions(data)
        except Exception as e:
            print(f"Error running Table 4 regressions: {e}")
    else:
        print("Skipping Table 4 regressions due to missing constraint groups")
    
    # Run Table 6 regressions (IV regressions)
    print("Running Table 6 regressions...")
    try:
        table6_results = run_table6_regressions(data)
    except Exception as e:
        print(f"Error running Table 6 regressions: {e}")
    
    # Run Table 7 regressions (IV by constraint)
    if has_constraint_groups:
        print("Running Table 7 regressions...")
        try:
            table7_results = run_table7_regressions(data)
        except Exception as e:
            print(f"Error running Table 7 regressions: {e}")
    else:
        print("Skipping Table 7 regressions due to missing constraint groups")
    
    # Create investment-cash flow sensitivity plots
    print("Creating investment-cash flow sensitivity plots...")
    try:
        plot_investment_cash_flow_sensitivity(data, plots_dir)
    except Exception as e:
        print(f"Error creating investment-cash flow sensitivity plots: {e}")
    
    print("Full regression analysis completed.")


def main():
    """
    Main function to parse arguments and run analysis.
    """
    parser = argparse.ArgumentParser(
        description='Analyze and visualize Lewellen and Lewellen (2016) replication results'
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        help='Path to regression sample data',
        default=None
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        help='Directory to save results',
        default=None
    )
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.data_path is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        args.data_path = os.path.join(data_dir, 'raw_merged_data.parquet')
    
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        print("Please run src/main.py first to prepare the data.")
        return
    
    # Create results directories
    tables_dir, plots_dir = create_results_directory(args.output_dir)
    
    # Load data
    print(f"Loading raw data from {args.data_path}")
    data = pd.read_parquet(args.data_path)
    print(f"Loaded {len(data)} observations")
    
    # Perform variable construction and sample preparation directly
    print("Constructing variables...")
    from src.variable_construction import construct_variables
    constructed_data = construct_variables(data)
    
    print("Preparing regression sample...")
    from src.sample_preparation import prepare_regression_sample
    regression_sample = prepare_regression_sample(constructed_data)
    print(f"Final regression sample has {len(regression_sample)} observations")
    
    # Run analyses
    start_time = datetime.now()
    print(f"Analysis started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    analyze_sample_characteristics(regression_sample, plots_dir)
    analyze_constraint_groups(regression_sample, plots_dir)
    run_baseline_analysis(regression_sample, plots_dir)
    analyze_cash_flow_uses(regression_sample, plots_dir)
    analyze_investment_financing(regression_sample, plots_dir)
    compare_with_original_paper(regression_sample, tables_dir)
    run_full_regression_analysis(regression_sample, tables_dir, plots_dir)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    print(f"\nAnalysis completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {duration:.2f} minutes")
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main() 