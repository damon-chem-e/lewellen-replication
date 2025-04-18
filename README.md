# Lewellen and Lewellen (2016) Replication Study

This repository contains Python code to replicate and extend the findings of Lewellen and Lewellen (2016) titled "Investment and Cash Flow: New Evidence" with a larger and more recent sample.

## Overview

The original paper by Lewellen and Lewellen (2016) examines the relationship between corporate investment and cash flow, particularly the sensitivity of investment to current and lagged cash flow, the impact of financing constraints, and the role of measurement error in Tobin's q. This replication study extends the sample to include more firms and a more recent period (1971–2023) to test the robustness of the findings in a contemporary context.

## Requirements

- Python 3.7+
- pandas
- numpy
- wrds (Wharton Research Data Services Python package)
- statsmodels (for regression analysis)
- matplotlib, seaborn (for visualization)

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/lewellen-replication.git
cd lewellen-replication
```

2. Install required packages:
```
pip install pandas numpy wrds statsmodels matplotlib seaborn
```

3. Set up WRDS credentials:
   - Edit `config/wrds_credentials.py` with your WRDS username and password

## Project Structure

```
lewellen-replication/
├── config/
│   └── wrds_credentials.py     # WRDS credentials configuration
├── data/
│   ├── raw_merged_data.csv     # Raw merged data from Compustat and CRSP
│   ├── constructed_variables.csv # Data with constructed variables
│   └── regression_sample.csv   # Final regression sample
├── results/
│   ├── tables/                 # Regression tables and comparison results
│   └── plots/                  # Visualizations and figures
├── src/
│   ├── utils.py                # Utility functions
│   ├── data_collection.py      # Data collection from WRDS
│   ├── variable_construction.py # Variable construction
│   ├── sample_preparation.py   # Sample filtering and preparation
│   ├── regression_analysis.py  # Regression analysis implementation
│   ├── analyze_results.py      # Analysis and visualization functions
│   └── main.py                 # Main script to run the pipeline
└── README.md                   # This file
```

## Usage

### Command-Line Options

The main script (`src/main.py`) accepts several command-line arguments:

```
python src/main.py [--start_year YEAR] [--end_year YEAR] [--force_refresh] [--run_analysis] [--analysis_only]
```

- `--start_year`: Start year for data collection (default: 1971)
- `--end_year`: End year for data collection (default: 2023)
- `--force_refresh`: Force refresh of all data files even if they exist
- `--run_analysis`: Run analysis after preparing the data
- `--analysis_only`: Skip data preparation and only run analysis

### Example Usage

1. Run the full pipeline with default settings:
```
python src/main.py
```

2. Run with custom time period:
```
python src/main.py --start_year 1980 --end_year 2020
```

3. Run data preparation and analysis:
```
python src/main.py --run_analysis
```

4. Run analysis only (assumes data is already prepared):
```
python src/main.py --analysis_only
```

5. Force refresh all data files and run analysis:
```
python src/main.py --force_refresh --run_analysis
```

### Analysis Results

When the analysis is run, it generates:

1. Tables in `results/tables/` directory:
   - Regression results from Tables 3, 4, 6, and 7
   - Comparison with original paper coefficients

2. Plots in `results/plots/` directory:
   - Variable distributions
   - Investment-cash flow sensitivity
   - Constraint group comparisons
   - Cash flow allocation
   - Investment financing

## Variables Constructed

The code constructs various measures including:

1. **Net Assets**: Total assets minus nondebt current liabilities
2. **Cash Flow**: Operating cash flow adjusted for noncash items
3. **Investment Measures**:
   - CAPX1: Capital Expenditures
   - CAPX2: CAPX + Acquisitions + Other investing activities
   - CAPX3: Total Long-Term Investment
4. **Other Cash Flow Uses**:
   - Change in Cash Holdings
   - Change in Net Working Capital
   - Change in Debt
   - Equity Issuance
   - Dividends
5. **Market-to-Book** ratio as proxy for Tobin's q
6. **Financial Constraint Classification** based on forecasted free cash flow

## Sample Selection

The sample includes all nonfinancial firms listed on Compustat with corresponding stock return data on CRSP from 1971 to 2023, with the following filters:

- Financial firms (SIC codes 6000-6999) are excluded
- Small firms below the NYSE 10th percentile of net assets are excluded
- Firms with missing data for key variables are removed
- All variables are winsorized at the 1st and 99th percentiles

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This code replicates the methodology described in Lewellen, J., & Lewellen, K. (2016). Investment and Cash Flow: New Evidence. *Journal of Financial and Quantitative Analysis*, 51(4), 1135-1164.
- Data is sourced from Compustat and CRSP via WRDS.

## Contributors

- Your Name - [youremail@example.com](mailto:youremail@example.com) 