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
├── notebooks/
│   └── analysis.ipynb          # Jupyter notebook for analysis
├── src/
│   ├── utils.py                # Utility functions
│   ├── data_collection.py      # Data collection from WRDS
│   ├── variable_construction.py # Variable construction
│   ├── sample_preparation.py   # Sample filtering and preparation
│   ├── main.py                 # Main script to run the pipeline
│   └── regression_analysis.py  # Regression analysis
└── README.md                   # This file
```

## Usage

### 1. Data Preparation

The entire data preparation pipeline can be run using the main script:

```
python src/main.py
```

This will:
1. Collect raw data from WRDS (Compustat and CRSP)
2. Construct all necessary variables
3. Prepare the regression sample with filtering criteria
4. Save all intermediate and final datasets

### 2. Individual Steps

You can also run each step separately:

```
# Step 1: Data Collection
python src/data_collection.py

# Step 2: Variable Construction
python src/variable_construction.py

# Step 3: Sample Preparation
python src/sample_preparation.py
```

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