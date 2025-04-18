To replicate the study by Lewellen and Lewellen (2016) titled "Investment and Cash Flow: New Evidence" with a larger and more recent sample of data, the following detailed plan outlines the steps required, including data sources, variable definitions, regression specifications, and methodological considerations. The plan assumes access to the same data sources used in the original study (Compustat and CRSP) and extends the sample period to include more recent data, potentially from 1971 to 2023, to capture a broader and more current dataset. The goal is to verify the original findings on investment-cash flow sensitivities while accounting for methodological advancements and data refinements introduced in the paper.

---

### Replication Plan

#### 1. Objective
Replicate the key findings of Lewellen and Lewellen (2016) regarding the relationship between investment and cash flow, particularly the sensitivity of investment to current and lagged cash flow, the impact of financing constraints, and the role of measurement error in Tobin's q. Extend the sample to include more firms and a more recent period (1971–2023) to test the robustness of the findings in a contemporary context.

#### 2. Data Sources
The original study uses data from Compustat and CRSP. The replication will leverage the same sources, ensuring consistency in data structure and availability.

- **Compustat (North America, Fundamentals Annual)**:
  - Provides financial statement data, including income statements, balance sheets, and statements of cash flows for U.S. firms.
  - Used to construct variables such as cash flow, investment measures, net assets, debt, equity, and other financial metrics.
  - Access historical and recent data to extend the sample period from 1971 to 2023.
- **CRSP (Center for Research in Security Prices)**:
  - Provides stock price and return data for U.S. firms.
  - Used to calculate market values, stock returns, and the market-to-book ratio as a proxy for Tobin's q.
  - Ensure linkage with Compustat via identifiers (e.g., PERMNO-PERMCO-GVKEY mapping).

#### 3. Sample Selection
- **Universe**: All nonfinancial firms listed on Compustat with corresponding stock return data on CRSP from 1971 to 2023.
- **Filters**:
  - Exclude financial firms (SIC codes 6000–6999) to focus on nonfinancial industries.
  - Require non-missing data for net assets (total assets minus nondebt current liabilities) and stock returns.
  - Exclude small firms below the NYSE 10th percentile of net assets at the beginning of each year to avoid noise from microcap firms, consistent with the original study (noting that firms above this cutoff represent ~98% of aggregate asset value).
  - Winsorize all variables at the 1st and 99th percentiles to mitigate the impact of outliers.
- **Sample Size**: The original study included ~1,800 firms per year. With a longer period and broader coverage, expect a larger sample, potentially 2,000–3,000 firms per year, depending on data availability and market coverage in recent years.

#### 4. Variables and Definitions
The study introduces refined measures of cash flow and investment, tracks all uses of cash flow, and uses a market-to-book ratio as a proxy for Tobin's q. Below are the key variables and their construction, based on Compustat and CRSP data items.

##### A. Net Assets
- **Definition**: Total assets minus nondebt current liabilities.
- **Compustat Items**:
  - Total Assets: `AT`
  - Nondebt Current Liabilities: Current liabilities (`LCT`) minus short-term debt (e.g., `DLC`).
  - Formula: `NET_ASSETS = AT - (LCT - DLC)`.
- **Purpose**: Used as a scaling factor for level and flow variables and as the denominator in the market-to-book ratio.

##### B. Cash Flow (CF)
- **Definition**: Operating cash flow adjusted for noncash items, derived from the statement of cash flows (SCF) to improve accuracy over the traditional measure (income before extraordinary items + depreciation).
- **Compustat Items (SCF)**:
  - Income Before Extraordinary Items: `IBC`
  - Extraordinary Items and Discontinued Operations: `XIDOC`
  - Depreciation and Amortization: `DPC`
  - Deferred Taxes: `TXDC`
  - Equity in Net Loss of Unconsolidated Subsidiaries: `ESUBC`
  - Losses from Sale of PPE: `SPPIV`
  - Funds from Operations (Other, including asset write-downs): `FOPO`
- **Formula**: 
  \[
  CF = IBC + XIDOC + DPC + TXDC + ESUBC + SPPIV + FOPO
  \]
- **Notes**:
  - Excludes spending on working capital, treated as an investment component.
  - Scaled by average net assets for the year: \((NET_ASSETS_t + NET_ASSETS_{t-1})/2\).
  - Compare with traditional measure: `PROF + DEPR = IBC + DPC`.

##### C. Investment Measures
Three measures of investment are used, progressively broader in scope:
- **CAPX1 (Capital Expenditures)**:
  - Compustat Item: `CAPX` (net capital expenditures).
  - Scaled by average net assets.
- **CAPX2 (Capital Expenditures + Other Investing Activities)**:
  - Includes `CAPX` plus other investing activities from SCF (e.g., purchases of patents, acquisitions).
  - Compustat Items: `CAPX` + `AQC` (acquisitions, cash portion) + other SCF investing items (e.g., `IVCH`, `SIV`).
  - Scaled by average net assets.
- **CAPX3 (Total Long-Term Investment)**:
  - Derived from year-over-year change in fixed assets, adjusted for noncash charges (e.g., depreciation, write-downs).
  - Compustat Items:
    - Property, Plant, and Equipment (PPE): `PPENT`
    - Depreciation and Amortization: `DPC`
    - Asset Write-Downs: Approximate via `WDP` or `FOPO` if available.
  - Formula: 
    \[
    CAPX3 = \Delta PPENT + DPC + \text{Write-Downs}
    \]
  - Notes: Includes all acquisitions (cash and stock-for-stock), unlike `AQC`. Scaled by average net assets.

##### D. Other Uses of Cash Flow
Track all seven uses of cash flow to provide a complete accounting:
- **Change in Cash Holdings (\(\Delta CASH\))**:
  - Compustat Item: Change in `CHE` (cash and equivalents).
  - Formula: \(\Delta CASH = CHE_t - CHE_{t-1}\).
  - Scaled by average net assets.
- **Change in Net Working Capital (\(\Delta NWC\))**:
  - Definition: Noncash current assets minus current operating liabilities.
  - Compustat Items:
    - Noncash Current Assets: `ACT` (current assets) minus `CHE`.
    - Current Operating Liabilities: `LCT` minus `DLC`.
  - Formula: 
    \[
    NWC = (ACT - CHE) - (LCT - DLC), \quad \Delta NWC = NWC_t - NWC_{t-1}
    \]
  - Scaled by average net assets.
- **Change in Debt (\(\Delta DEBT\))**:
  - Definition: Includes short-term debt, long-term debt, and other long-term liabilities, adjusted for deferred tax accruals.
  - Compustat Items:
    - Short-Term Debt: `DLC`
    - Long-Term Debt: `DLTT`
    - Other Liabilities: `LT` (total liabilities) minus `LCT` and `DLTT`.
    - Deferred Taxes: `TXDC`.
  - Formula: 
    \[
    DEBT = DLC + DLTT + (LT - LCT - DLTT), \quad \Delta DEBT = DEBT_t - DEBT_{t-1} - \Delta TXDC
    \]
  - Scaled by average net assets.
- **Equity Issuance (ISSUES)**:
  - Definition: Change in total equity minus change in retained earnings.
  - Compustat Items:
    - Total Equity: `CEQ` (common equity) + `PSTK` (preferred stock).
    - Retained Earnings: `RE`.
  - Formula: 
    \[
    ISSUES = (\Delta CEQ + \Delta PSTK) - \Delta RE
    \]
  - Scaled by average net assets.
- **Dividends (DIV)**:
  - Compustat Items: `DVC` (common dividends) + `DVP` (preferred dividends).
  - Formula: \(DIV = DVC + DVP\).
  - Scaled by average net assets.

##### E. Market-to-Book Ratio (MB) as Proxy for Tobin's q
- **Definition**: Market value of net assets divided by book value of net assets.
- **Compustat Items**:
  - Book Value of Net Assets: As defined above (`NET_ASSETS`).
  - Market Value of Equity: `PRCC_F` (fiscal year-end price) × `CSHO` (shares outstanding).
  - Debt: As defined in `DEBT`.
- **CRSP Items**:
  - Alternative market value using CRSP: Use `PRC` (closing price) × `SHROUT` (shares outstanding) from CRSP monthly file, matched to fiscal year-end.
- **Formula**:
  \[
  MB = \frac{\text{Market Value of Equity} + DEBT}{NET_ASSETS}
  \]
- **Notes**: Use beginning-of-year MB (\(MB_{t-1}\)) in regressions to align with investment decisions.

##### F. Stock Returns
- **Definition**: Lagged annual stock returns to instrument for MB in IV regressions.
- **CRSP Items**:
  - Monthly returns: `RET` from CRSP monthly file.
  - Calculate annual returns for lags 1 to 4 (e.g., \(RET_{t-1}\), \(RET_{t-2}\), etc.) by compounding monthly returns over the fiscal year.
- **Purpose**: Used as instruments to correct for measurement error in MB.

##### G. Cash Holdings and Debt for Control Variables
- **Cash Holdings (CASH)**:
  - Compustat Item: `CHE`, scaled by contemporaneous net assets.
- **Debt (DEBT2)**:
  - Same as `DEBT` above, scaled by contemporaneous net assets.
- **Notes**: Use beginning-of-year values (\(CASH_{t-1}\), \(DEBT2_{t-1}\)) in regressions.

##### H. Financial Constraint Classification
- **Definition**: Sort firms into constrained and unconstrained groups based on forecasted free cash flow.
- **Methodology**:
  - Free Cash Flow (FCF): Cash flow (`CF`) minus capital expenditures (`CAPX1`) and other investments (`CAPX2` or `CAPX3`).
  - Forecast FCF using past 3 years’ data (e.g., sales growth, profits, cash flow, returns, cash holdings, debt, investment).
  - Criteria for Unconstrained Firms:
    - High and increasing sales (`SALE`), profits (`IBC`), cash flow (`CF`), returns (`RET`), and cash holdings (`CHE`).
    - Low and decreasing debt (`DEBT`) and investment (`CAPX1`).
    - FCF exceeds capital expenditures by ~11.5% of net assets and total investment by ~2.1% of net assets.
    - Cash holdings and net working capital exceed total liabilities.
    - Debt can be paid off with ~1 year of cash flow.
  - Sort firms annually into terciles (orಸ

#### 5. Regressions to Run
The study employs OLS and IV regressions to estimate investment-cash flow sensitivities, with corrections for measurement error in MB. Replicate the key regressions from Sections IV and V, focusing on Tables 3, 4, 6, 7, and A1.

##### A. OLS Regressions (Section IV, Tables 3 and 4)
- **Purpose**: Estimate baseline investment-cash flow sensitivities without correcting for measurement error in MB.
- **Specifications**:
  - **Table 3 (Full Sample)**:
    - Model 1: Regress each dependent variable (\(\Delta CASH\), \(\Delta NWC\), \(CAPX1\), \(CAPX2\), \(CAPX3\), \(\Delta DEBT\), \(ISSUES\), \(DIV\)) on \(CF_t\) and \(MB_{t-1}\).
    - Model 2: Add \(CF_{t-1}\).
    - Model 3: Add \(CASH_{t-1}\) and \(DEBT2_{t-1}\).
  - **Table 4 (Constrained vs. Unconstrained)**:
    - Repeat Model 1 for constrained and unconstrained subsamples.
- **Notes**:
  - Use firm and year fixed effects to control for unobserved heterogeneity.
  - Report slopes and t-statistics, ensuring slopes sum to ~1 due to accounting identity (Eq. 10).
  - Expected Results: Significant cash flow effects (e.g., $0.14 for \(\Delta NWC\), $0.35 for \(CAPX3\)) for the full sample, stronger for constrained firms ($0.72 vs. $0.30 for total investment).

##### B. IV Regressions (Section V, Tables 6, 7, A1)
- **Purpose**: Correct for measurement error in MB using instrumental variables.
- **First-Stage Regression (Table 5)**:
  - Regress \(MB_{t-1}\) on:
    - \(CF_t\), \(RET_{t-1}\), \(RET_{t-2}\), \(RET_{t-3}\), \(RET_{t-4}\).
    - For Table 7, include \(CASH_{t-1}\), \(DEBT2_{t-1}\).
  - Report \(R^2\) and slopes (e.g., \(CF_t\) slope ~5.18 for full sample, weaker for constrained firms).
- **Second-Stage Regressions**:
  - **Table 6 (Full Sample)**:
    - Model 1: Regress dependent variables on \(CF_t\) and instrumented \(MB_{t-1}^*\).
    - Model 2: Add \(CF_{t-1}\).
    - Model 3: Add \(CASH_{t-1}\), \(DEBT2_{t-1}\).
  - **Table 7 (Constrained vs. Unconstrained)**:
    - Repeat Models 1–3 for subsamples.
  - **Table A1 (Robustness Checks)**:
    - Model 1: Use \(CF_t\), \(CF_{t-1}\), ..., \(CF_{t-4}\) as instruments.
    - Model 2: Use \(CF_t\), \(CF_t^2\), \(CF_{t-1}\).
    - Model 3: Add \(MB_{t-2}\) (instrumented).
    - Model 4: Use \(CF_t\), \(CF_{t-1}\), \(RET_{t-4}\), \(RET_{t-5}\).
- **Notes**:
  - Use two-stage least squares (2SLS) for IV estimation.
  - Expected Results: Cash flow effects remain significant but reduced (e.g., $0.63 for constrained firms’ \(CAPX3\)), with larger reductions for unconstrained firms ($0.32).

#### 6. Methodological Considerations
- **Measurement Error in q**: Use IV approach to address MB’s noise as a proxy for q, validating instruments’ strength (high \(R^2\) in first-stage) and exogeneity.
- **Financial Constraints**: Validate the sorting methodology by checking unconstrained firms’ characteristics (e.g., high FCF, low debt).
- **Data Quality**: Verify SCF data accuracy, as noncash items (e.g., write-downs) increased noise in traditional measures post-1980s.
- **Robustness**:
  - Test alternative size cutoffs (e.g., NYSE 1st percentile).
  - Exclude low-PPE firms to focus on capital-intensive industries.
  - Use de-meaned data to check fixed effects’ impact.
- **Software**: Use Stata, R, or Python (e.g., `statsmodels`, `linearmodels`) for regressions, ensuring robust standard errors (e.g., Newey-West).

#### 7. Expected Outcomes
- Confirm strong investment-cash flow sensitivities, especially for constrained firms ($0.63 vs. $0.32 for fixed investment).
- Validate that expected cash flow drives investment more than unexpected cash flow ($0.68 vs. $0.12).
- Find that measurement error explains more of the sensitivity for unconstrained firms, supporting financing constraints’ role.
- Detect potential free-cash-flow problems in unconstrained firms (negative MB-investment relation).

#### 8. Timeline
- **Month 1**: Data collection and cleaning (Compustat, CRSP).
- **Month 2**: Variable construction and sample preparation.
- **Month 3**: Run OLS and IV regressions, validate results.
- **Month 4**: Conduct robustness checks, document findings.

#### 9. Potential Challenges
- **Data Availability**: Ensure SCF items (e.g., `FOPO`) are consistently reported post-2009.
- **Sample Size**: Address potential survivorship bias in recent years.
- **Constraint Classification**: Refine FCF forecasting if recent data alters firm characteristics.
- **Computational Complexity**: Optimize IV regressions for large datasets.

---

This plan provides a comprehensive roadmap to replicate Lewellen and Lewellen (2016) with a larger, more recent sample, leveraging the same data sources and methodological innovations. It ensures fidelity to the original study while extending its scope to test the robustness of findings in a modern context.