"""tables.py
Generate professional-looking LaTeX versions of replication result tables.

This utility scans a given directory (default: ``results/tables``) for CSV
files created by ``analyze_results.py`` and converts each file into a
booktabs-style LaTeX table.  When several CSVs share the same *table prefix*
(e.g. ``table7_...``) they are treated as parts of the same table and saved
as ``table7_1.tex``, ``table7_2.tex`` … in the
``results/tables/professional`` sub-folder.

A Typer CLI is provided so the script can be invoked from the command line:

    python -m src.tables generate  # uses default locations
    python -m src.tables generate --source-dir path/to/dir --out-dir path/out
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

import pandas as pd
import typer

###############################################################################
# ---------------------------  Helper utilities  --------------------------- #
###############################################################################

def _latex_from_df(df: pd.DataFrame, caption: str, label: str) -> str:
    """Return a booktabs-style LaTeX table string for *df*.

    Parameters
    ----------
    df      : The data to convert.
    caption : The LaTeX caption.
    label   : The LaTeX label used for cross-referencing.
    """
    float_fmt = lambda x: f"{x:,.3f}" if isinstance(x, (int, float)) else x  # noqa: E731
    col_fmt = "l" + "r" * (df.shape[1] - 1)  # left-align first col, right others

    return df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        float_format=float_fmt,
        column_format=col_fmt,
        multicolumn=True,
        multicolumn_format="c",
    )


def _group_csv_files(files: List[Path]) -> dict[str, List[Path]]:
    """Group CSV *files* by table prefix (e.g. ``table7``) and return mapping."""
    groups: dict[str, List[Path]] = {}
    prefix_re = re.compile(r"^(table\d+)")
    for f in files:
        m = prefix_re.match(f.stem)
        if not m:
            continue  # skip files without expected prefix
        groups.setdefault(m.group(1), []).append(f)
    # Sort file lists for deterministic output
    for lst in groups.values():
        lst.sort()
    return groups

###############################################################################
# -------------------------  Core public function  ------------------------- #
###############################################################################

def generate_professional_tables(source_dir: str | Path = "results/tables",
                                 out_dir: str | Path = "results/tables/professional") -> None:
    """Convert every CSV in *source_dir* to a polished LaTeX table in *out_dir*.

    The naming scheme follows the original paper.  If multiple CSVs share the
    same prefix (``tableX``) they will be enumerated using ``_1``, ``_2`` …
    regardless of how many there are.
    """
    src_path = Path(source_dir).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(src_path.glob("*.csv"))
    if not csv_files:
        typer.echo(f"No CSV files found in {src_path}")
        return

    groups = _group_csv_files(csv_files)
    if not groups:
        typer.echo("No files matching pattern 'table<digit>' found – nothing to do.")
        return

    for prefix, files in sorted(groups.items()):
        for idx, file in enumerate(files, start=1):
            df = pd.read_csv(file)
            caption = f"Table {prefix[5:]}" + (f" – part {idx}" if len(files) > 1 else "")
            label = f"tab:{prefix}_{idx}" if len(files) > 1 else f"tab:{prefix}"
            latex = _latex_from_df(df, caption, label)
            tex_name = f"{prefix}_{idx}.tex" if len(files) > 1 else f"{prefix}.tex"
            tex_path = out_path / tex_name
            tex_path.write_text(latex)
            typer.echo(f"Saved → {tex_path.relative_to(Path.cwd())}")

###############################################################################
# -------------------------------  Typer CLI  ------------------------------ #
###############################################################################

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def generate(
    source_dir: str = typer.Option("results/tables", help="Directory with raw CSV tables."),
    out_dir: str = typer.Option("results/tables/professional", help="Output directory for LaTeX tables."),
):
    """Generate professional LaTeX tables from *source_dir* into *out_dir*."""
    generate_professional_tables(source_dir, out_dir)


###############################################################################
# ----------------------  Specialized builders (paper)  --------------------- #
###############################################################################

_DEP_VAR_LABELS = {
    "capx1_scaled": "CAPX1",
    "capx2_scaled": "CAPX2",
    "capx3_scaled": "CAPX3",
    "delta_cash_scaled": "Delta Cash",
    "delta_nwc_scaled": "Delta NWC",
    "delta_debt_scaled": "Delta Debt",
    "issues_scaled": "Equity Issues",
    "div_scaled": "Dividends",
}

_VAR_ORDER = [
    "cash_flow_scaled",
    "cash_flow_scaled_lag",
    "cash_lag",
    "debt_lag",
    "mb_lag",
]

_STATS_ROWS = ["R-squared", "Observations"]


def _read_reg_csv(path: Path) -> pd.DataFrame:
    """Helper: read a regression CSV and return as dataframe."""
    return pd.read_csv(path)


def _collect_table3(source_dir: Path) -> dict[str, dict[int, pd.DataFrame]]:
    """Return nested mapping {dep_var: {model_no: df}} for Table-3 CSV files."""
    pattern = re.compile(r"table3_model(\d+)_(.+_scaled)\.csv$")
    data: dict[str, dict[int, pd.DataFrame]] = {}
    for p in source_dir.glob("table3_model*_*.csv"):
        m = pattern.search(p.name)
        if not m:
            continue
        model_no = int(m.group(1))
        dep_raw = m.group(2)
        # retain only variables we map, else skip
        if dep_raw not in _DEP_VAR_LABELS:
            continue
        data.setdefault(dep_raw, {})[model_no] = _read_reg_csv(p)
    return data


def _format_coeff(coef: float, se: float) -> str:
    """Return coefficient with standard error below in parentheses for LaTeX."""
    if pd.isna(coef):
        return ""
    return f"{coef:.3f}\\\\\n({se:.3f})"


def build_table3(source_dir: str | Path = "results/tables",
                 out_dir: str | Path = "results/tables/professional",
                 max_dep_vars_per_panel: int = 4) -> None:
    """Construct Table 3 in one or more LaTeX panels from raw CSVs.

    Parameters
    ----------
    source_dir : directory containing ``table3_model*_*.csv`` files.
    out_dir    : where the ``table3_1.tex`` (and possibly ``_2``) files go.
    max_dep_vars_per_panel : split the wide table after this many dependent
                             variables so each panel fits on a page.
    """
    src = Path(source_dir).expanduser().resolve()
    dest = Path(out_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    collected = _collect_table3(src)
    if not collected:
        typer.echo("No Table-3 CSV files found – nothing to build.")
        return

    # Ensure deterministic order by dependent-variable label
    dep_vars = sorted(collected.keys(), key=lambda k: _DEP_VAR_LABELS.get(k, k))

    # Prepare column MultiIndex
    column_tuples = []  # (dep_var_label, model)
    
    # Gather raw data
    coef_data = {var: [] for var in _VAR_ORDER}
    se_data = {var: [] for var in _VAR_ORDER}
    r_squared = []
    n_obs = []

    for dep in dep_vars:
        for model in sorted(collected[dep].keys()):
            df = collected[dep][model]
            col_label = (_DEP_VAR_LABELS[dep], f"Model {model}")
            column_tuples.append(col_label)

            # Extract coefficients and standard errors
            for var in _VAR_ORDER:
                subset = df[df["variable"] == var]
                if subset.empty:
                    coef_data[var].append("")
                    se_data[var].append("")
                else:
                    coef_data[var].append(f"{subset['coefficient'].iloc[0]:.3f}")
                    se_data[var].append(f"({subset['std_error'].iloc[0]:.3f})")

            # Store statistics
            r_squared.append(f"{df['r_squared'].iloc[0]:.3f}")
            n_obs.append(f"{int(df['n_obs'].iloc[0]):,}")

    # Build DataFrame with alternating coefficient and SE rows
    columns = pd.MultiIndex.from_tuples(column_tuples, names=["Dependent", "Model"])
    
    # Prepare rows (empty for SE rows)
    rows = []
    row_names = []
    
    # Nice row label formatting
    nice_names = {
        "cash_flow_scaled": "Cash flow",
        "cash_flow_scaled_lag": "Cash flow (t-1)",
        "cash_lag": "Cash / Assets (t-1)",
        "debt_lag": "Debt / Assets (t-1)",
        "mb_lag": "Market-to-Book (t-1)",
    }
    
    # Add coefficient and SE rows alternately
    for var in _VAR_ORDER:
        # Coefficient row with label
        rows.append(coef_data[var])
        row_names.append(nice_names.get(var, var))
        
        # Standard error row without label
        rows.append(se_data[var])
        row_names.append("")  # Empty label for SE rows
    
    # Add statistics at the end
    rows.append(r_squared)
    row_names.append("R-squared")
    
    rows.append(n_obs)
    row_names.append("Observations")
    
    # Construct DataFrame
    df = pd.DataFrame(rows, index=row_names, columns=columns)
    
    # Split into panels if too wide
    panels = []
    for i in range(0, len(dep_vars), max_dep_vars_per_panel):
        subvars = dep_vars[i:i + max_dep_vars_per_panel]
        mask = df.columns.get_level_values(0).isin([_DEP_VAR_LABELS[d] for d in subvars])
        panels.append(df.loc[:, mask])

    # Save each panel
    for idx, panel in enumerate(panels, start=1):
        col_fmt = "l" + "r" * panel.shape[1]  # left for index, right for data
        tex = panel.to_latex(
            escape=False,
            multicolumn=True,
            multicolumn_format="c",
            column_format=col_fmt,
        )
        out_file = dest / f"table3_{idx}.tex"
        out_file.write_text(tex, encoding='utf-8')
        typer.echo(f"Saved Table 3 panel → {out_file.relative_to(Path.cwd())}")

###############################################################################
# -------------------------------  Typer CLI  ------------------------------ #
###############################################################################

@app.command(name="table3")
def table3_cmd(
    source_dir: str = typer.Option("results/tables", help="Directory with raw Table-3 CSVs."),
    out_dir: str = typer.Option("results/tables/professional", help="Output directory for LaTeX tables."),
    max_per_panel: int = typer.Option(4, help="Number of dependent variables per panel."),
):
    """Recreate Table 3 panels to match the original paper layout."""
    build_table3(source_dir, out_dir, max_per_panel)


if __name__ == "__main__":
    app() 