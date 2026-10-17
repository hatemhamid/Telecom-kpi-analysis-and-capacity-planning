#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAN KPI Analysis & Capacity Planning Pipeline
=============================================================

Processes CS/PS KPI data and Macro Network Traffic to provide:
1. Automated Outlier & Trend Analysis with specific Engineering Recommendations.
2. Accurate PRB Utilization Forecasting using IQR-cleaned OLS Regression.
3. Automated Hardware Expansion Planning based on a 70% PRB threshold.
"""
from __future__ import annotations

# Standard Library Imports
import argparse
import os
import re
import sys
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Data Science & Math Imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Visualization & Formatting Imports
import seaborn as sns
from tabulate import tabulate

# Forecasting Imports
import xgboost as xgb
from prophet import Prophet

# Matplotlib Headless Server Configuration
import matplotlib
if sys.platform.startswith("linux") and not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

class OperationCancelled(Exception):
    """Raised when the user cancels a long-running operation."""
    pass

@dataclass
class KPIAnalysisResult:
    images: List[Path]
    correlation: Optional[pd.DataFrame]
    regression: Optional[object]
    filters: str

@dataclass
class CapacityPlanningResult:
    best_model: str
    multiplier: float
    expansion_table: pd.DataFrame
    n_sites: int
    regression_summary: pd.DataFrame
    baseline_periods: List[str]
    n_inactive_sites: int
    csv_path: Optional[Path]
    unrecognised_configs: Dict[str, int]
    model_comparison: pd.DataFrame
    fold_metrics: pd.DataFrame
    forecast_table: pd.DataFrame
    multiplier_table: pd.DataFrame
    site_stats: pd.DataFrame
    expansion_summary: pd.DataFrame
    expansion_by_region: pd.DataFrame
    images: List[Path]


# ============================================================================
# Configuration & Schema
# ============================================================================
class Config:
    EXCEL_FILE = "Project Data - Traffic & KPIs.xlsx"
    OUTPUT_CSV = "Site_Expansion_Plan_2026.csv"
    PLOTS_DIRNAME = "plots"

    RATE_CAP = 100.0              
    TRAFFIC_WINSOR_PCT = 0.995    
    DROP_RATE_WINSOR_PCT = 0.99   
    MODZ_THRESHOLD = 3.5          

    FORECAST_END = "2026-12-31"
    BASELINE_WINDOW_MONTHS = 6    
    PRB_THRESHOLD_PCT = 70.0       

    PREVIEW_ROWS = 25
    PLOT_SAMPLE = 20000           
    CONSOLE_WIDTH = 100


class Col:
    SITE = "Site ID"
    YM = "Y-M"
    REGION = "Region"
    AVAIL = "Availability"
    CDR = "CDR"
    CSSR = "CSSR"
    CS_TRAFFIC = "CS Traffic - Total"
    ICM = "ICM"
    DATA_TRAFFIC = "Data Traffic (TB)"
    DATA_CSSR = "Data CSSR"
    THROUGHPUT = "Throughput (Mbps)"
    ERAB_DR = "ERAB DR"
    HO_SR = "HO SR"
    PRB = "PRB DL Utilization"
    CONFIG = "Configuration"
    DATE = "Date"
    DAILY_TRAFFIC = "Traffic (TBs Daily)"
    MONTH = "Month"              
    MONTH_START = "Month_Start"  
    OUTAGE = "Outage_Flag"

CS_COLUMNS = [Col.SITE, Col.YM, Col.REGION, Col.AVAIL, Col.CDR, Col.CSSR, Col.CS_TRAFFIC, Col.ICM]
CS_NUMERIC = [Col.AVAIL, Col.CDR, Col.CSSR, Col.CS_TRAFFIC, Col.ICM]
CS_RATE_KPIS = [Col.AVAIL, Col.CSSR]

PS_COLUMNS = [Col.SITE, Col.YM, Col.REGION, Col.DATA_TRAFFIC, Col.DATA_CSSR, Col.THROUGHPUT,
              Col.ERAB_DR, Col.HO_SR, Col.PRB]
PS_NUMERIC = [Col.DATA_TRAFFIC, Col.DATA_CSSR, Col.THROUGHPUT, Col.ERAB_DR, Col.HO_SR, Col.PRB]
PS_RATE_KPIS = [Col.DATA_CSSR, Col.PRB, Col.HO_SR]

SITES_COLUMNS = [Col.SITE, Col.CONFIG]
TRAFFIC_COLUMNS = [Col.DATE, Col.DAILY_TRAFFIC]


class DataValidationError(Exception):
    pass


def _norm_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _build_alias_map() -> Dict[str, str]:
    canonical = [
        Col.SITE, Col.YM, Col.REGION, Col.AVAIL, Col.CDR, Col.CSSR, Col.CS_TRAFFIC, Col.ICM,
        Col.DATA_TRAFFIC, Col.DATA_CSSR, Col.THROUGHPUT, Col.ERAB_DR, Col.HO_SR, Col.PRB,
        Col.CONFIG, Col.DATE, Col.DAILY_TRAFFIC,
    ]
    alias = {_norm_name(c): c for c in canonical}
    alias.update({  
        "site": Col.SITE, "month": Col.YM, "yearmonth": Col.YM, "ymonth": Col.YM,
        "cstraffic": Col.CS_TRAFFIC, "datatraffic": Col.DATA_TRAFFIC,
        "throughput": Col.THROUGHPUT, "hosr": Col.HO_SR, "ps4g3ghosr": Col.HO_SR,
        "prb": Col.PRB, "prbutilization": Col.PRB, "prbdlutilization": Col.PRB,
        "config": Col.CONFIG, "siteconfiguration": Col.CONFIG, 
        "confgurations": Col.CONFIG, "configurations": Col.CONFIG, 
        "dailytraffic": Col.DAILY_TRAFFIC, "traffic": Col.DAILY_TRAFFIC,
    })
    return alias

_ALIASES = _build_alias_map()


def to_numeric_clean(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype(str).str.replace(r"[%,\s]", "", regex=True)
        series = series.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    return pd.to_numeric(series, errors="coerce").astype(float)


def clean_text(series: pd.Series) -> pd.Series:
    raw_missing = series.isna()
    txt = series.astype(str).str.strip()
    bad = raw_missing | (txt == "") | (txt.str.lower().isin(["nan", "none", "null"]))
    txt = txt.astype(object)
    txt[bad] = np.nan
    return txt


_MONTH_FORMATS = [
    "%Y-%m", "%Y/%m", "%Y%m", "%b-%y", "%b-%Y", "%b %Y", "%B-%Y", "%B %Y",
    "%y-%b", "%m/%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
]


def parse_year_month(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.to_period("M").dt.to_timestamp()
    txt = series.astype(str).str.strip()
    best, best_rate = None, -1.0
    for fmt in _MONTH_FORMATS:
        parsed = pd.to_datetime(txt, format=fmt, errors="coerce")
        rate = float(parsed.notna().mean())
        if rate > best_rate:
            best, best_rate = parsed, rate
        if rate == 1.0:
            break
    if best_rate < 0.9:  
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            generic = pd.to_datetime(txt, errors="coerce")
        if float(generic.notna().mean()) > best_rate:
            best = generic
    return best.dt.to_period("M").dt.to_timestamp()


def parse_daily_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    txt = series.astype(str).str.strip()
    candidates = []
    for dayfirst in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(txt, errors="coerce", dayfirst=dayfirst)
        valid = round(float(parsed.notna().mean()), 3)
        continuity = float((parsed.diff().abs().dt.days == 1).mean())
        candidates.append((valid, continuity, parsed))
    return max(candidates, key=lambda c: (c[0], c[1]))[2]


class ConsoleUI:
    @staticmethod
    def banner(title: str) -> None:
        w = Config.CONSOLE_WIDTH
        print("\n" + "=" * w)
        print(title.center(w))
        print("=" * w)

    @staticmethod
    def section(title: str) -> None:
        w = Config.CONSOLE_WIDTH
        print("\n" + "-" * w)
        print(" " + title)
        print("-" * w)

    @staticmethod
    def info(msg: str) -> None:
        print("  " + msg)

    @staticmethod
    def warn(msg: str) -> None:
        print("  [WARNING] " + msg)

    @staticmethod
    def table(df: pd.DataFrame, title: Optional[str] = None, index: bool = False,
              floatfmt: str = ".3f", max_rows: Optional[int] = None, numparse: bool = True) -> None:
        if title:
            print("\n" + title)
        shown = df if max_rows is None else df.head(max_rows)
        print(tabulate(shown, headers="keys", tablefmt="psql", showindex=index,
                       floatfmt=floatfmt, disable_numparse=not numparse))
        if max_rows is not None and len(df) > max_rows:
            print("  ... showing first {:,} of {:,} rows".format(max_rows, len(df)))

    @staticmethod
    def ask_choice(prompt: str, options: Dict[str, str]) -> str:
        keys = {k.upper(): k for k in options}
        print("\n" + prompt)
        for k, desc in options.items():
            print("   [{}] {}".format(k, desc))
        while True:
            ans = input("  > ").strip().upper()
            if ans in keys:
                return ans
            print("  Invalid choice. Please enter one of: {}".format(", ".join(options)))

    @staticmethod
    def ask_multi(label: str, options: Sequence[str]) -> Optional[List[str]]:
        options = list(options)
        print("\nSelect {} (comma-separated numbers / names, or 'All'):".format(label))
        for i, opt in enumerate(options, 1):
            print("   {:>3}. {}".format(i, opt))
        lookup = {o.lower(): o for o in options}
        while True:
            ans = input("  > ").strip()
            if ans == "" or ans.lower() == "all":
                return None
            chosen: List[str] = []
            ok = True
            for tok in [t.strip() for t in re.split(r"[,;]", ans) if t.strip()]:
                if tok.lower() in lookup:
                    chosen.append(lookup[tok.lower()])
                elif re.fullmatch(r"\d+", tok) and 1 <= int(tok) <= len(options):
                    chosen.append(options[int(tok) - 1])
                elif re.fullmatch(r"\d+\s*-\s*\d+", tok):
                    a, b = [int(x) for x in re.split(r"\s*-\s*", tok)]
                    if 1 <= a <= b <= len(options):
                        chosen.extend(options[a - 1:b])
                    else:
                        ok = False
                else:
                    ok = False
            if ok and chosen:
                return list(dict.fromkeys(chosen)) 
            print("  Could not understand that selection - please try again.")


class Plotter:
    def __init__(self, out_dir: Path, show: bool = True) -> None:
        self.out_dir = out_dir
        self.show = show
        sns.set_theme(style="whitegrid")

    def finish(self, fig: plt.Figure, filename: str) -> None:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            ConsoleUI.info("[plot saved] {}".format(path))
        except OSError as exc:
            ConsoleUI.warn("Could not save plot {}: {}".format(filename, exc))
        if self.show and matplotlib.get_backend().lower() != "agg":
            plt.show()
        plt.close(fig)


# ============================================================================
# Excel Data Loading & Cleaning
# ============================================================================
class DataLoader:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def load_all(self) -> Dict[str, pd.DataFrame]:
        if not self.file_path.is_file():
            raise FileNotFoundError("Input Excel file not found: {}".format(self.file_path))
            
        ConsoleUI.info(f"Reading Excel workbook: {self.file_path.name} ...")
        try:
            xls = pd.ExcelFile(self.file_path)
        except Exception as exc:
            raise DataValidationError(f"Could not read Excel file: {exc}") from exc
            
        return {
            "cs": self._read(xls, "CS KPIs", CS_COLUMNS),
            "ps": self._read(xls, "PS KPIs", PS_COLUMNS),
            "sites": self._read(xls, "Sites Configurations", SITES_COLUMNS),
            "traffic": self._read(xls, "Network Traffic", TRAFFIC_COLUMNS),
        }

    def _read(self, xls: pd.ExcelFile, sheet_name: str, required: List[str]) -> pd.DataFrame:
        if sheet_name not in xls.sheet_names:
            raise DataValidationError(f"Sheet '{sheet_name}' is missing from the Excel workbook.")
            
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # FOOLPROOF FIX: Force columns for the "Sites Configurations" sheet due to Excel typos
        if sheet_name == "Sites Configurations" and len(df.columns) >= 2:
            df.columns = [Col.SITE, Col.CONFIG] + list(df.columns[2:])
        else:
            # Apply strict column renaming mapping for other sheets
            df = df.rename(columns={c: _ALIASES.get(_norm_name(c), c) for c in df.columns})
            
        df = df.loc[:, ~df.columns.duplicated()]
        
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataValidationError(
                "Sheet '{}' is missing required column(s): {}.\n  Columns found: {}".format(
                    sheet_name, missing, list(df.columns)))
                    
        ConsoleUI.info("Loaded {:<25s} {:>9,} rows".format(f"[{sheet_name}]", len(df)))
        return df[required].copy()


class DataCleaner:
    def __init__(self) -> None:
        self.log: List[dict] = []

    def _log(self, dataset: str, step: str, column: str, count: int) -> None:
        if count:
            self.log.append({"Dataset": dataset, "Step": step, "Column": column,
                             "Rows affected": int(count)})

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(self.log, columns=["Dataset", "Step", "Column", "Rows affected"])

    @staticmethod
    def modified_zscore(x: pd.Series) -> pd.Series:
        med = x.median()
        dev = (x - med).abs()
        mad = dev.median()
        if mad > 0:
            return 0.6745 * (x - med) / mad
        mean_ad = dev.mean()
        if mean_ad > 0:
            return (x - med) / (1.253314 * mean_ad)
        return pd.Series(0.0, index=x.index)

    def _winsorize_traffic(self, df: pd.DataFrame, dataset: str, col: str) -> None:
        if df[col].notna().sum() == 0:
            return
        cap = df[col].quantile(Config.TRAFFIC_WINSOR_PCT)
        mask = df[col] > cap
        self._log(dataset, "Winsorise @ P{:g}".format(Config.TRAFFIC_WINSOR_PCT * 100), col, mask.sum())
        df[col] = df[col].clip(upper=cap)

    def _winsorize_drop_rate(self, df: pd.DataFrame, dataset: str, col: str) -> None:
        if df[col].notna().sum() == 0:
            return
        cap = df[col].quantile(Config.DROP_RATE_WINSOR_PCT)
        flagged = self.modified_zscore(df[col]) > Config.MODZ_THRESHOLD
        new_vals = df[col].where(~flagged, np.minimum(df[col], cap))
        self._log(dataset, "Cap modZ>{:g} @ P{:g}".format(Config.MODZ_THRESHOLD,
                                                         Config.DROP_RATE_WINSOR_PCT * 100),
                  col, (new_vals != df[col]).sum())
        df[col] = new_vals

    def _base_clean(self, df: pd.DataFrame, dataset: str, numeric: List[str],
                    rate_kpis: List[str]) -> pd.DataFrame:
        df = df.copy()
        df[Col.SITE] = clean_text(df[Col.SITE])
        df[Col.REGION] = clean_text(df[Col.REGION])
        n0 = len(df)
        df = df.dropna(subset=[Col.SITE])
        self._log(dataset, "Dropped: missing Site ID", Col.SITE, n0 - len(df))

        df[Col.MONTH_START] = parse_year_month(df[Col.YM])
        n0 = len(df)
        df = df.dropna(subset=[Col.MONTH_START])
        self._log(dataset, "Dropped: unparseable Y-M", Col.YM, n0 - len(df))
        df[Col.MONTH] = df[Col.MONTH_START].dt.strftime("%Y-%m")

        for c in numeric:
            df[c] = to_numeric_clean(df[c])

        dup = df.duplicated([Col.SITE, Col.MONTH_START], keep="last")
        self._log(dataset, "Dropped: duplicate Site/Month", Col.SITE, dup.sum())
        df = df[~dup].sort_values([Col.SITE, Col.MONTH_START]).reset_index(drop=True)

        for c in rate_kpis:
            mask = df[c] > Config.RATE_CAP
            self._log(dataset, "Capped at {:g}%".format(Config.RATE_CAP), c, mask.sum())
            df[c] = df[c].clip(upper=Config.RATE_CAP)
        for c in numeric:
            mask = df[c] < 0
            self._log(dataset, "Floored at 0", c, mask.sum())
            df[c] = df[c].clip(lower=0.0)

        df[Col.REGION] = df.groupby(Col.SITE)[Col.REGION].ffill()
        df[Col.REGION] = df.groupby(Col.SITE)[Col.REGION].bfill()
        df[Col.REGION] = df[Col.REGION].fillna("Unknown")
        for c in numeric:
            n_missing = int(df[c].isna().sum())
            if not n_missing:
                continue
            site_med = df.groupby(Col.SITE)[c].transform("median")
            reg_med = df.groupby(Col.REGION)[c].transform("median")
            df[c] = df[c].fillna(site_med).fillna(reg_med).fillna(df[c].median())
            self._log(dataset, "Imputed (site>region>global median)", c, n_missing)
        return df

    def clean_cs(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._base_clean(df, "CS", CS_NUMERIC, CS_RATE_KPIS)
        out[Col.OUTAGE] = (out[Col.CS_TRAFFIC] == 0) | (out[Col.AVAIL] == 0)
        self._winsorize_drop_rate(out, "CS", Col.CDR)
        return out

    def clean_ps(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._base_clean(df, "PS", PS_NUMERIC, PS_RATE_KPIS)
        # Preserve the absolute raw Data Traffic for planning calculations before any clipping/winsorizing
        out["Raw_Data_Traffic"] = out[Col.DATA_TRAFFIC].copy()
        
        # Use PRB and Data Traffic to flag outage
        out[Col.OUTAGE] = (out[Col.PRB] == 0) | (out[Col.DATA_TRAFFIC] == 0)
        self._winsorize_traffic(out, "PS", Col.DATA_TRAFFIC)
        self._winsorize_drop_rate(out, "PS", Col.ERAB_DR)
        return out


def preprocess_kpi_data(cs_raw: pd.DataFrame, ps_raw: pd.DataFrame,
                        verbose: bool = True, return_report: bool = False) -> Tuple:
    cleaner = DataCleaner()
    cs = cleaner.clean_cs(cs_raw)
    ps = cleaner.clean_ps(ps_raw)
    rep = cleaner.report()
    if verbose:
        ConsoleUI.section("DATA CLEANING REPORT")
        ConsoleUI.info("CS rows after cleaning: {:,}  |  PS rows after cleaning: {:,}".format(
            len(cs), len(ps)))
        ConsoleUI.info("Outage/inactive rows kept & flagged -> CS: {:,}  PS: {:,}".format(
            int(cs[Col.OUTAGE].sum()), int(ps[Col.OUTAGE].sum())))
        if rep.empty:
            ConsoleUI.info("No corrections were necessary.")
        else:
            ConsoleUI.table(rep, floatfmt=".0f")
    if return_report:
        return cs, ps, rep
    return cs, ps


def clean_sites(sites: pd.DataFrame) -> pd.DataFrame:
    out = sites.copy()
    out[Col.SITE] = clean_text(out[Col.SITE])
    out[Col.CONFIG] = clean_text(out[Col.CONFIG])
    out = out.dropna(subset=[Col.SITE])
    dup = out.duplicated(Col.SITE, keep="first")
    if dup.any():
        ConsoleUI.warn("{:,} duplicate Site IDs in the configuration file - keeping the first.".format(
            int(dup.sum())))
    return out[~dup].reset_index(drop=True)


def build_monthly_series(traffic: pd.DataFrame) -> pd.Series:
    d = traffic[[Col.DATE, Col.DAILY_TRAFFIC]].copy()
    d[Col.DATE] = parse_daily_dates(d[Col.DATE])
    d[Col.DAILY_TRAFFIC] = to_numeric_clean(d[Col.DAILY_TRAFFIC])
    d = d.dropna(subset=[Col.DATE])
    
    try:
        d[Col.DATE] = d[Col.DATE].dt.to_period("M").dt.to_timestamp("ME")
    except ValueError:
        d[Col.DATE] = d[Col.DATE].dt.to_period("M").dt.to_timestamp("M")
        
    s = d.groupby(Col.DATE)[Col.DAILY_TRAFFIC].mean().sort_index()
    
    try:
        expected_dates = pd.date_range(start=s.index.min(), end=s.index.max(), freq="ME")
    except ValueError:
        expected_dates = pd.date_range(start=s.index.min(), end=s.index.max(), freq="M")
        
    s = s.reindex(expected_dates)
    n_gaps = int(s.isna().sum())
    s = s.interpolate(method="linear", limit_direction="both").clip(lower=0.0)
    
    if n_gaps:
        ConsoleUI.info(f"Network traffic: {n_gaps} missing month(s) linearly interpolated.")
        
    s.index.name = Col.DATE
    s.name = Col.DAILY_TRAFFIC
    
    if len(s) < 24:
        raise DataValidationError(
            f"Network_Traffic holds only {len(s)} months - at least 24 are needed to forecast."
        )
    return s


# ============================================================================
# KPI Analysis & Outlier Detection
# ============================================================================
@dataclass
class FilterSelection:
    family: str                     
    months: Optional[List[str]]     
    regions: Optional[List[str]]    

    def describe(self) -> str:
        return "KPI family={} | months={} | regions={}".format(
            self.family,
            "All" if self.months is None else ", ".join(self.months),
            "All" if self.regions is None else ", ".join(self.regions))


def apply_filters(df: pd.DataFrame, sel: FilterSelection) -> pd.DataFrame:
    out = df
    if sel.months is not None:
        out = out[out[Col.MONTH].isin(sel.months)]
    if sel.regions is not None:
        out = out[out[Col.REGION].isin(sel.regions)]
    return out

#Screen

class KPIAnalyzer:
    def __init__(self, plotter: Plotter) -> None:
        self.plotter = plotter

    def generate_insights(self, kpi_name: str, data: pd.Series) -> None:
        """Automated Insights Engine for advanced statistical summaries & recommendations."""
        data = data.dropna()
        if data.empty:
            ConsoleUI.info("No data available to generate insights.")
            return

        mean_val = data.mean()
        median_val = data.median()
        std_dev = data.std()
        p95 = data.quantile(0.95)
        p05 = data.quantile(0.05)
        skew = data.skew()
        
        ConsoleUI.section(f"INSIGHTS & RECOMMENDATIONS: {kpi_name}")
        ConsoleUI.info(f"Mean: {mean_val:.2f} | Median: {median_val:.2f} | Std Dev: {std_dev:.2f}")
        
        if skew > 1:
            ConsoleUI.info("Distribution: Highly Right-Skewed (Outliers primarily on the high end).")
            ConsoleUI.info(f"Top 5% of sites exceed: {p95:.2f}")
        elif skew < -1:
            ConsoleUI.info("Distribution: Highly Left-Skewed (Outliers primarily on the low end).")
            ConsoleUI.info(f"Bottom 5% of sites drop below: {p05:.2f}")
        else:
            ConsoleUI.info("Distribution: Relatively balanced.")
            
        print()
        
        kpi_lower = kpi_name.lower()
        if "throughput" in kpi_lower:
            ConsoleUI.info("Recommendation: High-traffic outlier sites degrading below target throughput need immediate capacity expansion (e.g., adding TDD carriers or M-MIMO). Investigate transmission bottlenecks.")
        elif "drop" in kpi_lower or "dr" in kpi_lower:
            ConsoleUI.info("Recommendation: Outliers with high drop rates indicate severe RF issues. Conduct Drive Testing around these sites to check for missing neighbors, overshooting, or external interference.")
        elif "cssr" in kpi_lower:
            ConsoleUI.info("Recommendation: Low Call Setup Success Rate outliers suggest high congestion or parameter misconfigurations (e.g., PRACH planning). Check control channel utilization.")
        elif "avail" in kpi_lower:
            ConsoleUI.info("Recommendation: Sites with unusually low availability have hardware or power failures. Cross-reference these outliers with the FM (Fault Management) alarms logs.")
        elif "traffic" in kpi_lower:
            ConsoleUI.info("Recommendation: Extreme traffic outliers identify your core revenue-generating clusters. Prioritize these areas for upcoming hardware modernizations and CAPEX allocation.")
        elif "prb" in kpi_lower:
            ConsoleUI.info("Recommendation: Sites with PRB Utilization approaching 100% are severely congested. Consider load balancing via parameter tuning (e.g., adjusting cell reselection offsets) before hardware expansion.")
        else:
            ConsoleUI.info("Recommendation: Review the plotted outliers to determine if they correlate with known network incidents, special events, or structural changes in the coverage footprint.")

    def plot_kpi_analysis_charts(self, df: pd.DataFrame, kpi_name: str, numeric_cols: list) -> None:
        """Plots combined boxplot/trendline, followed by separate scatter plots vs other KPIs."""
        ConsoleUI.section(f"KPI ANALYSIS VISUALIZATIONS - {kpi_name}")
        
        if kpi_name not in df.columns or Col.MONTH_START not in df.columns:
            ConsoleUI.warn(f"Target KPI '{kpi_name}' or Month column not found.")
            return

        valid_df = df.dropna(subset=[kpi_name, Col.MONTH_START]).copy()
        if valid_df.empty:
            ConsoleUI.warn("No valid data points available for this KPI.")
            return

        valid_df = valid_df.sort_values(Col.MONTH_START)

        # 1. Combined Chart: Boxplot + Trend Line Plot
        fig_combined, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.boxplot(y=valid_df[kpi_name], ax=axes[0], color="skyblue")
        axes[0].set_title(f"Distribution & Outliers", fontsize=12)
        axes[0].set_ylabel(kpi_name)

        # Check if only one specific month is selected
        unique_months = valid_df[Col.MONTH_START].nunique()
        
        if unique_months == 1:
            # Plot KPI per Region instead of Over Time
            sns.barplot(
                data=valid_df, x=Col.REGION, y=kpi_name, 
                ax=axes[1], color="crimson", errorbar=('ci', 95)
            )
            axes[1].set_title(f"Distribution per Region", fontsize=12)
            axes[1].set_xlabel("Region")
        else:
            # Plot Trend Over Time as usual
            sns.lineplot(
                data=valid_df, x=Col.MONTH_START, y=kpi_name, 
                ax=axes[1], color="crimson", marker="o", errorbar=('ci', 95)
            )
            axes[1].set_title(f"Trend Over Time", fontsize=12)
            axes[1].set_xlabel("Month")
            
        axes[1].set_ylabel(kpi_name)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].tick_params(axis='x', rotation=45)

        fig_combined.suptitle(f"KPI Analysis: {kpi_name}", fontsize=14, fontweight='bold')
        fig_combined.tight_layout()
        self.plotter.finish(fig_combined, f"combined_analysis_{_norm_name(kpi_name)}.png")

        # 2. Individual Scatter Plots vs Other KPIs in the same sheet
        other_kpis = [
            c for c in numeric_cols 
            if c != kpi_name and c in valid_df.columns and c not in (Col.OUTAGE, "Outage_Flag")
        ]
        
        for other_kpi in other_kpis:
            plot_data = valid_df[[kpi_name, other_kpi]].dropna()
            
            if plot_data.empty:
                continue
                
            if len(plot_data) > Config.PLOT_SAMPLE:
                plot_data = plot_data.sample(Config.PLOT_SAMPLE, random_state=42)

            fig_scatter, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(plot_data[other_kpi], plot_data[kpi_name], alpha=0.3, s=8)
            
            if len(plot_data) >= 3 and plot_data[other_kpi].nunique() > 1:
                m, b = np.polyfit(plot_data[other_kpi], plot_data[kpi_name], 1)
                xs = np.linspace(plot_data[other_kpi].min(), plot_data[other_kpi].max(), 100)
                ax.plot(xs, m * xs + b, color="crimson", lw=2)

            ax.set_xlabel(other_kpi)
            ax.set_ylabel(kpi_name)
            ax.set_title(f"{kpi_name} vs {other_kpi}")
            
            fig_scatter.tight_layout()
            self.plotter.finish(fig_scatter, f"scatter_{_norm_name(kpi_name)}_vs_{_norm_name(other_kpi)}.png")

        self.generate_insights(kpi_name, valid_df[kpi_name])

    def bad_sites_analysis(self, ps: pd.DataFrame, sites: pd.DataFrame, out_dir: Path, threshold: float = 5.0) -> Optional[pd.DataFrame]:
        ConsoleUI.section(f"BAD SITES ANALYSIS (Average Throughput <= {threshold} Mbps)")
        if ps.empty:
            ConsoleUI.warn("No PS rows available after applying filters.")
            return None

        site_metric = ps.groupby(Col.SITE)[Col.THROUGHPUT].mean().reset_index()
        bad_sites = site_metric[site_metric[Col.THROUGHPUT] <= threshold].copy()

        if bad_sites.empty:
            ConsoleUI.info(f"No sites have an average Throughput <= {threshold} Mbps.")
            return None

        latest_region = ps.sort_values(Col.MONTH_START).groupby(Col.SITE)[Col.REGION].last().reset_index()
        bad_sites = bad_sites.merge(latest_region, on=Col.SITE, how="left")
        bad_sites = bad_sites.merge(sites[[Col.SITE, Col.CONFIG]], on=Col.SITE, how="left")

        table = pd.DataFrame({
            "Site ID": bad_sites[Col.SITE],
            "Region": bad_sites[Col.REGION],
            "Current Config": bad_sites[Col.CONFIG].fillna("N/A"),
            "Average Throughput (Mbps)": bad_sites[Col.THROUGHPUT].round(3)
        }).sort_values("Average Throughput (Mbps)", ascending=True).reset_index(drop=True)

        ConsoleUI.info("Found {:,} bad site(s).".format(len(table)))
        ConsoleUI.table(table, title=f"Bad Sites Table (worst Throughput first, <= {threshold} Mbps)",
                        floatfmt=".3f", max_rows=Config.PREVIEW_ROWS)

        out_path = out_dir / "Bad_Sites_Throughput.csv"
        try:
            table.to_csv(out_path, index=False)
            ConsoleUI.info("[saved] {}  ({:,} rows)".format(out_path, len(table)))
        except OSError as exc:
            ConsoleUI.warn("Could not write {}: {}".format(out_path, exc))
            
        return table


# ============================================================================
# Regression & Capacity Planning
# ============================================================================
class PRBRegressor:
    def __init__(self) -> None:
        self.model = None
        self.results = None
        self.n_used = 0
        self.r2 = np.nan
        self.rmse = np.nan
        self.fitted = False
        self._slope = np.nan
        self._intercept = np.nan

    def fit(self, ps: pd.DataFrame, verbose: bool = True) -> "PRBRegressor":
        data = ps[[Col.DATA_TRAFFIC, Col.PRB]].dropna().copy()
        
        if len(data) < 3 or data[Col.DATA_TRAFFIC].nunique() < 2:
            raise ValueError("Not enough varied data points to fit the PRB regression.")
        
        # 1. Initial OLS Implementation (Before Outlier Removal)
        data['intercept'] = 1
        X_initial = data[['intercept', Col.DATA_TRAFFIC]]
        y_initial = data[Col.PRB]
        
        model_initial = sm.OLS(y_initial, X_initial)
        results_initial = model_initial.fit()
            
        # 2. Outlier Removal (using IQR bounds on the residuals)
        residuals = results_initial.resid
        Q1 = residuals.quantile(0.25)
        Q3 = residuals.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        clean_mask = (residuals >= lower_bound) & (residuals <= upper_bound)
        clean_data = data[clean_mask].copy()
        outliers_removed = len(data) - len(clean_data)
        
        if verbose:
            ConsoleUI.info(f"\nRemoved {outliers_removed} outlier(s) using the IQR method to fit the final model.")
            
        # 3. Final OLS Fit (Strictly on Cleaned Data)
        X_clean = clean_data[['intercept', Col.DATA_TRAFFIC]]
        y_clean = clean_data[Col.PRB]
        
        self.model = sm.OLS(y_clean, X_clean)
        self.results = self.model.fit()
        
        if verbose:
            ConsoleUI.section("OLS REGRESSION SUMMARY (CLEANED DATA)")
            print(self.results.summary())
        
        self._intercept = self.results.params['intercept']
        self._slope = self.results.params[Col.DATA_TRAFFIC]
        self.r2 = self.results.rsquared
        
        pred = self.results.predict(X_clean)
        self.rmse = float(np.sqrt(mean_squared_error(y_clean, pred)))
        
        self.n_used = len(clean_data)
        self.fitted = True
        return self

    @property
    def slope(self) -> float:
        return float(self._slope)

    @property
    def intercept(self) -> float:
        return float(self._intercept)

    def predict(self, traffic: Sequence[float]) -> np.ndarray:
        # Fed the UNCLEANED raw data from the SiteCapacityPlanner
        traffic_arr = np.asarray(traffic, dtype=float)
        pred = self.intercept + self.slope * traffic_arr
        return np.clip(pred, 0.0, None)

    def traffic_at_threshold(self, threshold: float) -> Optional[float]:
        if self.slope <= 0:
            return None
        return (threshold - self.intercept) / self.slope


@dataclass
class ExpansionDecision:
    state: str
    method: str


class ExpansionStateMachine:
    UNKNOWN = "Manual Review - Unrecognised Configuration"
    MISSING = "Manual Review - Configuration Not Found"

    RULES: List[Tuple[str, str, str]] = [
        ("FDD M-MIMO", r"fdd\s*m-?mimo|fdd\s*massive", "New Site"),
        ("TDD M-MIMO", r"tdd\s*m-?mimo|tdd\s*massive", "FDD M-MIMO"),
        ("TDD 2nd Carrier", r"tdd\s*2nd\s*carrier|2nd\s*carrier\s*tdd|tdd\s*2", "TDD M-MIMO"),
        ("TDD 1st Carrier", r"tdd\s*1st\s*carrier|1st\s*carrier\s*tdd|tdd\s*1|^tdd$", "TDD 2nd Carrier"),
        ("Normal Sites 4 Sectors", r"normal\s*sites\s*4\s*sectors?|4\s*sectors?", "TDD 1st Carrier"),
        ("Normal Sites 3 Sectors", r"normal\s*sites\s*3\s*sectors?|3\s*sectors?", "Normal Sites 4 Sectors"),
    ]

    def __init__(self) -> None:
        self._compiled = [(s, re.compile(p), m) for s, p, m in self.RULES]
        self.unrecognised: Dict[str, int] = {}

    def decide(self, config: object) -> ExpansionDecision:
        if config is None or (isinstance(config, float) and np.isnan(config)) or str(config).strip() == "":
            return ExpansionDecision("Missing", self.MISSING)
        text = str(config).lower()
        for state, pattern, method in self._compiled:
            if pattern.search(text):
                return ExpansionDecision(state, method)
        self.unrecognised[str(config)] = self.unrecognised.get(str(config), 0) + 1
        return ExpansionDecision("Unknown", self.UNKNOWN)


class SiteCapacityPlanner:
    def __init__(self, ps: pd.DataFrame, sites: pd.DataFrame, macro_actual: pd.Series,
                 macro_forecast: pd.Series, regressor: PRBRegressor) -> None:
        self.ps = ps
        self.sites = sites
        self.macro_actual = macro_actual
        self.macro_forecast = macro_forecast
        self.regressor = regressor
        self.target_period = pd.Period(Config.FORECAST_END, freq="M")
        self.baseline_periods: List[pd.Period] = []
        self.multiplier = float("nan")

    def growth_multiplier(self) -> pd.DataFrame:
        ms = sorted(self.ps[Col.MONTH_START].unique())
        window = [pd.Timestamp(m).to_period("M") for m in ms[-Config.BASELINE_WINDOW_MONTHS:]]
        actual_m = self.macro_actual.groupby(self.macro_actual.index.to_period("M")).mean()
        fc_m = self.macro_forecast.groupby(self.macro_forecast.index.to_period("M")).mean()
        if self.target_period not in fc_m.index:
            raise RuntimeError("Forecast does not cover target month {}.".format(self.target_period))
        usable = [p for p in window if p in actual_m.index]
        if not usable:
            raise RuntimeError("PS KPI months ({}) do not overlap the Network_Traffic history.".format(
                ", ".join(str(p) for p in window)))
        target_val = float(fc_m[self.target_period])
        base_mean = float(actual_m.loc[usable].mean())
        if base_mean <= 0:
            raise RuntimeError("Baseline macro traffic is zero - cannot compute a growth multiplier.")
        self.baseline_periods = usable
        self.multiplier = target_val / base_mean
        return pd.DataFrame({
            "Baseline Month": [str(p) for p in usable],
            "Macro Avg Daily (TB)": [float(actual_m[p]) for p in usable],
            "Target Avg Daily (TB)": target_val,
            "Multiplier to {}".format(self.target_period): [target_val / float(actual_m[p]) for p in usable],
        })

    def site_forecast(self) -> pd.DataFrame:
        if not self.baseline_periods:
            self.growth_multiplier()
        base_ts = [p.to_timestamp() for p in self.baseline_periods]
        window = self.ps[self.ps[Col.MONTH_START].isin(base_ts)]
        
        # Use the raw traffic data preserved by the DataCleaner
        traffic_col = "Raw_Data_Traffic" if "Raw_Data_Traffic" in window.columns else Col.DATA_TRAFFIC
        
        baseline_traffic = window[window[traffic_col] > 0].groupby(Col.SITE)[traffic_col].mean()
        baseline_prb = window[window[Col.PRB] > 0].groupby(Col.SITE)[Col.PRB].mean()
        
        latest_region = (self.ps.sort_values(Col.MONTH_START)
                         .groupby(Col.SITE)[Col.REGION].last().rename(Col.REGION))
        
        df = pd.concat([
            latest_region, 
            baseline_traffic.rename("Baseline_Traffic"),
            baseline_prb.rename("Baseline_PRB")
        ], axis=1).reset_index()
        
        df["Baseline_Traffic"] = df["Baseline_Traffic"].fillna(0.0)
        df["Baseline_PRB"] = df["Baseline_PRB"].fillna(0.0)
        df["Growth_Multiplier"] = self.multiplier
        
        # Forecasted Traffic calculation
        df["Forecast_Traffic_2026"] = df["Baseline_Traffic"] * df["Growth_Multiplier"]
        
        # Calculate expected PRB increase using the OLS Regression model
        df["Forecast_PRB_2026"] = self.regressor.predict(df["Forecast_Traffic_2026"])
        
        return df

    def expansion_table(self, site_df: pd.DataFrame) -> Tuple[pd.DataFrame, ExpansionStateMachine]:
        # Filter degraded sites based on threshold
        degraded = site_df[
            (site_df["Forecast_PRB_2026"] > Config.PRB_THRESHOLD_PCT)
        ].copy()
        
        merged = degraded.merge(self.sites[[Col.SITE, Col.CONFIG]], on=Col.SITE, how="left")
        sm = ExpansionStateMachine()
        merged["Expansion Method"] = [sm.decide(c).method for c in merged[Col.CONFIG]]
        
        table = pd.DataFrame({
            "Site ID": merged[Col.SITE],
            "Region": merged[Col.REGION],
            "Current Config": merged[Col.CONFIG].fillna("N/A"),
            "Baseline Traffic (TB)": merged["Baseline_Traffic"].round(4),
            "Forecasted Traffic (TB)": merged["Forecast_Traffic_2026"].round(4),
            "Baseline PRB (%)": merged["Baseline_PRB"].round(4),
            "Forecasted PRB (%)": merged["Forecast_PRB_2026"].round(4),
            "Expansion Method": merged["Expansion Method"],
        }).sort_values(["Forecasted PRB (%)", "Site ID"], ascending=[False, True]).reset_index(drop=True)
        return table, sm


# ============================================================================
# Advanced Forecasting Framework (Macro Traffic)
# ============================================================================
class BaseForecaster(ABC):
    name = "Base"
    note = ""  

    @abstractmethod
    def tune(self, train_series: pd.Series, season_length: int) -> None:
        pass

    @abstractmethod
    def fit(self, y: pd.Series) -> "BaseForecaster":
        pass

    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        pass



class HoltWintersForecaster(BaseForecaster):
    name = "Holt-Winters"
    note = "additive, tuned smoothing"
    
    def __init__(self):
        self.best_params = (0.8, 0.05, 0.9)
        self.train_series = None
        
    def _fit_additive_hw(self, values, season_length, alpha, beta, gamma):
        values = np.asarray(values, dtype=float)
        if len(values) < 2 * season_length:
            return 0, 0, np.zeros(season_length), np.full(len(values), np.nan)

        first_season_mean = float(np.mean(values[:season_length]))
        level = first_season_mean
        trend = float((np.mean(values[season_length:2 * season_length]) - first_season_mean) / season_length)
        seasonal = list(values[:season_length] - first_season_mean)
        fitted = np.full(len(values), np.nan, dtype=float)

        for t in range(season_length, len(values)):
            previous_level = level
            previous_season = seasonal[t - season_length]
            fitted[t] = level + trend + previous_season

            level = alpha * (values[t] - previous_season) + (1 - alpha) * (level + trend)
            trend = beta * (level - previous_level) + (1 - beta) * trend
            seasonal.append(gamma * (values[t] - level) + (1 - gamma) * previous_season)

        return level, trend, np.asarray(seasonal), fitted

    def tune(self, train_series: pd.Series, season_length: int):
        values = np.asarray(train_series.values, dtype=float)
        coarse_grid = np.round(np.arange(0.1, 1.0, 0.2), 2)
        best_sse = float("inf")
        best_parameters = None

        def score(parameters):
            _, _, _, fitted = self._fit_additive_hw(values, season_length, *parameters)
            valid = ~np.isnan(fitted)
            if np.sum(valid) == 0:
                return float("inf")
            return float(np.sum((values[valid] - fitted[valid]) ** 2))

        for candidate in product(coarse_grid, repeat=3):
            parameters = tuple(float(value) for value in candidate)
            candidate_sse = score(parameters)
            if candidate_sse < best_sse:
                best_sse, best_parameters = candidate_sse, parameters

        if best_parameters is None:
            best_parameters = (0.8, 0.05, 0.9)
            
        refined_axes = []
        for center in best_parameters:
            axis = np.arange(max(0.05, center - 0.19), min(0.95, center + 0.19) + 0.025, 0.05)
            refined_axes.append(np.unique(np.round(np.clip(axis, 0.05, 0.95), 2)))

        for candidate in product(*refined_axes):
            parameters = tuple(float(value) for value in candidate)
            candidate_sse = score(parameters)
            if candidate_sse < best_sse:
                best_sse, best_parameters = candidate_sse, parameters

        if best_parameters:
            self.best_params = best_parameters

    def fit(self, y: pd.Series) -> "HoltWintersForecaster":
        self.train_series = y
        return self

    def predict(self, steps: int) -> np.ndarray:
        values = np.asarray(self.train_series.values, dtype=float)
        season_length = 12
        alpha, beta, gamma = self.best_params
        level, trend, seasonal, _ = self._fit_additive_hw(values, season_length, alpha, beta, gamma)
        n = len(values)
        forecasts = []
        for horizon in range(1, steps + 1):
            seasonal_index = n - season_length + ((horizon - 1) % season_length)
            forecasts.append(level + horizon * trend + seasonal[seasonal_index])
        return np.asarray(forecasts, dtype=float)


class XGBoostForecaster(BaseForecaster):
    name = "XGBoost"
    note = "cyclic months + lags, tuned"
    
    def __init__(self):
        self.best_params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'objective': 'reg:squarederror'}
        self.train_series = None

    def _forecast_recursive(self, train_series, steps, params):
        history = train_series.copy()
        forecasts = []
        df_train = pd.DataFrame(history)
        
        months = df_train.index.month
        df_train['Month_Sin'] = np.sin(2 * np.pi * months / 12)
        df_train['Month_Cos'] = np.cos(2 * np.pi * months / 12)
        
        for lag in [1, 2, 3, 6, 12]:
            df_train[f'Lag_{lag}'] = df_train.iloc[:, 0].shift(lag)
            
        df_train = df_train.dropna()
        if df_train.empty:
            return np.zeros(steps)
            
        X_train = df_train.drop(columns=[df_train.columns[0]])
        y_train = df_train.iloc[:, 0]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        current_history = history.copy()
        for i in range(steps):
            next_date = current_history.index[-1] + pd.offsets.MonthEnd(1)
            next_month = next_date.month
            
            next_row = {
                'Month_Sin': np.sin(2 * np.pi * next_month / 12),
                'Month_Cos': np.cos(2 * np.pi * next_month / 12)
            }
            for lag in [1, 2, 3, 6, 12]:
                next_row[f'Lag_{lag}'] = current_history.iloc[-lag]
                
            X_test = pd.DataFrame([next_row])
            pred = model.predict(X_test)[0]
            forecasts.append(pred)
            current_history.loc[next_date] = pred
            
        return np.asarray(forecasts, dtype=float)

    def tune(self, train_series: pd.Series, season_length: int):
        train_len = len(train_series) - season_length
        train_split = train_series.iloc[:train_len]
        val_split = train_series.iloc[train_len:]
        best_sse = float('inf')
        best_params = {}
        for md, lr, n_est in product([2, 3, 4], [0.05, 0.1], [50, 100]):
            params = {'max_depth': md, 'learning_rate': lr, 'n_estimators': n_est, 'objective': 'reg:squarederror'}
            preds = self._forecast_recursive(train_split, season_length, params)
            sse = np.sum((val_split.values - preds)**2)
            if sse < best_sse:
                best_sse = sse
                best_params = params
        if best_params:
            self.best_params = best_params

    def fit(self, y: pd.Series) -> "XGBoostForecaster":
        self.train_series = y
        return self

    def predict(self, steps: int) -> np.ndarray:
        return self._forecast_recursive(self.train_series, steps, self.best_params)


class ProphetForecaster(BaseForecaster):
    name = "Prophet"
    note = "yearly seasonality, tuned priors"
    
    def __init__(self):
        self.best_params = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0}
        self.model = None
        self.train_series = None
        self._silence_logging()

    @staticmethod
    def _silence_logging():
        import logging
        for noisy in ("cmdstanpy", "prophet", "fbprophet", "stan"):
            lg = logging.getLogger(noisy)
            lg.setLevel(logging.CRITICAL)
            lg.propagate = False
            lg.handlers = [logging.NullHandler()]
            
    def _fit_prophet(self, train_series, params):
        df_train = train_series.reset_index()
        df_train.columns = ['ds', 'y']
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, **params)
        m.fit(df_train)
        return m

    def _forecast_prophet(self, m, steps):
        try:
            future = m.make_future_dataframe(periods=steps, freq='ME')
        except ValueError:
            future = m.make_future_dataframe(periods=steps, freq='M')
        forecast = m.predict(future)
        return forecast['yhat'].tail(steps).values

    def tune(self, train_series: pd.Series, season_length: int):
        train_len = len(train_series) - season_length
        if train_len <= 0:
            return
        train_split = train_series.iloc[:train_len]
        val_split = train_series.iloc[train_len:]
        best_sse = float('inf')
        best_params = {}
        for cps, sps in product([0.01, 0.05, 0.1], [0.1, 1.0, 10.0]):
            params = {'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps}
            m = self._fit_prophet(train_split, params)
            preds = self._forecast_prophet(m, season_length)
            sse = np.sum((val_split.values - preds)**2)
            if sse < best_sse:
                best_sse = sse
                best_params = params
        if best_params:
            self.best_params = best_params

    def fit(self, y: pd.Series) -> "ProphetForecaster":
        self.train_series = y
        self.model = self._fit_prophet(y, self.best_params)
        return self

    def predict(self, steps: int) -> np.ndarray:
        return self._forecast_prophet(self.model, steps)



class MacroTrafficAnalysis:
    def __init__(self, series: pd.Series) -> None:
        self.series = series
        self.factories: Dict[str, Callable[[], BaseForecaster]] = {
            XGBoostForecaster.name: XGBoostForecaster,
            ProphetForecaster.name: ProphetForecaster,
            HoltWintersForecaster.name: HoltWintersForecaster,
        }
        self.fold_years = [2023, 2024, 2025]
        self.fold_metrics = []
        self.average_metrics = pd.DataFrame()
        self.best_model_name = None
        self.forecast_series = None
        self.all_forecasts = {}
        self.model_notes = {
            XGBoostForecaster.name: "cyclic months + lags, tuned",
            ProphetForecaster.name: "yearly seasonality, tuned priors",
            HoltWintersForecaster.name: "additive, tuned smoothing"
        }

    def evaluate(self) -> pd.DataFrame:
        ConsoleUI.info("Evaluating models using time-based validation (Folds: 2023, 2024, 2025)...")
        self.fold_metrics = []
        
        model_status = {name: "OK" for name in self.factories.keys()}
        
        for year in self.fold_years:
            train = self.series[self.series.index.year < year]
            test = self.series[self.series.index.year == year]
            if len(test) != 12:
                continue
            
            for name, factory in self.factories.items():
                if model_status[name] != "OK":
                    continue
                try:
                    model = factory()
                    model.tune(train, season_length=12)
                    model.fit(train)
                    preds = model.predict(len(test))
                    
                    mae = float(np.mean(np.abs(test.values - preds)))
                    rmse = float(np.sqrt(np.mean((test.values - preds) ** 2)))
                    mape = float(np.mean(np.abs((test.values - preds) / test.values)) * 100)
                    
                    self.fold_metrics.append({
                        'Fold Year': year,
                        'Model': name,
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE (%)': mape
                    })
                except Exception as exc:
                    print(f"Failed {name} on {year}: {exc}")
                    
        metrics_df = pd.DataFrame(self.fold_metrics)
        if metrics_df.empty:
            raise RuntimeError("No models could be evaluated.")
            
        avg_metrics = metrics_df.groupby('Model')[['MAE', 'RMSE', 'MAPE (%)']].mean().reset_index()
        valid_models = avg_metrics
            
        best_row = valid_models.loc[valid_models['RMSE'].idxmin()]
        self.best_model_name = best_row['Model']
        
        final_rows = []
        for name in self.factories.keys():
            if name in avg_metrics['Model'].values:
                row = avg_metrics[avg_metrics['Model'] == name].iloc[0].to_dict()
                row['Status'] = "OK  <-- SELECTED" if name == self.best_model_name else "OK"
            else:
                row = {'Model': name, 'MAE': float('nan'), 'RMSE': float('nan'), 'MAPE (%)': float('nan'), 'Status': model_status[name]}
            row['Notes'] = self.model_notes[name]
            final_rows.append(row)
            
        self.average_metrics = pd.DataFrame(final_rows)
        return self.average_metrics

    def comparison_table(self) -> pd.DataFrame:
        df = self.average_metrics.copy()
        df = df.rename(columns={'RMSE': 'RMSE (TB/day)'})
        return df[['Model', 'RMSE (TB/day)', 'MAPE (%)', 'Notes', 'Status']]

    def forecast(self) -> pd.Series:
        if not self.best_model_name:
            raise RuntimeError("Call evaluate() before forecast().")
            
        try:
            forecast_dates = pd.date_range(
                start=self.series.index[-1] + pd.offsets.MonthEnd(1),
                periods=12,
                freq='ME'
            )
        except ValueError:
            forecast_dates = pd.date_range(
                start=self.series.index[-1] + pd.offsets.MonthEnd(1),
                periods=12,
                freq='M'
            )

        self.all_forecasts = {}
        
        name = self.best_model_name
        model = self.factories[name]()
        model.tune(self.series, season_length=12)
        model.fit(self.series)
        preds = model.predict(12)
        
        self.all_forecasts[name] = pd.Series(preds, index=forecast_dates)
        self.forecast_series = self.all_forecasts[name]
        
        return self.forecast_series

    def monthly_table(self) -> pd.DataFrame:
        combined = pd.concat([self.series, self.forecast_series])
        grp = combined.groupby(combined.index.to_period("M"))
        out = pd.DataFrame({"Avg Daily Traffic (TB)": grp.mean(), "Total Traffic (TB)": grp.sum(),
                            "Days": grp.size()})
        last_actual = self.series.index[-1].to_period("M")
        out["Type"] = np.where(out.index <= last_actual, "Actual", "Forecast")
        return out

    def plot(self, plotter: Plotter) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(13, 9))
        
        ax = axes[0]
        year = 2025
        train = self.series[self.series.index.year < year]
        test = self.series[self.series.index.year == year]
        if len(test) > 0:
            recent_start = test.index[0] - pd.DateOffset(months=6)
            ax.plot(self.series.loc[recent_start:test.index[-1]].index, 
                    self.series.loc[recent_start:test.index[-1]], color="black", lw=1.4, label="Actual")
            
            for name, factory in self.factories.items():
                try:
                    model = factory()
                    model.tune(train, season_length=12)
                    model.fit(train)
                    preds = model.predict(len(test))
                    pred_series = pd.Series(preds, index=test.index)
                    lw = 2.4 if name == self.best_model_name else 1.1
                    
                    metric_row = self.average_metrics[self.average_metrics['Model'] == name]
                    if not metric_row.empty and not pd.isna(metric_row['RMSE'].values[0]):
                        rmse_val = metric_row['RMSE'].values[0]
                        mape_val = metric_row['MAPE (%)'].values[0]
                        ax.plot(pred_series.index, pred_series, lw=lw, label=f"{name} (Avg RMSE {rmse_val:.2f}, MAPE {mape_val:.1f}%)")
                    else:
                        ax.plot(pred_series.index, pred_series, lw=lw, label=f"{name}")
                except Exception as e:
                    pass
                    
            ax.axvline(test.index[0], color="grey", ls="--")
            ax.set_title(f"2025 Holdout Backtest")
            ax.set_ylabel("TB / day")
            ax.legend(loc="upper left", fontsize=9)

        ax = axes[1]
        ax.plot(self.series.index, self.series, color="black", lw=0.9, label="Actual")
        
        name = self.best_model_name
        f_series = self.forecast_series
        ax.plot(f_series.index, f_series, color="crimson", lw=2.5,
                label=f"Forecast - {name} (Selected Best)")
                        
        ax.set_title(f"Macro network traffic forecast to {self.forecast_series.index[-1].date()}")
        ax.set_ylabel("TB / day")
        ax.legend(loc="upper left")
        
        fig.tight_layout()
        plotter.finish(fig, "macro_traffic_forecast.png")


# ============================================================================
# Main Application
# ============================================================================
class Application:
    def __init__(self, cs: pd.DataFrame, ps: pd.DataFrame, sites: pd.DataFrame,
                 traffic_series: pd.Series, out_dir: Path, show_plots: bool,
                 preview_rows: int, debug: bool = False,
                 checkpoint: Optional[Callable[[], None]] = None) -> None:
        self.cs, self.ps, self.sites, self.traffic = cs, ps, sites, traffic_series
        self.out_dir = out_dir
        self.preview_rows = preview_rows
        self.debug = debug
        self.checkpoint = checkpoint or (lambda: None)
        self.plotter = Plotter(out_dir / Config.PLOTS_DIRNAME, show=show_plots)
        self.analyzer = KPIAnalyzer(self.plotter)
        self._macro: Optional[MacroTrafficAnalysis] = None  

    def _check(self):
        self.checkpoint()

    def run(self) -> None:
        while True:
            ConsoleUI.banner("RAN KPI ANALYSIS & CAPACITY PLANNING")
            choice = ConsoleUI.ask_choice("Main menu - select an option:", {
                "A": "KPI Analysis",
                "B": "Capacity Planning & Traffic Forecasting",
                "Q": "Quit",
            })
            if choice == "Q":
                print("\nGoodbye.")
                return
            try:
                if choice == "A":
                    self.kpi_analysis()
                elif choice == "B":
                    self._cli_capacity_planning()
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception as exc:  
                sys.stderr.write("\nERROR: {}: {}\n".format(type(exc).__name__, exc))
                if self.debug:
                    traceback.print_exc()

    def _prompt_filters(self) -> FilterSelection:
        fam = ConsoleUI.ask_choice("KPI family:", {
            "1": "CS (Voice & Availability) KPIs",
            "2": "PS (Data) KPIs",
            "3": "Both",
        })
        family = {"1": "CS", "2": "PS", "3": "BOTH"}[fam]
        frames = {"CS": [self.cs], "PS": [self.ps], "BOTH": [self.cs, self.ps]}[family]
        months = sorted(set().union(*[set(f[Col.MONTH].unique()) for f in frames]))
        regions = sorted(set().union(*[set(f[Col.REGION].unique()) for f in frames]))
        return FilterSelection(family,
                               ConsoleUI.ask_multi("month(s)", months),
                               ConsoleUI.ask_multi("region(s)", regions))

    def run_kpi_analysis(self, sel: FilterSelection, kpi_dataset: str, kpi_name: Optional[str], 
                         run_correlation: bool, run_regression: bool) -> KPIAnalysisResult:
        self._check()
        active_df = apply_filters(self.cs if kpi_dataset == "CS" else self.ps, sel)
        
        # Clear out old KPI charts so they don't carry over
        plot_dir = self.out_dir / Config.PLOTS_DIRNAME
        if plot_dir.exists():
            for p in plot_dir.glob("*.png"):
                if p.name.startswith(("combined_analysis_", "scatter_")):
                    try: p.unlink()
                    except OSError: pass
        
        images = []
        if kpi_name:
            self.analyzer.plot_kpi_analysis_charts(active_df, kpi_name, CS_NUMERIC if kpi_dataset == "CS" else PS_NUMERIC)
            if plot_dir.exists():
                images = [p for p in plot_dir.glob("*.png") if p.name.startswith(("combined_analysis_", "scatter_"))]
                
        return KPIAnalysisResult(
            images=images,
            correlation=None,
            regression=None,
            filters=sel.describe()
        )

    def kpi_analysis(self) -> None:
        ConsoleUI.banner("CHOICE A - KPI ANALYSIS")
        sel = self._prompt_filters()
        ConsoleUI.info("Filters: " + sel.describe())
        
        active_df = pd.DataFrame()
        if sel.family == "CS":
            active_df = apply_filters(self.cs, sel)
            numeric_cols = CS_NUMERIC
        elif sel.family == "PS":
            active_df = apply_filters(self.ps, sel)
            numeric_cols = PS_NUMERIC
        else:
            active_df = apply_filters(self.ps, sel)
            numeric_cols = PS_NUMERIC
        
        if not active_df.empty:
            kpi_options = {str(i+1): col for i, col in enumerate(numeric_cols)}
            ConsoleUI.info("\nSelect a specific KPI for Analysis:")
            for k, v in kpi_options.items():
                print(f"   [{k}] {v}")
                
            kpi_choice = input("  > ").strip()
            if kpi_choice in kpi_options:
                selected_kpi = kpi_options[kpi_choice]
                self.analyzer.plot_kpi_analysis_charts(active_df, selected_kpi, numeric_cols)
            else:
                ConsoleUI.warn("Invalid selection. Skipping Analysis.")

        ConsoleUI.banner("EXPORTING BAD SITES (THROUGHPUT <= 5 Mbps)")
        active_ps = apply_filters(self.ps, sel)
        self.analyzer.bad_sites_analysis(active_ps, self.sites, self.out_dir)

    def plot_configuration_shift(self, csv_path: Path) -> None:
        if not csv_path.exists():
            return
            
        try:
            df = pd.read_csv(csv_path)
            if "Current Config" not in df.columns or "Expansion Method" not in df.columns:
                return
                
            before_counts = df["Current Config"].value_counts().rename("Before Expansion")
            after_counts = df["Expansion Method"].value_counts().rename("After Expansion")
            
            combined = pd.concat([before_counts, after_counts], axis=1).fillna(0)
            
            custom_order = [
                "FDD M-MIMO", 
                "Normal Sites 3 Sectors", 
                "Normal Sites 4 Sectors", 
                "TDD 1st Carrier", 
                "TDD 2nd Carrier", 
                "TDD M-MIMO", 
                "New Site"
            ]
            
            final_order = custom_order + [c for c in combined.index if c not in custom_order]
            combined = combined.reindex(final_order).fillna(0)
            combined = combined[(combined.T != 0).any()]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            combined.plot(kind="bar", ax=ax, color=["#3498db", "#e74c3c"], edgecolor="black")
            
            ax.set_title("Site Capacity Forecast: Configuration Shift (Before vs After 2026)", fontsize=14)
            ax.set_ylabel("Number of Sites", fontsize=11)
            ax.set_xlabel("Hardware Configuration", fontsize=11)
            plt.xticks(rotation=45, ha="right")
            ax.legend(title="Status")
            
            fig.tight_layout()
            self.plotter.finish(fig, "configuration_before_vs_after.png")
            
        except Exception as exc:
            ConsoleUI.warn(f"Failed to generate Configuration Shift Chart: {exc}")

    def plot_regional_expansion(self, csv_path: Path) -> None:
        if not csv_path.exists():
            return
            
        try:
            df = pd.read_csv(csv_path)
            if "Region" not in df.columns or "Expansion Method" not in df.columns:
                return
                
            cross_tab = pd.crosstab(df["Region"], df["Expansion Method"])
            
            custom_order = [
                "FDD M-MIMO", "Normal Sites 3 Sectors", "Normal Sites 4 Sectors", 
                "TDD 1st Carrier", "TDD 2nd Carrier", "TDD M-MIMO", "New Site"
            ]
            
            existing_cols = [c for c in custom_order if c in cross_tab.columns]
            existing_cols += [c for c in cross_tab.columns if c not in custom_order]
            cross_tab = cross_tab[existing_cols]

            fig, ax = plt.subplots(figsize=(14, 7))
            cross_tab.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="black")
            
            ax.set_title("Required Site Expansions by Region (Forecasted 2026)", fontsize=14)
            ax.set_ylabel("Number of Sites to Expand", fontsize=11)
            ax.set_xlabel("Region", fontsize=11)
            plt.xticks(rotation=45, ha="right")
            ax.legend(title="Expansion Method", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            fig.tight_layout()
            self.plotter.finish(fig, "regional_expansions_stacked.png")
            
        except Exception as exc:
            ConsoleUI.warn(f"Failed to generate Regional Expansion Chart: {exc}")

    def capacity_planning(self, on_stage: Optional[Callable[[str], None]] = None) -> CapacityPlanningResult:
        def stage(msg):
            if on_stage: on_stage(msg)
            else: ConsoleUI.info(msg)

        stage("Evaluating macro traffic models...")
        self._check()
        
        # Clear out old capacity charts so they don't carry over
        plot_dir = self.out_dir / Config.PLOTS_DIRNAME
        cap_names = {"macro_traffic_forecast.png", "configuration_before_vs_after.png", "regional_expansions_stacked.png"}
        if plot_dir.exists():
            for p in plot_dir.glob("*.png"):
                if p.name in cap_names:
                    try: p.unlink()
                    except OSError: pass

        if self._macro is None or self._macro.forecast_series is None:
            macro = MacroTrafficAnalysis(self.traffic)
            macro.evaluate()
            macro.forecast()
            self._macro = macro
        else:
            macro = self._macro

        stage("Forecasting site traffic and calculating PRB...")
        self._check()
        regressor = PRBRegressor().fit(self.ps, verbose=False)
        planner = SiteCapacityPlanner(self.ps, self.sites, self.traffic, macro.forecast_series, regressor)
        mult_tbl = planner.growth_multiplier()
        site_df = planner.site_forecast()
        
        stage("Generating expansion plan...")
        self._check()
        table, sm = planner.expansion_table(site_df)
        
        csv_path = self.out_dir / Config.OUTPUT_CSV
        try:
            table.to_csv(csv_path, index=False)
            self.plot_configuration_shift(csv_path)
            self.plot_regional_expansion(csv_path)
        except OSError:
            csv_path = None

        macro.plot(self.plotter)
        
        images = []
        if plot_dir.exists():
            images = [p for p in plot_dir.glob("*.png") if p.name in cap_names]
        
        reg_summary = pd.DataFrame([
            {"Statistic": "Intercept", "Value": round(regressor.intercept, 4)},
            {"Statistic": "Slope", "Value": round(regressor.slope, 4)},
            {"Statistic": "R^2", "Value": round(regressor.r2, 4)},
            {"Statistic": "RMSE", "Value": round(regressor.rmse, 4)}
        ])

        summary = table.groupby("Expansion Method").size().rename("Sites").reset_index() if not table.empty else pd.DataFrame()
        by_region = pd.crosstab(table["Expansion Method"], table["Region"]).reset_index() if not table.empty else pd.DataFrame()

        return CapacityPlanningResult(
            best_model=macro.best_model_name,
            multiplier=planner.multiplier,
            expansion_table=table,
            n_sites=len(site_df),
            regression_summary=reg_summary,
            baseline_periods=[str(p) for p in planner.baseline_periods],
            n_inactive_sites=int((site_df["Baseline_Traffic"] == 0).sum()),
            csv_path=csv_path,
            unrecognised_configs=sm.unrecognised,
            model_comparison=macro.average_metrics,
            fold_metrics=pd.DataFrame(macro.fold_metrics),
            forecast_table=macro.monthly_table().reset_index(),
            multiplier_table=mult_tbl,
            site_stats=site_df[["Baseline_Traffic", "Forecast_Traffic_2026", "Baseline_PRB", "Forecast_PRB_2026"]].describe().T.reset_index(),
            expansion_summary=summary,
            expansion_by_region=by_region,
            images=images
        )

    def _cli_capacity_planning(self) -> None:
        ConsoleUI.banner("CHOICE B - CAPACITY PLANNING & TRAFFIC FORECASTING")

        ConsoleUI.section("STEP 1 - MACRO TRAFFIC MODEL COMPARISON ({} -> {})".format(
            self.traffic.index[0].year, self.traffic.index[-1].year))
        if self._macro is None or self._macro.forecast_series is None:
            macro = MacroTrafficAnalysis(self.traffic)
            macro.evaluate()
            macro.forecast()
            self._macro = macro
        else:
            macro = self._macro
            ConsoleUI.info("(re-using the model comparison from this session)")
            
        ConsoleUI.table(macro.comparison_table(), title="Average Hold-out accuracy (Folds: 2023, 2024, 2025)",
                        floatfmt=".3f", numparse=False)
        ConsoleUI.info("Selected model (lowest RMSE): {}".format(macro.best_model_name))

        monthly = macro.monthly_table()
        fc_months = monthly[monthly["Type"] == "Forecast"].copy()
        fc_months.index = fc_months.index.astype(str)
        ConsoleUI.table(fc_months.reset_index().rename(columns={"index": "Month", "Date": "Month"}),
                        title="Forecast through {} ({})".format(Config.FORECAST_END, macro.best_model_name),
                        floatfmt=".2f")
        macro.plot(self.plotter)

        ConsoleUI.section("STEP 2 - SITE-LEVEL TRAFFIC & PRB FORECAST (EoY 2026)")
        regressor = PRBRegressor().fit(self.ps, verbose=True)
        
        planner = SiteCapacityPlanner(self.ps, self.sites, self.traffic, macro.forecast_series, regressor)
        mult_tbl = planner.growth_multiplier()
        ConsoleUI.table(mult_tbl, title="Macro growth multipliers (baseline month -> {})".format(
            planner.target_period), floatfmt=".4f")
        ConsoleUI.info("Applied multiplier (mean of baseline months {}): x{:.4f}".format(
            ", ".join(str(p) for p in planner.baseline_periods), planner.multiplier))
        
        site_df = planner.site_forecast()
        n_zero = int((site_df["Baseline_Traffic"] == 0).sum())
        ConsoleUI.info("Sites forecast: {:,} | with no positive-traffic baseline (inactive): {:,}".format(
            len(site_df), n_zero))
        
        stats = site_df[["Baseline_Traffic", "Forecast_Traffic_2026", "Baseline_PRB", "Forecast_PRB_2026"]].describe().T
        ConsoleUI.table(stats.reset_index().rename(columns={"index": "Metric"}), floatfmt=".3f",
                        title="Site forecast distribution")

        ConsoleUI.section("STEP 3 - SITE EXPANSION TABLE (Forecast PRB > {:g}%)".format(
            Config.PRB_THRESHOLD_PCT))
        table, sm = planner.expansion_table(site_df)
        ConsoleUI.info("Degraded sites: {:,} of {:,} ({:.1f}%)".format(
            len(table), len(site_df), 100.0 * len(table) / max(len(site_df), 1)))
        if table.empty:
            ConsoleUI.info("No site is forecast above the threshold - no expansion required.")
        else:
            ConsoleUI.table(table, title="Site Expansion Table (worst PRB first)",
                            floatfmt=".3f", max_rows=self.preview_rows)
            summary = (table.groupby("Expansion Method").size().rename("Sites").reset_index()
                       .sort_values("Sites", ascending=False))
            ConsoleUI.table(summary, title="Expansion summary", floatfmt=".0f")
            by_region = pd.crosstab(table["Expansion Method"], table["Region"]).reset_index()
            ConsoleUI.table(by_region, title="Expansion method by region (site counts)", floatfmt=".0f")
        if sm.unrecognised:
            ConsoleUI.warn("Unrecognised configuration text (edit ExpansionStateMachine.RULES): " +
                           "; ".join("'{}' x{}".format(k, v) for k, v in sm.unrecognised.items()))
        out_path = self.out_dir / Config.OUTPUT_CSV
        try:
            table.to_csv(out_path, index=False)
            ConsoleUI.info("[saved] {}  ({:,} rows)".format(out_path, len(table)))
            
            self.plot_configuration_shift(out_path)
            self.plot_regional_expansion(out_path)
            
        except OSError as exc:
            ConsoleUI.warn("Could not write {}: {}".format(out_path, exc))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAN KPI Analysis & Capacity Planning pipeline")
    p.add_argument("--excel-file", default=Config.EXCEL_FILE,
                   help="path to the Excel workbook (default: Project Data - Traffic & KPIs.xlsx)")
    p.add_argument("--no-show", action="store_true", help="save plots but do not open windows")
    p.add_argument("--preview-rows", type=int, default=Config.PREVIEW_ROWS,
                   help="rows of the expansion table printed to the console")
    p.add_argument("--debug", action="store_true", help="print full tracebacks on errors")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    warnings.filterwarnings("ignore", category=FutureWarning)

    ConsoleUI.banner("RAN KPI ANALYSIS & CAPACITY PLANNING PIPELINE (EXCEL)")
    
    excel_path = Path(args.excel_file).expanduser().resolve()
    if not excel_path.is_file():
        cwd_path = Path.cwd() / args.excel_file
        if cwd_path.is_file():
            excel_path = cwd_path
        else:
            sys.stderr.write(f"\nERROR: Could not find Excel file: {args.excel_file}\n"
                             f"Please place it in the same folder as this script or use --excel-file.\n")
            return 2

    out_dir = Path(os.path.abspath(__file__)).parent
    
    try:
        ConsoleUI.section("LOADING DATA")
        raw = DataLoader(excel_path).load_all()
        cs, ps = preprocess_kpi_data(raw["cs"], raw["ps"])
        sites = clean_sites(raw["sites"])
        
        traffic = build_monthly_series(raw["traffic"])
        ConsoleUI.info("Macro traffic: {} -> {} ({:,} months)".format(
            traffic.index[0].date(), traffic.index[-1].date(), len(traffic)))
    except (FileNotFoundError, DataValidationError) as exc:
        sys.stderr.write("\nERROR: {}\n".format(exc))
        return 2

    app = Application(cs, ps, sites, traffic, out_dir, show_plots=not args.no_show,
                      preview_rows=args.preview_rows, debug=args.debug)
    try:
        app.run()
    except (EOFError, KeyboardInterrupt):
        print("\nInput closed - exiting.")
    except Exception as exc:  
        sys.stderr.write("\nERROR: {}: {}\n".format(type(exc).__name__, exc))
        if args.debug:
            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())