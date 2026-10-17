# 📡 RAN KPI Analysis & Capacity Planning Pipeline

An end-to-end **Radio Access Network (RAN)** analytics, traffic forecasting, and automated hardware capacity planning suite built in Python. Available via both a **multi-threaded Tkinter Desktop GUI** and an **interactive CLI**.

---

## 🏗️ System Architecture

The application is engineered using a decoupled **three-tier architecture**:

| File | Layer | Responsibility |
| :--- | :--- | :--- |
| `Final_Project_G2.py` | **Processing Layer** | Data ingestion, Winsorization & Modified Z-Score cleaning, KPI outlier diagnostics, ML time-series forecasting, IQR-cleaned OLS regression, and the Hardware Expansion State Machine. |
| `ran_kpi_backend.py` | **Controller Layer** | Headless execution controller managing lazy module imports, pre-flight workbook validation, background worker threads (`TaskRunner`), stdout queue redirection (`QueueWriter`), and cooperative task cancellation. |
| `ran_kpi_gui.py` | **Presentation Layer** | Modern `Tkinter` / `ttk` desktop interface featuring sortable data tables, dynamic chart rendering (`Pillow`), filterable KPI views, and real-time activity logs. |

---

## ✨ Key Engineering Features

### 1. Automated Telecom Data Cleaning (`DataCleaner`)
* **Schema Normalization:** Automatically resolves column naming variations across `CS KPIs`, `PS KPIs`, `Sites Configurations`, and `Network Traffic` sheets.
* **Statistical Sanitization:** Caps rate KPIs (`Availability`, `CSSR`, `Data CSSR`, `HO SR`, `PRB DL Utilization`) at 100%, applies **Winsorization** to extreme traffic spikes (P99.5), and filters drop-rate anomalies (`CDR`, `ERAB DR`) using **Modified Z-Score** ($\text{ModZ} > 3.5$) at P99.
* **Hierarchical Imputation:** Imputes missing records using a `Site Median → Region Median → Global Median` fallback chain while flagging zero-traffic/zero-availability outage rows.

### 2. Multi-Model Macro Traffic Forecasting (`MacroTrafficAnalysis`)
Evaluates three distinct forecasting architectures across rolling hold-out validation folds (**2023, 2024, and 2025**) and automatically selects the lowest-RMSE model to project traffic through **December 2026**:
* **XGBoost (`XGBoostForecaster`):** Recursive autoregression utilizing cyclic month encodings ($\sin/\cos$) and temporal lags ($1, 2, 3, 6, 12$ months) with hyperparameter grid search.
* **Facebook Prophet (`ProphetForecaster`):** Yearly seasonality decomposition with tuned `changepoint_prior_scale` and `seasonality_prior_scale`.
* **Holt-Winters (`HoltWintersForecaster`):** Custom additive triple exponential smoothing with two-stage coarse-to-fine grid optimization for $(\alpha, \beta, \gamma)$.

### 3. Site-Level PRB Regression & Expansion State Machine
* **IQR-Cleaned OLS Regression (`PRBRegressor`):** Fits an initial Ordinary Least Squares model mapping `Data Traffic (TB)` to `PRB DL Utilization (%)`, strips residual outliers outside $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$, and refits on clean data.
* **Automated Upgrade Path (`ExpansionStateMachine`):** Flags sites forecasted to exceed the **70% PRB threshold** (configurable) and prescribes hardware upgrades using deterministic state transitions:
  * `Normal Sites 3 Sectors` ➔ `Normal Sites 4 Sectors`
  * `Normal Sites 4 Sectors` ➔ `TDD 1st Carrier`
  * `TDD 1st Carrier` ➔ `TDD 2nd Carrier`
  * `TDD 2nd Carrier` ➔ `TDD M-MIMO`
  * `TDD M-MIMO` ➔ `FDD M-MIMO`
  * `FDD M-MIMO` ➔ `New Site`

### 4. KPI Diagnostics & Bad Sites Identification (`KPIAnalyzer`)
* Generates combined **Distribution Boxplots & Confidence-Interval Trendlines** alongside cross-KPI scatter regressions.
* **Automated Insights Engine:** Evaluates skewness, 5th/95th percentiles, and outputs targeted RF engineering recommendations (e.g., PRACH tuning, missing neighbor drive tests, load balancing, or FM alarm cross-checks).
* Isolates severely congested sites with average throughput below threshold (default $\le 5\text{ Mbps}$) and exports `Bad_Sites_Throughput.csv`.

---

## 🚀 Getting Started

### 1. Clone & Install Dependencies
```bash
git clone [https://github.com/YOUR_USERNAME/ran-kpi-capacity-planning.git](https://github.com/YOUR_USERNAME/ran-kpi-capacity-planning.git)
cd ran-kpi-capacity-planning

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt