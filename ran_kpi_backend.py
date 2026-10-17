#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application / controller layer for the RAN KPI desktop app.

This module sits between the Tkinter presentation layer (ran_kpi_gui.py) and the
processing layer (Final_Project_G2.py). It deliberately imports NO Tkinter so it can be
unit-tested headless and so that the GUI can never be blocked by it.

Responsibilities
  * lazy import of the heavy processing module (Prophet / XGBoost take seconds to load);
  * input validation that produces plain-language errors (UserInputError);
  * running work on a background thread and talking back through a queue;
  * cooperative cancellation (checked between model fits and between stages);
  * translating exceptions into "what happened / what to do" messages.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import logging
import queue
import sys
import tempfile
import threading
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# The GUI shows the PNG files the pipeline saves; it never hosts live Matplotlib figures.
# The Agg backend must be selected BEFORE the processing module imports pyplot, and it is the
# only backend that is safe to use from a worker thread.
import matplotlib

matplotlib.use("Agg")

PIPELINE_MODULE = "Final_Project_G2"
LOG_DIR = Path.home() / ".ran_kpi_gui"
ALLOWED_SUFFIXES = (".xlsx", ".xlsm", ".xls")

# Config attribute for every setting the GUI exposes (single place to extend).
CLEANING_SETTINGS = {
    "rate_cap": "RATE_CAP",
    "traffic_winsor_pct": "TRAFFIC_WINSOR_PCT",
    "drop_rate_winsor_pct": "DROP_RATE_WINSOR_PCT",
    "modz_threshold": "MODZ_THRESHOLD",
}


class UserInputError(Exception):
    """A problem the user can fix (bad or missing input). Shown without a traceback."""


# --------------------------------------------------------------------------------------
# Logging (detailed diagnostics go to a file; the GUI log pane shows the same text)
# --------------------------------------------------------------------------------------
def get_file_logger() -> logging.Logger:
    logger = logging.getLogger("ran_kpi_gui")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            handler: logging.Handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
        except OSError:  # unwritable home directory: keep running without a file log
            handler = logging.NullHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


# --------------------------------------------------------------------------------------
# Error translation
# --------------------------------------------------------------------------------------
@dataclass
class ErrorInfo:
    title: str
    message: str
    hint: str = ""


def describe_error(exc: BaseException) -> ErrorInfo:
    """Turn an exception into a short explanation plus a practical next step."""
    name = type(exc).__name__
    text = str(exc)
    if isinstance(exc, UserInputError):
        return ErrorInfo("Check your input", text)
    if isinstance(exc, ModuleNotFoundError):
        pkg = exc.name or "a required package"
        return ErrorInfo("Missing Python package", f"The package '{pkg}' is not installed.",
                         "Install the requirements:  pip install -r requirements.txt")
    if isinstance(exc, PermissionError):
        return ErrorInfo("Permission denied", text or "The file or folder cannot be accessed.",
                         "Close the file if it is open in Excel, or choose a different output folder.")
    if isinstance(exc, FileNotFoundError):
        return ErrorInfo("File not found", text, "Check the path on the Data tab.")
    if name == "DataValidationError":
        return ErrorInfo("The workbook is not in the expected format", text,
                         "The workbook needs the sheets and columns listed on the Help tab.")
    if "Forecast does not cover target month" in text:
        return ErrorInfo("Forecast does not reach the target month", text,
                         "The traffic history should end in Dec 2025 so that the 12-month forecast "
                         "ends in Dec 2026 (see Data checks on the Data tab).")
    if "do not overlap the Network_Traffic history" in text:
        return ErrorInfo("KPI months and traffic history do not overlap", text,
                         "Check that the PS KPI months fall inside the Network Traffic dates.")
    if "No models could be evaluated" in text:
        return ErrorInfo("Forecasting models could not be evaluated", text,
                         "Each validation year (2023-2025) needs 12 months of traffic plus earlier history.")
    if isinstance(exc, MemoryError):
        return ErrorInfo("Out of memory", "The analysis ran out of memory.", "Close other programs and retry.")
    return ErrorInfo("Unexpected error", f"{name}: {text}", "The full traceback is in the log pane.")


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------
@dataclass
class LoadedData:
    excel_path: Path
    out_dir: Path
    cs: pd.DataFrame
    ps: pd.DataFrame
    sites: pd.DataFrame
    traffic: pd.Series
    cleaning_report: pd.DataFrame
    raw_rows: Dict[str, int]
    months: Dict[str, List[str]]     # {"CS": [...], "PS": [...]}
    regions: Dict[str, List[str]]
    checks: List[Tuple[str, str]] = field(default_factory=list)  # (level, message)


@dataclass
class KPIParams:
    family: str                       # "CS" | "PS" | "BOTH"
    months: Optional[List[str]]
    regions: Optional[List[str]]
    kpi: Optional[Tuple[str, str]]    # (dataset, column) for the outlier chart
    run_correlation: bool
    run_regression: bool


@dataclass
class CapacityParams:
    prb_threshold: float
    baseline_months: int
    refit_models: bool = False


@dataclass
class KPIOutcome:
    result: Any                       # pipeline.KPIAnalysisResult
    regression_table: Optional[pd.DataFrame]


@dataclass
class BadSitesOutcome:
    table: Optional[pd.DataFrame]
    csv_path: Optional[Path]


# --------------------------------------------------------------------------------------
# Back-end (everything that touches the processing layer)
# --------------------------------------------------------------------------------------
class Backend:
    def __init__(self) -> None:
        self.pipeline = None
        self.defaults: Dict[str, float] = {}
        self.cancel_event = threading.Event()
        self.data: Optional[LoadedData] = None
        self._app = None
        self._macro = None

    # ---- start-up -------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def import_pipeline(self) -> None:
        """Import the processing module (slow: pulls in Prophet, XGBoost, statsmodels)."""
        if self.pipeline is not None:
            return
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.pipeline = importlib.import_module(PIPELINE_MODULE)
        cfg = self.pipeline.Config
        self.defaults = {k: getattr(cfg, attr) for k, attr in CLEANING_SETTINGS.items()}
        self.defaults["prb_threshold"] = cfg.PRB_THRESHOLD_PCT
        self.defaults["baseline_months"] = cfg.BASELINE_WINDOW_MONTHS

    # ---- cancellation ----------------------------------------------------------------
    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise self.pipeline.OperationCancelled("Cancelled by the user.")

    # ---- helpers -----------------------------------------------------------------------
    @staticmethod
    def _check_writable(folder: Path) -> None:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=folder):
                pass
        except OSError as exc:
            raise PermissionError(f"Cannot write to the output folder '{folder}': {exc}") from exc

    def _require_data(self) -> LoadedData:
        if self.data is None:
            raise UserInputError("Load and clean a workbook first (Data tab).")
        return self.data

    def _get_app(self):
        """One Application per (data, output folder); the expensive model comparison is kept
        across runs exactly like the original session cache (see Application._macro)."""
        data = self._require_data()
        if self._app is None or self._app.out_dir != data.out_dir:
            app = self.pipeline.Application(
                data.cs, data.ps, data.sites, data.traffic, data.out_dir, show_plots=False,
                preview_rows=self.pipeline.Config.PREVIEW_ROWS, checkpoint=self.check_cancel)
            app._macro = self._macro
            self._app = app
        return self._app

    # ---- 1. load & clean -----------------------------------------------------------------
    def load(self, excel_path: str, out_dir: str, settings: Dict[str, float]) -> LoadedData:
        P = self.pipeline
        if not excel_path.strip():
            raise UserInputError("Choose an Excel workbook first.")
        path = Path(excel_path).expanduser()
        if not path.is_file():
            raise UserInputError(f"The workbook was not found:\n{path}")
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise UserInputError(f"Unsupported file type '{path.suffix}'. Use an Excel workbook "
                                 f"({', '.join(ALLOWED_SUFFIXES)}).")
        out = Path(out_dir).expanduser() if out_dir.strip() else path.parent
        self._check_writable(out)

        for key, attr in CLEANING_SETTINGS.items():  # Config is class-level, set before cleaning
            setattr(P.Config, attr, settings[key])

        raw = P.DataLoader(path.resolve()).load_all()
        raw_rows = {k: len(v) for k, v in raw.items()}
        for key, label in (("cs", "CS KPIs"), ("ps", "PS KPIs"),
                           ("sites", "Sites Configurations"), ("traffic", "Network Traffic")):
            if raw[key].empty:
                raise P.DataValidationError(f"Sheet '{label}' has headers but no data rows.")

        cs, ps, report = P.preprocess_kpi_data(raw["cs"], raw["ps"], return_report=True)
        sites = P.clean_sites(raw["sites"])
        traffic = P.build_monthly_series(raw["traffic"])
        P.ConsoleUI.info("Macro traffic: {} -> {} ({:,} months)".format(
            traffic.index[0].date(), traffic.index[-1].date(), len(traffic)))

        data = LoadedData(
            excel_path=path.resolve(), out_dir=out.resolve(), cs=cs, ps=ps, sites=sites,
            traffic=traffic, cleaning_report=report, raw_rows=raw_rows,
            months={"CS": sorted(cs[P.Col.MONTH].unique()), "PS": sorted(ps[P.Col.MONTH].unique())},
            regions={"CS": sorted(cs[P.Col.REGION].unique()), "PS": sorted(ps[P.Col.REGION].unique())})
        data.checks = self._preflight(data)
        self.data, self._app, self._macro = data, None, None  # new data invalidates cached models
        return data

    def _preflight(self, data: LoadedData) -> List[Tuple[str, str]]:
        """Surface the script's hidden assumptions before a long run fails or silently skips work."""
        P = self.pipeline
        checks: List[Tuple[str, str]] = []
        traffic = data.traffic
        for year in P.MacroTrafficAnalysis(traffic).fold_years:
            n = int((traffic.index.year == year).sum())
            earlier = int((traffic.index.year < year).sum())
            if n != 12 or earlier == 0:
                checks.append(("warn", f"Validation year {year} has {n} of 12 months"
                                       f"{'' if earlier else ' and no earlier history'}; it will be skipped."))
        last = traffic.index[-1].to_period("M")
        target = pd.Period(P.Config.FORECAST_END, freq="M")
        if not (last < target <= last + 12):
            checks.append(("error", f"The 12-month forecast covers {last + 1} to {last + 12}, which does not "
                                    f"include the target month {target}. Capacity planning will stop with an error."))
        ps_months = [pd.Timestamp(m).to_period("M") for m in sorted(data.ps[P.Col.MONTH_START].unique())]
        window = ps_months[-P.Config.BASELINE_WINDOW_MONTHS:]
        actual = set(traffic.index.to_period("M"))
        if not any(p in actual for p in window):
            checks.append(("error", "The latest PS KPI months do not overlap the Network Traffic history."))
        if not checks:
            checks.append(("ok", "Traffic history, validation years and forecast target look consistent."))
        return checks

    # ---- 2. KPI analysis (workflow A) -----------------------------------------------------
    def kpi_analysis(self, params: KPIParams) -> KPIOutcome:
        P = self.pipeline
        data = self._require_data()
        if not (params.kpi or params.run_correlation or params.run_regression):
            raise UserInputError("Select at least one analysis to run.")
        sel = P.FilterSelection(params.family, params.months, params.regions)
        need_cs = params.run_correlation or (params.kpi and params.kpi[0] == "CS")
        need_ps = params.run_regression or (params.kpi and params.kpi[0] == "PS")
        if need_cs and P.apply_filters(data.cs, sel).empty:
            raise UserInputError("No CS rows match the selected months and regions.")
        if need_ps and P.apply_filters(data.ps, sel).empty:
            raise UserInputError("No PS rows match the selected months and regions.")
        app = self._get_app()
        result = app.run_kpi_analysis(
            sel, kpi_dataset=params.kpi[0] if params.kpi else "PS",
            kpi_name=params.kpi[1] if params.kpi else None,
            run_correlation=params.run_correlation, run_regression=params.run_regression)
        table = result.regression.summary() if result.regression is not None else None
        return KPIOutcome(result, table)

    # ---- 3. capacity planning (workflow B) ---------------------------------------------------
    def capacity_planning(self, params: CapacityParams, on_stage: Callable[[str], None]):
        P = self.pipeline
        self._require_data()
        P.Config.PRB_THRESHOLD_PCT = params.prb_threshold
        P.Config.BASELINE_WINDOW_MONTHS = params.baseline_months
        app = self._get_app()
        if params.refit_models:
            app._macro = None

        def stage(message: str) -> None:
            self.check_cancel()          # stage boundaries are cancellation points too
            on_stage(message)

        try:
            return app.capacity_planning(on_stage=stage)
        finally:
            self._macro = app._macro     # keep the model comparison even if a later stage failed

    # ---- 4. bad sites -------------------------------------------------------------------------
    def bad_sites(self, prb_threshold: float) -> BadSitesOutcome:
        P = self.pipeline
        data = self._require_data()
        P.Config.PRB_THRESHOLD_PCT = prb_threshold
        app = self._get_app()
        table = app.analyzer.bad_sites_analysis(data.ps, data.sites, data.out_dir)
        csv = data.out_dir / "Bad_Sites_Configuration.csv"
        return BadSitesOutcome(table, csv if csv.exists() and table is not None and not table.empty else None)


# --------------------------------------------------------------------------------------
# Background execution
# --------------------------------------------------------------------------------------
class QueueWriter(io.TextIOBase):
    """File-like object that forwards everything printed by the processing layer to the GUI.

    Redirecting stdout keeps the original console-oriented code (ConsoleUI, statsmodels
    summaries) completely unchanged while still surfacing its output in the log pane.
    """

    def __init__(self, messages: "queue.Queue", buffer: List[str]) -> None:
        self._messages = messages
        self._buffer = buffer

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if text:
            self._buffer.append(text)
            self._messages.put(("log", text))
        return len(text)

    def flush(self) -> None:
        pass


class TaskRunner:
    """Runs one job at a time on a daemon thread; results come back through `messages`.

    Message kinds: ("log", text) ("stage", text) ("done", result, report_text)
                   ("cancelled",) ("error", exception, traceback_text)
    The GUI drains the queue from its own thread, so no widget is ever touched here.
    """

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.messages: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, job: Callable[[Callable[[str], None]], Any]) -> bool:
        """`job` receives a `stage(text)` callback. Returns False if a job is already running."""
        if self.busy:
            return False
        self.backend.cancel_event.clear()
        self._thread = threading.Thread(target=self._run, args=(job,), daemon=True, name="ran-kpi-worker")
        self._thread.start()
        return True

    def cancel(self) -> None:
        self.backend.cancel_event.set()

    def _run(self, job: Callable[[Callable[[str], None]], Any]) -> None:
        buffer: List[str] = []
        writer = QueueWriter(self.messages, buffer)

        def stage(text: str) -> None:
            self.messages.put(("stage", text))

        cancelled_type = None
        try:
            with contextlib.redirect_stdout(writer):
                result = job(stage)
            self.messages.put(("done", result, "".join(buffer)))
        except BaseException as exc:  # noqa: BLE001 - reported to the GUI, never swallowed
            if self.backend.pipeline is not None:
                cancelled_type = self.backend.pipeline.OperationCancelled
            if cancelled_type is not None and isinstance(exc, cancelled_type):
                self.messages.put(("cancelled",))
            else:
                self.messages.put(("error", exc, traceback.format_exc()))
