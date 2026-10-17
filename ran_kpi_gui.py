#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAN KPI analysis and capacity planning - desktop interface (presentation layer).

Layers
  Final_Project_G2.py   processing layer  (the original analysis code, unchanged algorithms)
  ran_kpi_backend.py    controller layer  (validation, worker thread, cancellation, errors)
  ran_kpi_gui.py        this file         (Tkinter / ttk widgets only)

Threading rule: widgets are touched ONLY from the Tk main thread. The worker thread posts
messages to a queue and `MainWindow._poll` (scheduled with `after`) drains it.

Run:  python ran_kpi_gui.py
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd
from PIL import Image, ImageTk

import ran_kpi_backend as B

APP_TITLE = "RAN KPI analysis and capacity planning"

PALETTE = {
    "bg": "#F1F4F6", "surface": "#FFFFFF", "ink": "#1B2A33", "muted": "#5B6B76",
    "line": "#D3DAE0", "stripe": "#F6F8FA", "head": "#E7ECF0",
    "accent": "#0E7C86", "accent_dark": "#0A5E66", "accent_soft": "#D9EEF0",
    "header": "#12313B", "header_text": "#FFFFFF", "header_muted": "#9FB7C0",
    "ok": "#2E7D32", "warn": "#9A5B00", "danger": "#B3261E",
    "log_bg": "#13212A", "log_fg": "#D6E2EA", "log_warn": "#F2B25C", "log_err": "#FF8A80",
    "log_stage": "#6FD0D9",
}

def open_path(path: Path) -> None:
    try:
        if platform.system() == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError as exc:
        messagebox.showerror("Could not open", f"{path}\n\n{exc}\n\nOpen it manually from your file manager.")

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=PALETTE["bg"])
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.configure(yscrollcommand=self._on_scroll_set)
        self.inner.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._window, width=e.width))
        for widget in (self.canvas, self.inner):
            widget.bind("<Enter>", self._bind_wheel)
            widget.bind("<Leave>", self._unbind_wheel)

    def _on_scroll_set(self, lo: str, hi: str) -> None:
        self.vsb.set(lo, hi)
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.vsb.grid_remove()
        else:
            self.vsb.grid(row=0, column=1, sticky="ns")

    def _bind_wheel(self, _e: object) -> None:
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", self._wheel)
        self.canvas.bind_all("<Button-5>", self._wheel)

    def _unbind_wheel(self, _e: object) -> None:
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(seq)

    def _wheel(self, event: tk.Event) -> None:
        if self.vsb.winfo_ismapped():
            step = -1 if (event.num == 4 or getattr(event, "delta", 0) > 0) else 1
            self.canvas.yview_scroll(step * 2, "units")

class DataTable(ttk.Frame):
    def __init__(self, parent: tk.Misc, empty_text: str = "Nothing to show yet.", height: int = 8, decimals: int = 3) -> None:
        super().__init__(parent)
        self.decimals = decimals
        self.empty_text = empty_text
        self._df: Optional[pd.DataFrame] = None
        self._sort_col: Optional[str] = None
        self._sort_desc = False
        self._font = tkfont.nametofont("TkDefaultFont")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(self, show="headings", height=height, selectmode="browse")
        ysb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        xsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        self.tree.tag_configure("odd", background=PALETTE["stripe"])
        self._empty = ttk.Label(self.tree, text=empty_text, style="Empty.TLabel")
        self.set_dataframe(None)

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        return self._df

    def _fmt(self, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, int) or (hasattr(value, "dtype") and "int" in str(getattr(value, "dtype", ""))):
            return f"{int(value):,}"
        if isinstance(value, float):
            return f"{value:,.{self.decimals}f}"
        return str(value)

    def set_dataframe(self, df: Optional[pd.DataFrame], empty_text: Optional[str] = None) -> None:
        self._df = df.reset_index(drop=True) if df is not None else None
        self._sort_col, self._sort_desc = None, False
        self.tree.delete(*self.tree.get_children())
        if self._df is None or len(self._df.columns) == 0:
            self.tree["columns"] = ()
            self._empty.configure(text=empty_text or self.empty_text)
            self._empty.place(relx=0.5, rely=0.45, anchor="center")
            return
        cols = [str(c) for c in self._df.columns]
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c, command=lambda col=c: self._sort(col))
            sample = [self._fmt(v) for v in self._df[c].head(200)]
            width = max([int(self._font.measure(c) * 1.2) + 16] + [self._font.measure(s) for s in sample] + [40]) + 28
            self.tree.column(c, width=min(max(width, 70), 380), anchor="w", stretch=False)
        self._fill()
        if self._df.empty:
            self._empty.configure(text=empty_text or "No rows.")
            self._empty.place(relx=0.5, rely=0.45, anchor="center")
        else:
            self._empty.place_forget()

    def _fill(self) -> None:
        self.tree.delete(*self.tree.get_children())
        assert self._df is not None
        for i, row in enumerate(self._df.itertuples(index=False)):
            self.tree.insert("", "end", values=[self._fmt(v) for v in row], tags=("odd",) if i % 2 else ())

    def _sort(self, col: str) -> None:
        if self._df is None or self._df.empty:
            return
        self._sort_desc = (not self._sort_desc) if self._sort_col == col else False
        self._sort_col = col
        series = self._df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        key = numeric if numeric.notna().sum() >= 0.8 * max(series.notna().sum(), 1) else series.astype(str)
        order = key.sort_values(ascending=not self._sort_desc, na_position="last").index
        self._df = self._df.loc[order].reset_index(drop=True)
        self._fill()
        for c in self.tree["columns"]:
            arrow = (" \u25bc" if self._sort_desc else " \u25b2") if c == col else ""
            self.tree.heading(c, text=c + arrow)

class ImageViewer(ttk.Frame):
    def __init__(self, parent: tk.Misc, empty_text: str, labeler: Callable[[Path], str]) -> None:
        super().__init__(parent)
        self.labeler = labeler
        self.empty_text = empty_text
        self._paths: List[Path] = []
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._resize_job: Optional[str] = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        bar.columnconfigure(0, weight=1)
        self.choice = tk.StringVar()
        self.combo = ttk.Combobox(bar, textvariable=self.choice, state="readonly")
        self.combo.grid(row=0, column=0, sticky="ew")
        self.combo.bind("<<ComboboxSelected>>", lambda _e: self._render())
        self.open_btn = ttk.Button(bar, text="Open in image viewer", command=self._open_external)
        self.open_btn.grid(row=0, column=1, padx=(8, 0))
        self.save_btn = ttk.Button(bar, text="Save a copy...", command=self._save_copy)
        self.save_btn.grid(row=0, column=2, padx=(8, 0))
        self.canvas = tk.Canvas(self, background=PALETTE["surface"], highlightthickness=1,
                                highlightbackground=PALETTE["line"])
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._schedule_render)
        self.set_images([])

    def set_images(self, paths: Sequence[Path]) -> None:
        self._paths = [Path(p) for p in paths]
        names = [self.labeler(p) for p in self._paths]
        self.combo["values"] = names
        state = "normal" if self._paths else "disabled"
        self.open_btn.configure(state=state)
        self.save_btn.configure(state=state)
        if self._paths:
            self.combo.current(0)
        else:
            self.choice.set("")
        self._render()

    def _current(self) -> Optional[Path]:
        idx = self.combo.current()
        return self._paths[idx] if 0 <= idx < len(self._paths) else None

    def _schedule_render(self, _event: object = None) -> None:
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(120, self._render)

    def _render(self) -> None:
        self._resize_job = None
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        path = self._current()
        if path is None:
            self.canvas.create_text(w / 2, h * 0.45, text=self.empty_text, fill=PALETTE["muted"],
                                    font=tkfont.nametofont("TkDefaultFont"), width=max(w - 60, 100), justify="center")
            return
        if w < 20 or h < 20:
            return
        try:
            with Image.open(path) as img:
                scale = min((w - 12) / img.width, (h - 12) / img.height)
                resized = img.convert("RGB").resize(
                    (max(int(img.width * scale), 1), max(int(img.height * scale), 1)), Image.LANCZOS)
        except (OSError, ValueError) as exc:
            self.canvas.create_text(w / 2, h / 2, text=f"Could not display {path.name}:\n{exc}",
                                    fill=PALETTE["danger"], width=max(w - 60, 100))
            return
        self._photo = ImageTk.PhotoImage(resized) 
        self.canvas.create_image(w / 2, h / 2, image=self._photo, anchor="center")

    def _open_external(self) -> None:
        path = self._current()
        if path:
            open_path(path)

    def _save_copy(self) -> None:
        path = self._current()
        if not path:
            return
        target = filedialog.asksaveasfilename(defaultextension=".png", initialfile=path.name,
                                              filetypes=[("PNG image", "*.png")])
        if target:
            try:
                Path(target).write_bytes(path.read_bytes())
            except OSError as exc:
                messagebox.showerror("Could not save the chart", f"{exc}\n\nChoose a folder you can write to.")

class ReportText(ttk.Frame):
    def __init__(self, parent: tk.Misc, empty_text: str) -> None:
        super().__init__(parent)
        self.empty_text = empty_text
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        mono = tkfont.nametofont("TkFixedFont")
        self.text = tk.Text(self, wrap="none", font=mono, background=PALETTE["surface"],
                            foreground=PALETTE["ink"], relief="flat", highlightthickness=1,
                            highlightbackground=PALETTE["line"], padx=10, pady=8)
        ysb = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        xsb = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        self.set_text("")

    def set_text(self, content: str) -> None:
        self.text.configure(state="normal", foreground=PALETTE["ink"] if content else PALETTE["muted"])
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content.strip("\n") if content else self.empty_text)
        self.text.configure(state="disabled")

# ======================================================================================
# Main window
# ======================================================================================
class MainWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.backend = B.Backend()
        self.runner = B.TaskRunner(self.backend)
        self.file_log = B.get_file_logger()
        self._on_done: Optional[Callable[[object, str], None]] = None
        self._task_label = ""
        self._task_started = 0.0
        self._action_buttons: List[ttk.Button] = []
        self._out_user_set = False
        self._last_capacity = None

        root.title(APP_TITLE)
        root.geometry("1320x860")
        root.minsize(1060, 720)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._apply_theme()
        self._declare_variables()
        self._build_layout()
        self._refresh_enabled()
        self.root.after(150, self._place_sash)
        self.root.after(100, self._poll)
        self._run_task("Loading analysis libraries (first start takes a few seconds)...",
                       lambda stage: self.backend.import_pipeline(), self._after_import)

    def _apply_theme(self) -> None:
        families = set(tkfont.families(self.root))
        ui = next((f for f in ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Noto Sans", "DejaVu Sans")
                   if f in families), "TkDefaultFont")
        mono = next((f for f in ("Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono", "Courier New")
                     if f in families), "TkFixedFont")
        self.ui_font, self.mono_font = ui, mono
        for name, size in (("TkDefaultFont", 10), ("TkTextFont", 10), ("TkMenuFont", 10), ("TkHeadingFont", 10)):
            tkfont.nametofont(name).configure(family=ui, size=size)
        tkfont.nametofont("TkFixedFont").configure(family=mono, size=10)

        P = PALETTE
        s = ttk.Style(self.root)
        s.theme_use("clam")
        self.root.configure(background=P["bg"])
        s.configure(".", background=P["bg"], foreground=P["ink"], bordercolor=P["line"],
                    lightcolor=P["bg"], darkcolor=P["bg"], focuscolor=P["accent"])
        s.configure("TFrame", background=P["bg"])
        s.configure("Card.TFrame", background=P["surface"])
        s.configure("Header.TFrame", background=P["header"])
        s.configure("TLabel", background=P["bg"], foreground=P["ink"])
        s.configure("Card.TLabel", background=P["surface"])
        s.configure("Muted.TLabel", foreground=P["muted"])
        s.configure("CardMuted.TLabel", background=P["surface"], foreground=P["muted"])
        s.configure("Empty.TLabel", background=P["surface"], foreground=P["muted"])
        s.configure("Title.TLabel", background=P["header"], foreground=P["header_text"],
                    font=(ui, 17, "bold"))
        s.configure("Subtitle.TLabel", background=P["header"], foreground=P["header_muted"])
        s.configure("Chip.TLabel", background=P["header"], foreground=P["header_text"], font=(ui, 10, "bold"))
        s.configure("H2.TLabel", font=(ui, 11, "bold"))
        s.configure("CardH2.TLabel", background=P["surface"], font=(ui, 11, "bold"))
        s.configure("Metric.TLabel", background=P["surface"], font=(ui, 20, "bold"), foreground=P["accent_dark"])
        s.configure("Ok.TLabel", foreground=P["ok"])
        s.configure("Warn.TLabel", foreground=P["warn"])
        s.configure("Err.TLabel", foreground=P["danger"])
        s.configure("TLabelframe", background=P["surface"], bordercolor=P["line"], relief="solid")
        s.configure("TLabelframe.Label", background=P["surface"], foreground=P["ink"], font=(ui, 10, "bold"))
        s.configure("TButton", background=P["surface"], foreground=P["ink"], padding=(12, 6),
                    bordercolor=P["line"], relief="solid")
        s.map("TButton", background=[("active", P["head"]), ("disabled", P["bg"])],
              foreground=[("disabled", "#94A1AA")])
        s.configure("Accent.TButton", background=P["accent"], foreground="#FFFFFF", bordercolor=P["accent"],
                    padding=(16, 8), font=(ui, 10, "bold"))
        s.map("Accent.TButton", background=[("active", P["accent_dark"]), ("disabled", "#A9C9CD")],
              foreground=[("disabled", "#F2F7F8")], bordercolor=[("disabled", "#A9C9CD")])
        s.configure("Danger.TButton", foreground=P["danger"])
        s.configure("TCheckbutton", background=P["surface"])
        s.configure("TRadiobutton", background=P["surface"])
        s.configure("TEntry", fieldbackground=P["surface"], bordercolor=P["line"], padding=4)
        s.configure("TCombobox", fieldbackground=P["surface"], background=P["surface"], padding=4)
        s.configure("TSpinbox", fieldbackground=P["surface"], padding=4)
        s.configure("TNotebook", background=P["bg"], bordercolor=P["line"], tabmargins=(0, 0, 0, 0))
        s.configure("TNotebook.Tab", background=P["head"], foreground=P["muted"], padding=(16, 8))
        s.map("TNotebook.Tab", background=[("selected", P["surface"])], foreground=[("selected", P["ink"])])
        s.configure("Sub.TNotebook", background=P["surface"])
        s.configure("Treeview", background=P["surface"], fieldbackground=P["surface"], rowheight=24,
                    bordercolor=P["line"])
        s.configure("Treeview.Heading", background=P["head"], foreground=P["ink"], padding=(8, 5),
                    font=(ui, 10, "bold"), relief="flat")
        s.map("Treeview", background=[("selected", P["accent_soft"])], foreground=[("selected", P["ink"])])
        s.configure("Horizontal.TProgressbar", troughcolor=P["head"], background=P["accent"],
                    bordercolor=P["line"], lightcolor=P["accent"], darkcolor=P["accent"])
        s.configure("TPanedwindow", background=P["bg"])

    def _declare_variables(self) -> None:
        self.excel_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.rate_cap_var = tk.StringVar(value="100")
        self.traffic_winsor_var = tk.StringVar(value="0.995")
        self.drop_winsor_var = tk.StringVar(value="0.99")
        self.modz_var = tk.StringVar(value="3.5")
        self.family_var = tk.StringVar(value="BOTH")
        self.all_months_var = tk.BooleanVar(value=True)
        self.all_regions_var = tk.BooleanVar(value=True)
        self.outlier_var = tk.BooleanVar(value=True)
        self.kpi_var = tk.StringVar()
        self.prb_var = tk.StringVar(value="70")
        self.baseline_var = tk.StringVar(value="6")
        self.thr_var = tk.StringVar(value="5")
        self.refit_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Starting...")
        self.elapsed_var = tk.StringVar(value="")
        self.chip_var = tk.StringVar(value="No workbook loaded")
        self.summary_vars = {k: tk.StringVar(value="-") for k in
                             ("Sites", "PS rows", "CS rows", "Months (PS)", "Regions", "Traffic history")}
        self._kpi_choices: Dict[str, tuple] = {}

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(20, 14))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Traffic forecasting, KPI diagnostics and site expansion planning from one Excel workbook",
                  style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(header, textvariable=self.chip_var, style="Chip.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        self.panes = ttk.PanedWindow(self.root, orient="vertical")
        self.panes.grid(row=1, column=0, sticky="nsew", padx=12, pady=(10, 0))
        self.notebook = ttk.Notebook(self.panes)
        self.panes.add(self.notebook, weight=4)
        self._build_data_tab()
        self._build_kpi_tab()
        self._build_capacity_tab()
        self._build_log_pane()

        status = ttk.Frame(self.root, padding=(14, 6))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status, textvariable=self.elapsed_var, style="Muted.TLabel").grid(row=0, column=1, padx=10)
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=170)
        self.progress.grid(row=0, column=2, padx=(0, 10))
        self.cancel_btn = ttk.Button(status, text="Cancel", style="Danger.TButton", command=self._on_cancel)
        self.cancel_btn.grid(row=0, column=3)

    def _place_sash(self) -> None:
        self.root.update_idletasks()
        height = self.panes.winfo_height()
        if height > 200:
            self.panes.sashpos(0, max(int(height * 0.55), height - 150))

    def _card(self, parent: tk.Misc, title: str) -> ttk.LabelFrame:
        return ttk.LabelFrame(parent, text=title, padding=12)

    def _action(self, button: ttk.Button) -> ttk.Button:
        self._action_buttons.append(button)
        return button

    def _build_data_tab(self) -> None:
        outer = ttk.Frame(self.notebook, padding=(14, 14, 6, 14))
        self.notebook.add(outer, text="Data")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        scroller = ScrollableFrame(outer)
        scroller.grid(row=0, column=0, sticky="nsew")
        tab = scroller.inner
        tab.columnconfigure(0, weight=1)

        src = self._card(tab, "Workbook")
        src.grid(row=0, column=0, sticky="ew")
        src.columnconfigure(1, weight=1)
        ttk.Label(src, text="Excel workbook", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(src, textvariable=self.excel_var).grid(row=0, column=1, sticky="ew", padx=8)
        self.browse_excel = self._action(ttk.Button(src, text="Browse...", command=self._browse_excel))
        self.browse_excel.grid(row=0, column=2)
        ttk.Label(src, text="Output folder", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=4)
        out_entry = ttk.Entry(src, textvariable=self.out_var)
        out_entry.grid(row=1, column=1, sticky="ew", padx=8)
        out_entry.bind("<Key>", lambda _e: setattr(self, "_out_user_set", True))
        self.browse_out = self._action(ttk.Button(src, text="Browse...", command=self._browse_out))
        self.browse_out.grid(row=1, column=2)
        ttk.Label(src, style="CardMuted.TLabel", wraplength=900, justify="left",
                  text="Needs the sheets CS KPIs, PS KPIs, Sites Configurations and Network Traffic. Charts and CSV "
                       "files are written to the output folder (default: the workbook's folder).").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        btns = ttk.Frame(src, style="Card.TFrame")
        btns.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.load_btn = self._action(ttk.Button(btns, text="Load and clean data", style="Accent.TButton",
                                                command=self._on_load))
        self.load_btn.pack(side="left")
        self.adv_btn = ttk.Button(btns, text="Show cleaning thresholds", command=self._toggle_advanced)
        self.adv_btn.pack(side="left", padx=10)

        self.adv_frame = self._card(tab, "Cleaning thresholds (applied when the data is loaded)")
        for i, (label, var, hint) in enumerate((
                ("Rate cap (%)", self.rate_cap_var, "Availability, CSSR, Data CSSR, PRB and HO SR are capped here."),
                ("Traffic winsor percentile", self.traffic_winsor_var, "0.5 to 1. Data traffic above it is clipped."),
                ("Drop-rate winsor percentile", self.drop_winsor_var, "0.5 to 1. Used for CDR and ERAB DR outliers."),
                ("Modified z-score threshold", self.modz_var, "Drop-rate values above this score are capped."))):
            ttk.Label(self.adv_frame, text=label, style="Card.TLabel").grid(row=i, column=0, sticky="w", pady=3)
            ttk.Entry(self.adv_frame, textvariable=var, width=10).grid(row=i, column=1, padx=10)
            ttk.Label(self.adv_frame, text=hint, style="CardMuted.TLabel").grid(row=i, column=2, sticky="w")
        ttk.Button(self.adv_frame, text="Restore script defaults", command=self._reset_cleaning).grid(
            row=4, column=0, sticky="w", pady=(8, 0))
        self._adv_visible = False

        mid = ttk.Frame(tab)
        mid.grid(row=2, column=0, sticky="ew", pady=10)
        mid.columnconfigure(1, weight=1)
        summary = self._card(mid, "Loaded data")
        summary.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        for i, key in enumerate(self.summary_vars):
            ttk.Label(summary, text=key, style="CardMuted.TLabel").grid(row=i, column=0, sticky="w", padx=(0, 16))
            ttk.Label(summary, textvariable=self.summary_vars[key], style="Card.TLabel").grid(row=i, column=1, sticky="w")
        self.checks_card = self._card(mid, "Data checks")
        self.checks_card.grid(row=0, column=1, sticky="nsew")
        self.checks_card.columnconfigure(0, weight=1)
        self._set_checks([("info", "Load a workbook to see consistency checks for the forecasting stages.")])

        cleaning = self._card(tab, "Cleaning report (what was corrected, and how many rows)")
        cleaning.grid(row=3, column=0, sticky="nsew")
        cleaning.columnconfigure(0, weight=1)
        cleaning.rowconfigure(0, weight=1)
        self.cleaning_table = DataTable(cleaning, "Load a workbook to see the cleaning report.", height=9, decimals=0)
        self.cleaning_table.grid(row=0, column=0, sticky="nsew")

    def _build_kpi_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(tab, text="KPI analysis")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        left = ttk.Frame(tab, width=340)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)

        left.rowconfigure(0, weight=1)
        scroller = ScrollableFrame(left)
        scroller.grid(row=0, column=0, sticky="nsew")
        body = scroller.inner
        body.columnconfigure(0, weight=1)

        fam = self._card(body, "KPI family")
        fam.grid(row=0, column=0, sticky="ew")
        for text, val in (("CS - voice and availability", "CS"), ("PS - data", "PS"), ("Both", "BOTH")):
            ttk.Radiobutton(fam, text=text, value=val, variable=self.family_var,
                            command=self._on_family_change).pack(anchor="w", pady=1)

        flt = self._card(body, "Filter")
        flt.grid(row=1, column=0, sticky="ew", pady=8)
        flt.columnconfigure((0, 1), weight=1)
        ttk.Label(flt, text="Months", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(flt, text="Regions", style="Card.TLabel").grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.month_list = self._make_listbox(flt, 0)
        self.region_list = self._make_listbox(flt, 1)
        ttk.Checkbutton(flt, text="All months", variable=self.all_months_var,
                        command=lambda: self._toggle_all(self.all_months_var, self.month_list)).grid(
            row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(flt, text="All regions", variable=self.all_regions_var,
                        command=lambda: self._toggle_all(self.all_regions_var, self.region_list)).grid(
            row=2, column=1, sticky="w", pady=(4, 0), padx=(10, 0))

        ana = self._card(body, "Analyses")
        ana.grid(row=2, column=0, sticky="ew")
        ana.columnconfigure(0, weight=1)
        self.outlier_chk = ttk.Checkbutton(ana, text="Outlier chart and insights for one KPI", variable=self.outlier_var,
                                           command=self._refresh_kpi_controls)
        self.outlier_chk.grid(row=0, column=0, sticky="w")
        self.kpi_combo = ttk.Combobox(ana, textvariable=self.kpi_var, state="readonly")
        self.kpi_combo.grid(row=1, column=0, sticky="ew", pady=(2, 8))

        self.kpi_run = self._action(ttk.Button(left, text="Run KPI analysis", style="Accent.TButton",
                                               command=self._on_run_kpi))
        self.kpi_run.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        bad = self._card(body, "Bad sites (Congestion)")
        bad.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        bad.columnconfigure(1, weight=1)
        ttk.Label(bad, text="Throughput (Mbps)", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Spinbox(bad, textvariable=self.thr_var, from_=1, to=100, increment=1, width=8).grid(row=0, column=1, sticky="e")
        self.bad_run = self._action(ttk.Button(bad, text="Find bad sites", style="Accent.TButton", command=self._on_run_bad))
        self.bad_run.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        self.bad_open = ttk.Button(bad, text="Open CSV folder", command=self._open_out_folder)
        self.bad_open.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.bad_open.state(["disabled"])

        right = ttk.Notebook(tab, style="Sub.TNotebook")
        right.grid(row=0, column=1, sticky="nsew")
        self.kpi_nb = right
        
        charts = ttk.Frame(right, padding=10, style="Card.TFrame")
        charts.columnconfigure(0, weight=1)
        charts.rowconfigure(0, weight=1)
        right.add(charts, text="Charts")
        self.kpi_images = ImageViewer(charts, "Run a KPI analysis to see charts here.", self._pretty_image)
        self.kpi_images.grid(row=0, column=0, sticky="nsew")
        
        rep = ttk.Frame(right, padding=10, style="Card.TFrame")
        rep.columnconfigure(0, weight=1)
        rep.rowconfigure(0, weight=1)
        right.add(rep, text="Report")
        self.kpi_report = ReportText(rep, "The insights and statistical summaries of the last run appear here.")
        self.kpi_report.grid(row=0, column=0, sticky="nsew")

        bs_tab = ttk.Frame(right, padding=10, style="Card.TFrame")
        bs_tab.columnconfigure(0, weight=1)
        bs_tab.rowconfigure(1, weight=1)
        right.add(bs_tab, text="Bad Sites")
        bar = ttk.Frame(bs_tab, style="Card.TFrame")
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(bar, text="Sites with low Average Throughput", style="CardH2.TLabel").pack(side="left")
        self.bad_msg = ttk.Label(bar, text="", style="Muted.TLabel")
        self.bad_msg.pack(side="left", padx=10)
        self.bad_table = DataTable(bs_tab, "Run 'Find bad sites' to list severely congested sites.", decimals=3)
        self.bad_table.grid(row=1, column=0, sticky="nsew")

        self._on_family_change()

    def _make_listbox(self, parent: tk.Misc, column: int) -> tk.Listbox:
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=1, column=column, sticky="nsew", padx=(10, 0) if column else (0, 0))
        lb = tk.Listbox(frame, selectmode="extended", height=5, width=12, exportselection=False,
                        relief="flat", highlightthickness=1, highlightbackground=PALETTE["line"],
                        selectbackground=PALETTE["accent_soft"], selectforeground=PALETTE["ink"],
                        activestyle="none", background=PALETTE["surface"], 
                        foreground=PALETTE["ink"], disabledforeground="#A0ACB4")
        sb = ttk.Scrollbar(frame, orient="vertical", command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        lb.configure(state="disabled")
        return lb

    def _build_capacity_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(tab, text="Capacity planning")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        left = ttk.Frame(tab, width=340)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        scroller = ScrollableFrame(left)
        scroller.grid(row=0, column=0, sticky="nsew")
        body = scroller.inner
        body.columnconfigure(0, weight=1)
        par = self._card(body, "Planning settings")
        par.grid(row=0, column=0, sticky="ew")
        par.columnconfigure(1, weight=1)
        ttk.Label(par, text="PRB threshold (%)", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Spinbox(par, textvariable=self.prb_var, from_=1, to=100, increment=1, width=8).grid(row=0, column=1, sticky="e")
        ttk.Label(par, text="Sites whose forecast PRB exceeds this are flagged for expansion.",
                  style="CardMuted.TLabel", wraplength=290, justify="left").grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(par, text="Baseline window (months)", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 4))
        ttk.Spinbox(par, textvariable=self.baseline_var, from_=1, to=12, increment=1, width=8).grid(row=2, column=1, sticky="e")
        ttk.Label(par, text="Latest PS months averaged as the starting point for each site.",
                  style="CardMuted.TLabel", wraplength=290, justify="left").grid(row=3, column=0, columnspan=2, sticky="w")
        self.target_label = ttk.Label(par, text="Forecast target: Dec 2026 (fixed in the script)", style="CardMuted.TLabel",
                                      wraplength=290, justify="left")
        self.target_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Checkbutton(par, text="Re-fit the forecasting models", variable=self.refit_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(par, text="Off: reuse the model comparison from an earlier run on this data.",
                  style="CardMuted.TLabel", wraplength=290, justify="left").grid(row=6, column=0, columnspan=2, sticky="w")
        self.cap_run = self._action(ttk.Button(left, text="Run capacity planning", style="Accent.TButton",
                                               command=self._on_run_capacity))
        self.cap_run.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(body, style="Muted.TLabel", wraplength=300, justify="left",
                  text="The first run tunes three forecasting models over three validation years and can take several "
                       "minutes. You can cancel between model fits.").grid(row=1, column=0, sticky="w", pady=(10, 0))

        right = ttk.Notebook(tab, style="Sub.TNotebook")
        right.grid(row=0, column=1, sticky="nsew")
        self.cap_nb = right

        summ = ttk.Frame(right, padding=14, style="Card.TFrame")
        summ.columnconfigure((0, 1, 2, 3), weight=1, uniform="m")
        right.add(summ, text="Summary")
        self.metric_vars: Dict[str, tk.StringVar] = {}
        for i, (key, sub) in enumerate((("Selected model", "lowest average RMSE"),
                                        ("Growth multiplier", "to the target month"),
                                        ("Sites flagged", "above the PRB threshold"),
                                        ("Regression R\u00b2", "traffic vs PRB fit"))):
            box = ttk.Frame(summ, style="Card.TFrame", padding=(0, 0, 14, 0))
            box.grid(row=0, column=i, sticky="nw")
            ttk.Label(box, text=key, style="CardMuted.TLabel").pack(anchor="w")
            var = tk.StringVar(value="-")
            self.metric_vars[key] = var
            ttk.Label(box, textvariable=var, style="Metric.TLabel").pack(anchor="w")
            ttk.Label(box, text=sub, style="CardMuted.TLabel").pack(anchor="w")
        self.cap_notes = ttk.Label(summ, text="Run capacity planning to see the results.", style="CardMuted.TLabel",
                                   wraplength=760, justify="left")
        self.cap_notes.grid(row=1, column=0, columnspan=4, sticky="w", pady=(22, 10))
        act = ttk.Frame(summ, style="Card.TFrame")
        act.grid(row=2, column=0, columnspan=4, sticky="w")
        self.open_folder_btn = ttk.Button(act, text="Open output folder", command=self._open_out_folder)
        self.open_folder_btn.pack(side="left")
        self.open_csv_btn = ttk.Button(act, text="Open expansion plan CSV", command=self._open_plan_csv)
        self.open_csv_btn.pack(side="left", padx=8)
        self.open_folder_btn.state(["disabled"])
        self.open_csv_btn.state(["disabled"])

        models = ttk.Frame(right, padding=10, style="Card.TFrame")
        models.columnconfigure(0, weight=1)
        models.rowconfigure((1, 3), weight=1)
        right.add(models, text="Models")
        ttk.Label(models, text="Average hold-out accuracy (validation years 2023, 2024, 2025)",
                  style="CardH2.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.model_table = DataTable(models, "Run capacity planning to compare the forecasting models.", height=4)
        self.model_table.grid(row=1, column=0, sticky="nsew")
        ttk.Label(models, text="Accuracy by validation year", style="CardH2.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 4))
        self.fold_table = DataTable(models, "Per-year metrics appear here.", height=6)
        self.fold_table.grid(row=3, column=0, sticky="nsew")

        fc = ttk.Frame(right, padding=10, style="Card.TFrame")
        fc.columnconfigure(0, weight=1)
        fc.rowconfigure(1, weight=1)
        right.add(fc, text="Forecast")
        ttk.Label(fc, text="Monthly macro network traffic forecast", style="CardH2.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.forecast_table = DataTable(fc, "Run capacity planning to see the monthly forecast.", decimals=2)
        self.forecast_table.grid(row=1, column=0, sticky="nsew")

        reg = ttk.Frame(right, padding=10, style="Card.TFrame")
        reg.columnconfigure(0, weight=1)
        reg.rowconfigure((1, 3, 5), weight=1)
        right.add(reg, text="Regression and growth")
        ttk.Label(reg, text="PRB vs data traffic regression", style="CardH2.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.cap_reg_table = DataTable(reg, "Regression summary appears here.", height=4)
        self.cap_reg_table.grid(row=1, column=0, sticky="nsew")
        ttk.Label(reg, text="Macro growth multipliers", style="CardH2.TLabel").grid(row=2, column=0, sticky="w", pady=(8, 4))
        self.mult_table = DataTable(reg, "Growth multipliers appear here.", height=4, decimals=4)
        self.mult_table.grid(row=3, column=0, sticky="nsew")
        ttk.Label(reg, text="Site forecast distribution", style="CardH2.TLabel").grid(row=4, column=0, sticky="w", pady=(8, 4))
        self.stats_table = DataTable(reg, "Distribution of site forecasts appears here.", height=5)
        self.stats_table.grid(row=5, column=0, sticky="nsew")

        exp = ttk.Frame(right, padding=10, style="Card.TFrame")
        exp.columnconfigure((0, 1), weight=1)
        exp.rowconfigure(1, weight=3)
        exp.rowconfigure(3, weight=1)
        right.add(exp, text="Site expansion")
        bar = ttk.Frame(exp, style="Card.TFrame")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        bar.columnconfigure(0, weight=1)
        self.exp_title = ttk.Label(bar, text="Site expansion table (worst forecast PRB first)", style="CardH2.TLabel")
        self.exp_title.grid(row=0, column=0, sticky="w")
        self.export_btn = ttk.Button(bar, text="Export table as CSV...", command=self._export_expansion)
        self.export_btn.grid(row=0, column=1)
        self.export_btn.state(["disabled"])
        self.exp_table = DataTable(exp, "Run capacity planning to see which sites need expansion.", decimals=4)
        self.exp_table.grid(row=1, column=0, columnspan=2, sticky="nsew")
        ttk.Label(exp, text="Expansion summary", style="CardH2.TLabel").grid(row=2, column=0, sticky="sw", pady=(10, 4))
        ttk.Label(exp, text="By region", style="CardH2.TLabel").grid(row=2, column=1, sticky="sw", pady=(10, 4), padx=(10, 0))
        self.exp_summary = DataTable(exp, "", height=4, decimals=0)
        self.exp_summary.grid(row=3, column=0, sticky="nsew")
        self.exp_region = DataTable(exp, "", height=4, decimals=0)
        self.exp_region.grid(row=3, column=1, sticky="nsew", padx=(10, 0))

        ch = ttk.Frame(right, padding=10, style="Card.TFrame")
        ch.columnconfigure(0, weight=1)
        ch.rowconfigure(0, weight=1)
        right.add(ch, text="Charts")
        self.cap_images = ImageViewer(ch, "Run capacity planning to see the forecast and expansion charts.", self._pretty_image)
        self.cap_images.grid(row=0, column=0, sticky="nsew")
        rep = ttk.Frame(right, padding=10, style="Card.TFrame")
        rep.columnconfigure(0, weight=1)
        rep.rowconfigure(0, weight=1)
        right.add(rep, text="Report")
        self.cap_report = ReportText(rep, "The full console report of the last run appears here.")
        self.cap_report.grid(row=0, column=0, sticky="nsew")


    def _build_log_pane(self) -> None:
        frame = ttk.Frame(self.panes)
        self.panes.add(frame, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        bar = ttk.Frame(frame)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        bar.columnconfigure(0, weight=1)
        ttk.Label(bar, text="Activity log", style="H2.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(bar, text="Save log...", command=self._save_log).grid(row=0, column=1, padx=6)
        ttk.Button(bar, text="Clear", command=self._clear_log).grid(row=0, column=2)
        self.log = tk.Text(frame, height=5, wrap="none", state="disabled", relief="flat",
                           background=PALETTE["log_bg"], foreground=PALETTE["log_fg"], padx=10, pady=6,
                           font=tkfont.Font(family=self.mono_font, size=9), insertbackground=PALETTE["log_fg"])
        ysb = ttk.Scrollbar(frame, orient="vertical", command=self.log.yview)
        xsb = ttk.Scrollbar(frame, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.log.grid(row=1, column=0, sticky="nsew")
        ysb.grid(row=1, column=1, sticky="ns")
        xsb.grid(row=2, column=0, sticky="ew")
        self.log.tag_configure("warn", foreground=PALETTE["log_warn"])
        self.log.tag_configure("error", foreground=PALETTE["log_err"])
        self.log.tag_configure("stage", foreground=PALETTE["log_stage"])

    def _log_text(self, text: str, tag: Optional[str] = None) -> None:
        if tag is None:
            tag = "warn" if "[WARNING]" in text else ("error" if "ERROR" in text else None)
        self.log.configure(state="normal")
        self.log.insert("end", text, tag or ())
        if int(self.log.index("end-1c").split(".")[0]) > 8000:
            self.log.delete("1.0", "2000.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _log_line(self, text: str, tag: Optional[str] = None) -> None:
        self._log_text(text + "\n", tag)

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _save_log(self) -> None:
        target = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="ran_kpi_log.txt",
                                              filetypes=[("Text file", "*.txt")])
        if target:
            try:
                Path(target).write_text(self.log.get("1.0", "end"), encoding="utf-8")
            except OSError as exc:
                self._show_error("Could not save the log", f"{exc}", "Choose a folder you can write to.")

    def _run_task(self, label: str, job: Callable[[Callable[[str], None]], object],
                  on_done: Optional[Callable[[object, str], None]] = None) -> None:
        if self.runner.busy:
            messagebox.showinfo("A task is already running",
                                "Wait for the current task to finish, or press Cancel, before starting another.")
            return
        self._on_done = on_done
        self._task_label = label
        self._task_started = time.time()
        self.status_var.set(label)
        self._log_line("-- " + label, "stage")
        self.progress.configure(mode="indeterminate")
        self.progress.start(14)
        self._refresh_enabled(busy=True)
        self.runner.start(job)

    def _poll(self) -> None:
        try:
            chunk: List[str] = []
            for _ in range(400):
                try:
                    msg = self.runner.messages.get_nowait()
                except Exception:
                    break
                kind = msg[0]
                if kind == "log":
                    chunk.append(msg[1])
                    continue
                if chunk:
                    self._log_text("".join(chunk))
                    chunk = []
                if kind == "stage":
                    self.status_var.set(msg[1])
                    self._log_line("-- " + msg[1], "stage")
                elif kind == "done":
                    self._finish_task()
                    self.status_var.set("Done: " + self._task_label.rstrip(".").split(" (")[0])
                    if self._on_done:
                        self._safe_callback(self._on_done, msg[1], msg[2])
                elif kind == "cancelled":
                    self._finish_task()
                    self.status_var.set("Cancelled. Nothing was saved from the interrupted run.")
                    self._log_line("-- Cancelled by the user.", "warn")
                elif kind == "error":
                    self._finish_task()
                    self._handle_error(msg[1], msg[2])
            if chunk:
                self._log_text("".join(chunk))
            if self.runner.busy:
                self.elapsed_var.set("{:d}:{:02d}".format(*divmod(int(time.time() - self._task_started), 60)))
        finally:
            self.root.after(100, self._poll)

    def _safe_callback(self, fn: Callable[[object, str], None], result: object, report: str) -> None:
        try:
            fn(result, report)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.file_log.error("display error\n%s", tb)
            self._log_text(tb, "error")
            self._show_error("The results could not be displayed", f"{type(exc).__name__}: {exc}",
                             "The analysis finished; see the log pane for details.")

    def _finish_task(self) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.elapsed_var.set("")
        self._refresh_enabled(busy=False)

    def _handle_error(self, exc: BaseException, tb: str) -> None:
        info = B.describe_error(exc)
        if not isinstance(exc, B.UserInputError):
            self.file_log.error("%s\n%s", info.message, tb)
            self._log_text(tb, "error")
        self._log_line("ERROR: {} - {}".format(info.title, info.message.replace("\n", " ")), "error")
        self.status_var.set("Failed: " + info.title)
        self._show_error(info.title, info.message, info.hint, warn=isinstance(exc, B.UserInputError))
        if isinstance(exc, ModuleNotFoundError):
            self.status_var.set("A required package is missing - see the message.")

    def _show_error(self, title: str, message: str, hint: str = "", warn: bool = False) -> None:
        text = message + (("\n\n" + hint) if hint else "")
        (messagebox.showwarning if warn else messagebox.showerror)(title, text)

    def _on_cancel(self) -> None:
        if self.runner.busy:
            self.runner.cancel()
            self.status_var.set("Cancelling... (stops after the current model fit)")
            self.cancel_btn.state(["disabled"])

    def _refresh_enabled(self, busy: Optional[bool] = None) -> None:
        busy = self.runner.busy if busy is None else busy
        ready = self.backend.ready
        has_data = self.backend.data is not None
        for btn in self._action_buttons:
            needs_data = btn in (self.kpi_run, self.cap_run, self.bad_run)
            ok = ready and not busy and (has_data or not needs_data)
            btn.state(["!disabled"] if ok else ["disabled"])
        self.cancel_btn.state(["!disabled"] if busy else ["disabled"])

    def _on_close(self) -> None:
        if self.runner.busy:
            if not messagebox.askyesno("Exit while a task is running?",
                                       "An analysis is still running. Exit anyway? Unsaved results will be lost."):
                return
            self.runner.cancel()
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        finally:
            self.root.destroy()

    def _after_import(self, _result: object, _report: str) -> None:
        d = self.backend.defaults
        self.rate_cap_var.set(f"{d['rate_cap']:g}")
        self.traffic_winsor_var.set(f"{d['traffic_winsor_pct']:g}")
        self.drop_winsor_var.set(f"{d['drop_rate_winsor_pct']:g}")
        self.modz_var.set(f"{d['modz_threshold']:g}")
        self.prb_var.set(f"{d['prb_threshold']:g}")
        self.baseline_var.set(f"{d['baseline_months']:g}")
        target = pd.Timestamp(self.backend.pipeline.Config.FORECAST_END)
        self.target_label.configure(text=f"Forecast target: {target:%b %Y} (fixed in the script)")
        self.status_var.set("Ready. Choose a workbook on the Data tab.")
        self._refresh_enabled()

    def _toggle_advanced(self) -> None:
        self._adv_visible = not self._adv_visible
        if self._adv_visible:
            self.adv_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
            self.adv_btn.configure(text="Hide cleaning thresholds")
        else:
            self.adv_frame.grid_remove()
            self.adv_btn.configure(text="Show cleaning thresholds")

    def _reset_cleaning(self) -> None:
        if self.backend.ready:
            d = self.backend.defaults
            self.rate_cap_var.set(f"{d['rate_cap']:g}")
            self.traffic_winsor_var.set(f"{d['traffic_winsor_pct']:g}")
            self.drop_winsor_var.set(f"{d['drop_rate_winsor_pct']:g}")
            self.modz_var.set(f"{d['modz_threshold']:g}")

    def _browse_excel(self) -> None:
        path = filedialog.askopenfilename(title="Choose the Excel workbook",
                                          filetypes=[("Excel workbooks", "*.xlsx *.xlsm *.xls"), ("All files", "*.*")])
        if path:
            self.excel_var.set(path)
            if not self._out_user_set:
                self.out_var.set(str(Path(path).parent))

    def _browse_out(self) -> None:
        path = filedialog.askdirectory(title="Choose the output folder", mustexist=False)
        if path:
            self.out_var.set(path)
            self._out_user_set = True

    @staticmethod
    def _parse_number(text: str, name: str, low: float, high: float, integer: bool = False) -> float:
        try:
            value = float(text.strip().replace(",", "."))
        except ValueError:
            raise B.UserInputError(f"{name} must be a number (you entered '{text}').") from None
        if integer and value != int(value):
            raise B.UserInputError(f"{name} must be a whole number.")
        if not (low <= value <= high):
            raise B.UserInputError(f"{name} must be between {low:g} and {high:g}.")
        return int(value) if integer else value

    def _cleaning_settings(self) -> Dict[str, float]:
        n = self._parse_number
        return {
            "rate_cap": n(self.rate_cap_var.get(), "Rate cap", 1, 1000),
            "traffic_winsor_pct": n(self.traffic_winsor_var.get(), "Traffic winsor percentile", 0.5, 1),
            "drop_rate_winsor_pct": n(self.drop_winsor_var.get(), "Drop-rate winsor percentile", 0.5, 1),
            "modz_threshold": n(self.modz_var.get(), "Modified z-score threshold", 0.5, 100),
        }

    def _validated(self, action: Callable[[], None]) -> None:
        try:
            action()
        except B.UserInputError as exc:
            self._handle_error(exc, "")

    def _on_load(self) -> None:
        def go() -> None:
            settings = self._cleaning_settings()
            excel, out = self.excel_var.get(), self.out_var.get()
            self._run_task("Loading and cleaning the workbook...",
                           lambda stage: self.backend.load(excel, out, settings), self._after_load)
        self._validated(go)

    def _after_load(self, data: B.LoadedData, _report: str) -> None:
        self.out_var.set(str(data.out_dir))
        self.chip_var.set(f"{data.excel_path.name}  |  {data.ps['Site ID'].nunique():,} sites")
        sv = self.summary_vars
        sv["Sites"].set(f"{data.ps['Site ID'].nunique():,} with PS data, {len(data.sites):,} configured")
        sv["PS rows"].set(f"{len(data.ps):,} after cleaning (from {data.raw_rows['ps']:,})")
        sv["CS rows"].set(f"{len(data.cs):,} after cleaning (from {data.raw_rows['cs']:,})")
        months = data.months["PS"]
        sv["Months (PS)"].set(f"{len(months)}  ({months[0]} to {months[-1]})" if months else "-")
        sv["Regions"].set(str(len(data.regions["PS"])))
        t = data.traffic
        sv["Traffic history"].set(f"{len(t)} months  ({t.index[0]:%b %Y} to {t.index[-1]:%b %Y})")
        self.cleaning_table.set_dataframe(data.cleaning_report, "No corrections were necessary.")
        self._set_checks(data.checks)
        for table in (self.model_table, self.fold_table, self.forecast_table,
                      self.cap_reg_table, self.mult_table, self.stats_table, self.exp_table, self.exp_summary,
                      self.exp_region, self.bad_table):
            table.set_dataframe(None)
        for viewer in (self.kpi_images, self.cap_images):
            viewer.set_images([])
        self.kpi_report.set_text("")
        self.cap_report.set_text("")
        for var in self.metric_vars.values():
            var.set("-")
        self.cap_notes.configure(text="Run capacity planning to see the results.", style="CardMuted.TLabel")
        for b in (self.open_folder_btn, self.open_csv_btn, self.export_btn, self.bad_open):
            b.state(["disabled"])
        self.bad_msg.configure(text="")
        self._on_family_change()
        self._refresh_enabled()
        self._log_line(f"-- Loaded {data.excel_path.name}; output folder: {data.out_dir}", "stage")

    def _set_checks(self, checks: Sequence[tuple]) -> None:
        for child in self.checks_card.winfo_children():
            child.destroy()
        styles = {"ok": "Ok.TLabel", "warn": "Warn.TLabel", "error": "Err.TLabel", "info": "Muted.TLabel"}
        prefix = {"ok": "OK: ", "warn": "Warning: ", "error": "Problem: ", "info": ""}
        for i, (level, text) in enumerate(checks):
            ttk.Label(self.checks_card, text=prefix[level] + text, style=styles[level], wraplength=640,
                      justify="left", background=PALETTE["surface"]).grid(row=i, column=0, sticky="w", pady=2)

    def _family_frames(self) -> List[str]:
        return {"CS": ["CS"], "PS": ["PS"], "BOTH": ["CS", "PS"]}[self.family_var.get()]

    def _on_family_change(self) -> None:
        fam = self.family_var.get()
        data = self.backend.data
        months = sorted(set().union(*[set(data.months[k]) for k in self._family_frames()])) if data else []
        regions = sorted(set().union(*[set(data.regions[k]) for k in self._family_frames()])) if data else []
        for lb, items, flag in ((self.month_list, months, self.all_months_var),
                                (self.region_list, regions, self.all_regions_var)):
            lb.configure(state="normal")
            lb.delete(0, "end")
            for item in items:
                lb.insert("end", item)
            self._toggle_all(flag, lb)
        self._kpi_choices = {}
        if self.backend.pipeline is not None:
            for ds in self._family_frames()[::-1] if fam == "BOTH" else self._family_frames():
                cols = self.backend.pipeline.CS_NUMERIC if ds == "CS" else self.backend.pipeline.PS_NUMERIC
                for c in cols:
                    label = c if fam != "BOTH" else f"{ds}: {c}"
                    self._kpi_choices[label] = (ds, c)
        self.kpi_combo["values"] = list(self._kpi_choices)
        if self._kpi_choices:
            self.kpi_var.set(next(iter(self._kpi_choices)))
        self._refresh_kpi_controls()

    def _refresh_kpi_controls(self) -> None:
        self.kpi_combo.configure(state="readonly" if self.outlier_var.get() and self._kpi_choices else "disabled")

    @staticmethod
    def _toggle_all(flag: tk.BooleanVar, lb: tk.Listbox) -> None:
        if flag.get():
            lb.selection_clear(0, "end")
            lb.configure(state="disabled")
        else:
            lb.configure(state="normal")

    @staticmethod
    def _picked(flag: tk.BooleanVar, lb: tk.Listbox, noun: str) -> Optional[List[str]]:
        if flag.get():
            return None
        chosen = [lb.get(i) for i in lb.curselection()]
        if not chosen:
            raise B.UserInputError(f"Select at least one {noun}, or tick 'All {noun}s'.")
        return chosen

    def _on_run_kpi(self) -> None:
        def go() -> None:
            params = B.KPIParams(
                family=self.family_var.get(),
                months=self._picked(self.all_months_var, self.month_list, "month"),
                regions=self._picked(self.all_regions_var, self.region_list, "region"),
                kpi=self._kpi_choices.get(self.kpi_var.get()) if self.outlier_var.get() else None,
                run_correlation=False,
                run_regression=False)
            self._run_task("Running KPI analysis...", lambda stage: self.backend.kpi_analysis(params), self._after_kpi)
        self._validated(go)

    def _after_kpi(self, outcome: B.KPIOutcome, report: str) -> None:
        res = outcome.result
        self.kpi_images.set_images(res.images)
        self.kpi_report.set_text(report)
        self.kpi_nb.select(0 if res.images else 1)
        self.status_var.set(f"KPI analysis complete: {len(res.images)} chart(s). Filters: {res.filters}")

    def _planning_numbers(self) -> tuple:
        threshold = self._parse_number(self.prb_var.get(), "PRB threshold", 1, 100)
        baseline = self._parse_number(self.baseline_var.get(), "Baseline window", 1, 12, integer=True)
        return float(threshold), int(baseline)

    def _on_run_capacity(self) -> None:
        def go() -> None:
            threshold, baseline = self._planning_numbers()
            params = B.CapacityParams(threshold, baseline, self.refit_var.get())
            self._run_task("Running capacity planning...",
                           lambda stage: self.backend.capacity_planning(params, stage), self._after_capacity)
        self._validated(go)

    def _after_capacity(self, res, report: str) -> None:
        self._last_capacity = res
        self.metric_vars["Selected model"].set(res.best_model)
        self.metric_vars["Growth multiplier"].set(f"x{res.multiplier:.4f}")
        self.metric_vars["Sites flagged"].set(f"{len(res.expansion_table):,} of {res.n_sites:,}")
        r2_row = res.regression_summary[res.regression_summary["Statistic"] == "R^2"]
        self.metric_vars["Regression R\u00b2"].set(str(r2_row["Value"].iloc[0]) if not r2_row.empty else "-")
        notes: List[str] = [f"Baseline months: {', '.join(res.baseline_periods)}.",
                            f"{res.n_inactive_sites:,} site(s) had no positive-traffic baseline (treated as inactive)."]
        style = "CardMuted.TLabel"
        if res.csv_path is None:
            notes.append("The expansion plan CSV could not be written - see the log for the reason.")
            style = "Warn.TLabel"
        if res.unrecognised_configs:
            notes.append("Unrecognised configuration text (edit ExpansionStateMachine.RULES in Final_Project_G2.py): "
                         + "; ".join(f"'{k}' x{v}" for k, v in res.unrecognised_configs.items()))
            style = "Warn.TLabel"
        if not res.expansion_table.empty and res.expansion_table["Forecasted PRB (%)"].max() > 100:
            notes.append("Some forecast PRB values exceed 100%: the linear regression extrapolates past full capacity, "
                         "so read them as 'well above the threshold'.")
        if res.expansion_table.empty:
            notes.append("No site is forecast above the threshold - no expansion required.")
        self.cap_notes.configure(text="\n".join(notes), style=style)

        self.model_table.set_dataframe(res.model_comparison)
        self.fold_table.set_dataframe(res.fold_metrics)
        self.forecast_table.set_dataframe(res.forecast_table)
        self.cap_reg_table.set_dataframe(res.regression_summary)
        self.mult_table.set_dataframe(res.multiplier_table)
        self.stats_table.set_dataframe(res.site_stats)
        self.exp_table.set_dataframe(res.expansion_table, "No site is forecast above the threshold.")
        self.exp_summary.set_dataframe(res.expansion_summary if not res.expansion_summary.empty else None, "No expansions.")
        self.exp_region.set_dataframe(res.expansion_by_region if not res.expansion_by_region.empty else None, "No expansions.")
        self.exp_title.configure(text=f"Site expansion table: {len(res.expansion_table):,} site(s), worst forecast PRB first")
        self.cap_images.set_images(res.images)
        self.cap_report.set_text(report)
        self.open_folder_btn.state(["!disabled"])
        self.open_csv_btn.state(["!disabled"] if res.csv_path else ["disabled"])
        self.export_btn.state(["!disabled"] if not res.expansion_table.empty else ["disabled"])
        self.cap_nb.select(0)
        self.status_var.set(f"Capacity planning complete: {len(res.expansion_table):,} site(s) flagged "
                            f"(model: {res.best_model}, multiplier x{res.multiplier:.3f}).")

    def _export_expansion(self) -> None:
        df = self.exp_table.dataframe
        if df is None or df.empty:
            return
        target = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="Site_Expansion_Plan_export.csv",
                                              filetypes=[("CSV file", "*.csv")])
        if target:
            try:
                df.to_csv(target, index=False)
                self.status_var.set(f"Exported {len(df):,} rows to {target}")
            except OSError as exc:
                self._show_error("Could not export the table", str(exc), "Close the file if it is open in Excel, or pick another location.")

    def _open_out_folder(self) -> None:
        if self.backend.data:
            open_path(self.backend.data.out_dir)

    def _open_plan_csv(self) -> None:
        if self._last_capacity is not None and self._last_capacity.csv_path:
            open_path(self._last_capacity.csv_path)

    def _on_run_bad(self) -> None:
        def go() -> None:
            threshold = self._parse_number(self.thr_var.get(), "Throughput threshold", 0, 1000)
            self._run_task("Finding bad sites...", lambda stage: self.backend.bad_sites(float(threshold)), self._after_bad)
        self._validated(go)

    def _after_bad(self, out: B.BadSitesOutcome, _report: str) -> None:
        if out.table is None:
            self.bad_table.set_dataframe(None, "There are no PS rows to analyse.")
            self.bad_msg.configure(text="No PS rows available.")
            return
        self.bad_table.set_dataframe(out.table, f"No site has an average Throughput <= {self.thr_var.get()} Mbps.")
        n = len(out.table)
        self.bad_msg.configure(text=(f"{n:,} site(s) below or equal to {self.thr_var.get()} Mbps. Saved to CSV."
                                     if n else f"No site has an average Throughput <= {self.thr_var.get()} Mbps."))
        self.bad_open.state(["!disabled"] if out.csv_path else ["disabled"])
        self.kpi_nb.select(2) 

    def _pretty_image(self, path: Path) -> str:
        stem = path.stem
        P = self.backend.pipeline
        kpis = {}
        if P is not None:
            kpis = {P._norm_name(c): c for c in list(P.CS_NUMERIC) + list(P.PS_NUMERIC)}
        for prefix, label in (("outlier_boxplot_", "Outliers: "), ("trend_lineplot_", "Trend: ")):
            if stem.startswith(prefix):
                return label + kpis.get(stem[len(prefix):], stem[len(prefix):])
        if stem.startswith("correlation_"):
            return "Correlation: " + stem[len("correlation_"):].replace("_vs_", " vs ")
        fixed = {"prb_regression": "PRB regression", "macro_traffic_forecast": "Traffic forecast and back-test",
                 "configuration_before_vs_after": "Configuration shift (before vs after)",
                 "regional_expansions_stacked": "Expansions by region"}
        return fixed.get(stem, stem.replace("_", " "))

def main() -> int:
    try: 
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except Exception: 
        pass
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()
    return 0

if __name__ == "__main__":
    sys.exit(main())