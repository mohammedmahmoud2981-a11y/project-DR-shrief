"""
project_Dr_shreif_ui.py - desktop interface (Tkinter / ttk) for project_Dr_shreif.py

Run it from the folder that contains project_Dr_shreif.py:

    python project_Dr_shreif_ui.py

The command-line program is NOT modified and keeps working exactly as before. This window only
calls its functions (image_to_matrix, forward_elimination, check_consistency, print_matrix,
fmt) and its settings (SHOW_STEPS, DECIMALS, THICKEN_INK), so the mathematics and the OCR flow
are the ones in project_Dr_shreif.py.

What the program does - and what it does NOT do:
  * It performs FORWARD elimination on an augmented matrix with exact Fraction arithmetic and
    produces a Row Echelon Form (leading 1 in each pivot row, zeros below each pivot).
  * It does NOT produce the Reduced Row Echelon Form and does NOT compute the values of the
    variables.
  * Rounded decimals change only how numbers are DISPLAYED; the calculation is always exact.
"""

import contextlib
import glob
import importlib.util
import io
import os
import queue
import re
import sys
import threading
import webbrowser
from collections import deque
from fractions import Fraction
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
APP_TITLE = "Forward Gaussian Elimination"
MAX_ROWS = 60
MAX_COLS = 61                 # 60 unknowns + the right-hand-side column
BIG_ROWS, BIG_COLS = 10, 11   # same "big matrix" limits as main() in project_Dr_shreif.py
CORE_NAMES = ("project_Dr_shreif.py", "project Dr shreif.py")
OCR_PACKAGES = ("easyocr", "cv2")
OCR_INSTALL_COMMAND = "pip install easyocr opencv-python"

SCOPE_TEXT = ("Forward elimination only: the result is a Row Echelon Form (leading 1 in each pivot row, "
              "zeros below each pivot). It is NOT the Reduced Row Echelon Form, and the values of the "
              "variables are NOT calculated. The calculation uses exact fractions.")

COLOR_CELL = "#ffffff"
COLOR_RHS = "#e8f1fb"        # right-hand-side column
COLOR_INVALID = "#ffc9c9"
COLOR_CHECK = "#ffe58a"      # the OCR was not sure about this cell
COLOR_ERROR_TEXT = "#9b1c1c"
COLOR_OK_TEXT = "#1d5e2f"
COLOR_HINT = "#4a4a4a"
COLOR_ACCENT = "#0b5cad"


# ---------------------------------------------------------------
# Finding and loading the existing program
# ---------------------------------------------------------------
def find_core(folder):
    """Look for the main program next to this file (any of its usual names)."""
    for name in CORE_NAMES:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    this_file = os.path.abspath(__file__)
    for path in sorted(glob.glob(os.path.join(folder, "project*shreif*.py"))):
        if os.path.abspath(path) != this_file and not path.endswith("_ui.py"):
            return path
    return None


def load_core(path):
    """Import project_Dr_shreif.py without running its main() (its name is not '__main__')."""
    spec = importlib.util.spec_from_file_location("project_Dr_shreif_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    needed = ("image_to_matrix", "forward_elimination", "check_consistency", "print_matrix", "fmt")
    missing = [n for n in needed if not hasattr(module, n)]
    if missing:
        raise ImportError("This file does not look like the Gaussian elimination program "
                          f"(missing: {', '.join(missing)}).")
    return module


# ---------------------------------------------------------------
# Helpers without any GUI (easy to test)
# ---------------------------------------------------------------
def parse_cell(text):
    """Same rule as the command-line program (Fraction(text)).
    Returns (value, None) or (None, explanation)."""
    text = text.strip()
    if text == "":
        return None, "is empty. Enter a number (type 0 for zero)."
    if "e" in text.lower():
        return None, (f"'{text}' uses scientific notation, which is not accepted here. "
                      "Type the number in full, for example 1000 or 0.001.")
    try:
        return Fraction(text), None
    except ZeroDivisionError:
        return None, f"'{text}' has a denominator of zero, which is not allowed."
    except ValueError:
        return None, (f"'{text}' is not a valid number. Use an integer (-3), a decimal with a dot (2.5) "
                      "or a fraction (1/2).")


def frac_text(value):
    """Fraction -> text for a grid cell: '3' or '-1/2'."""
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def parse_table_text(text):
    """Split pasted text into rows of cell strings. One row per line; cells are separated by
    spaces, tabs, semicolons or '|'. Commas are NOT separators (they could be decimal commas).
    Brackets like those printed by the program are ignored."""
    rows = []
    for line in text.splitlines():
        for ch in "\t;|[]":
            line = line.replace(ch, " ")
        parts = line.split()
        if parts:
            rows.append(parts)
    if not rows:
        raise ValueError("There is no text to read.")
    if len({len(r) for r in rows}) != 1:
        raise ValueError("The rows have different numbers of values.")
    return rows


def ocr_available():
    """True when EasyOCR and OpenCV can be imported (checked without importing them)."""
    return all(importlib.util.find_spec(name) is not None for name in OCR_PACKAGES)


class QueueWriter(io.TextIOBase):
    """Stands in for sys.stdout while the program runs in a worker thread: whatever the program
    prints is put in a queue and shown by the window from the main thread."""

    def __init__(self, q, kind, keep=None):
        self.q, self.kind, self.keep = q, kind, keep

    def write(self, text):
        if text:
            self.q.put((self.kind, text))
            if self.keep is not None:
                self.keep.append(text)
        return len(text)

    def flush(self):
        pass


def capture(function, *args):
    """Run a printing function of the program and return what it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        function(*args)
    return buffer.getvalue()


# ---------------------------------------------------------------
# The window
# ---------------------------------------------------------------
class App(tk.Tk):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.title(APP_TITLE)
        self.geometry("1200x880")
        self.minsize(980, 700)

        self.q = queue.Queue()                  # messages coming from the worker threads
        self.cells = []                         # cells[i][j] = (StringVar, Entry)
        self.invalid = {}                       # (i, j) -> explanation
        self.warn_cells = set()                 # cells the OCR was unsure about
        self.pending_warnings = set()
        self.busy = False
        self._bulk = False                      # True while many cells are changed at once
        self._was_big = None
        self.image_path = ""
        self.result = None                      # data of the last calculation (used by Save)
        self.stale = False                      # True when the grid changed after that calculation
        self.ocr_ok = ocr_available()
        self.buffers = {"result": deque(), "steps": deque(), "messages": deque()}

        self._setup_style()
        self._build_menu()
        self._build_ui()
        self.build_grid(3, 4)
        self.load_example()
        self._bind_shortcuts()
        self.after(30, self._poll)

    # =============================================================
    # Look and layout
    # =============================================================
    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("vista" if "vista" in style.theme_names() else "clam")
        tkfont.nametofont("TkDefaultFont").configure(size=10)
        tkfont.nametofont("TkTextFont").configure(size=10)
        style.configure("Accent.TButton", font=("TkDefaultFont", 11, "bold"), padding=(14, 6))
        style.configure("Hint.TLabel", foreground=COLOR_HINT)
        style.configure("Title.TLabel", font=("TkDefaultFont", 15, "bold"), foreground=COLOR_ACCENT)
        families = set(tkfont.families())
        self.mono = next((f for f in ("Consolas", "DejaVu Sans Mono", "Menlo", "Courier New")
                          if f in families), "Courier")

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Choose image...", accelerator="Ctrl+O", underline=0,
                              command=self.on_choose_image)
        file_menu.add_command(label="Save results...", accelerator="Ctrl+S", underline=0,
                              command=self.on_save)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", underline=1, command=self.destroy)
        menubar.add_cascade(label="File", underline=0, menu=file_menu)

        run_menu = tk.Menu(menubar, tearoff=False)
        run_menu.add_command(label="Calculate (forward elimination)", accelerator="Ctrl+Enter", underline=0,
                             command=self.on_calculate)
        menubar.add_cascade(label="Run", underline=0, menu=run_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="What this program does", underline=0, command=self.show_about)
        help_menu.add_command(label="Reading matrices from images (OCR setup)", underline=0,
                              command=self.show_ocr_help)
        menubar.add_cascade(label="Help", underline=0, menu=help_menu)
        self.configure(menu=menubar)

    def _bind_shortcuts(self):
        self.bind_all("<Control-Return>", lambda e: self.on_calculate())
        self.bind_all("<F5>", lambda e: self.on_calculate())
        self.bind_all("<Control-o>", lambda e: self.on_choose_image())
        self.bind_all("<Control-s>", lambda e: self.on_save())
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(seq, self._on_wheel, add="+")

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ---- header ----
        head = ttk.Frame(self, padding=(12, 10, 12, 4))
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(0, weight=1)
        ttk.Label(head, text=APP_TITLE, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        scope = tk.Label(head, text=SCOPE_TEXT, justify="left", anchor="w", background="#e8f1fb",
                         foreground="#0a3a6b", padx=10, pady=6, relief="flat")
        scope.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        scope.bind("<Configure>", lambda e: scope.configure(wraplength=max(200, e.width - 20)))

        # ---- the two big areas, separated by a draggable bar ----
        self.paned = ttk.PanedWindow(self, orient="vertical")
        self.paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        top = ttk.Frame(self.paned)
        bottom = ttk.Frame(self.paned)
        self.paned.add(top, weight=4)
        self.paned.add(bottom, weight=3)
        self._build_input_area(top)
        self._build_result_area(bottom)

        # ---- status bar ----
        self.var_status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.var_status, anchor="w", relief="sunken",
                  padding=(8, 3)).grid(row=2, column=0, sticky="ew")

    # ---------- input area ----------
    def _build_input_area(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        matrix_box = ttk.LabelFrame(parent, text=" 1. Augmented matrix ", padding=8)
        matrix_box.grid(row=0, column=0, sticky="nsew")
        matrix_box.columnconfigure(0, weight=1)
        matrix_box.rowconfigure(4, weight=1)

        # row 0: size controls
        size_row = ttk.Frame(matrix_box)
        size_row.grid(row=0, column=0, sticky="ew")
        ttk.Label(size_row, text="Rows:").pack(side="left")
        self.var_rows = tk.StringVar(value="3")
        ttk.Spinbox(size_row, from_=1, to=MAX_ROWS, width=4, textvariable=self.var_rows).pack(side="left", padx=(4, 14))
        ttk.Label(size_row, text="Columns (including the right-hand side):").pack(side="left")
        self.var_cols = tk.StringVar(value="4")
        ttk.Spinbox(size_row, from_=2, to=MAX_COLS, width=4, textvariable=self.var_cols).pack(side="left", padx=(4, 14))
        ttk.Button(size_row, text="Create grid", command=self.on_create_grid).pack(side="left", padx=3)
        ttk.Button(size_row, text="Paste matrix", command=self.on_paste_button).pack(side="left", padx=3)
        ttk.Button(size_row, text="Fill blanks with 0", command=self.on_fill_blanks).pack(side="left", padx=3)
        ttk.Button(size_row, text="Clear", command=self.on_clear).pack(side="left", padx=3)
        ttk.Button(size_row, text="Example", command=self.load_example).pack(side="left", padx=3)

        ttk.Label(matrix_box, style="Hint.TLabel", wraplength=1100, justify="left",
                  text="Needs at least 2 columns: the coefficients, then the right-hand side (shaded and separated by "
                       "the blue bar). Accepted values: integers (-3), decimals with a dot (2.5) and fractions (1/2). "
                       "Enter / Up / Down move between rows; Ctrl+V in a cell pastes a table."
                  ).grid(row=1, column=0, sticky="w", pady=(4, 2))

        # row 2: image controls (two short lines so nothing is cut off on narrow windows)
        image_box = ttk.Frame(matrix_box)
        image_box.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        line1 = ttk.Frame(image_box)
        line1.pack(fill="x")
        ttk.Label(line1, text="Or read it from an image:").pack(side="left")
        ttk.Button(line1, text="Choose image...", command=self.on_choose_image).pack(side="left", padx=(6, 6))
        self.lbl_image = ttk.Label(line1, text="(none chosen)", style="Hint.TLabel", width=30)
        self.lbl_image.pack(side="left")
        ttk.Button(line1, text="Open image", command=self.on_open_image).pack(side="left", padx=3)
        line2 = ttk.Frame(image_box)
        line2.pack(fill="x", pady=(4, 0))
        ttk.Label(line2, text="Size, if you know it (rows x columns):").pack(side="left")
        self.var_hint_r, self.var_hint_c = tk.StringVar(), tk.StringVar()
        ttk.Entry(line2, width=4, textvariable=self.var_hint_r).pack(side="left", padx=(6, 0))
        ttk.Label(line2, text="x").pack(side="left", padx=3)
        ttk.Entry(line2, width=4, textvariable=self.var_hint_c).pack(side="left")
        self.var_thicken = tk.BooleanVar(value=bool(getattr(self.core, "THICKEN_INK", True)))
        ttk.Checkbutton(line2, text="Thicken ink", variable=self.var_thicken).pack(side="left", padx=12)
        self.btn_read = ttk.Button(line2, text="Read matrix from image", command=self.on_read_image)
        self.btn_read.pack(side="left", padx=3)

        # row 3: OCR availability / review notice
        self.lbl_ocr = tk.Label(matrix_box, anchor="w", justify="left", padx=2)
        self.lbl_ocr.grid(row=3, column=0, sticky="ew", pady=(2, 2))
        self.lbl_ocr.bind("<Configure>", lambda e: self.lbl_ocr.configure(wraplength=max(200, e.width - 10)))
        self._refresh_ocr_label()

        # row 4: the grid with scrollbars
        grid_box = ttk.Frame(matrix_box)
        grid_box.grid(row=4, column=0, sticky="nsew")
        grid_box.columnconfigure(0, weight=1)
        grid_box.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(grid_box, highlightthickness=1, highlightbackground="#b5b5b5",
                                background="#f3f3f3", height=190)
        sy = ttk.Scrollbar(grid_box, orient="vertical", command=self.canvas.yview)
        sx = ttk.Scrollbar(grid_box, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        self.grid_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # row 5: validation messages
        self.lbl_validation = tk.Label(matrix_box, anchor="w", justify="left", foreground=COLOR_ERROR_TEXT, padx=2)
        self.lbl_validation.grid(row=5, column=0, sticky="ew", pady=(3, 0))
        self.lbl_validation.bind("<Configure>", lambda e: self.lbl_validation.configure(wraplength=max(200, e.width - 10)))

        # ---- options + calculate ----
        opt = ttk.LabelFrame(parent, text=" 2. Options ", padding=8)
        opt.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.var_steps = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Show row-operation steps", variable=self.var_steps).pack(side="left")
        ttk.Separator(opt, orient="vertical").pack(side="left", fill="y", padx=12)
        ttk.Label(opt, text="Show numbers as:").pack(side="left")
        self.var_format = tk.StringVar(value="exact")
        ttk.Radiobutton(opt, text="Exact fractions", variable=self.var_format, value="exact",
                        command=self._sync_places_state).pack(side="left", padx=(8, 2))
        ttk.Radiobutton(opt, text="Rounded decimals", variable=self.var_format, value="decimal",
                        command=self._sync_places_state).pack(side="left", padx=(8, 2))
        ttk.Label(opt, text="decimal places:").pack(side="left", padx=(6, 2))
        self.var_places = tk.StringVar(value="4")
        self.spin_places = ttk.Spinbox(opt, from_=0, to=15, width=3, textvariable=self.var_places)
        self.spin_places.pack(side="left")
        self.progress = ttk.Progressbar(opt, mode="indeterminate", length=130)
        self.progress.pack(side="right")
        self.btn_calc = ttk.Button(opt, text="Calculate  (Ctrl+Enter)", style="Accent.TButton",
                                   command=self.on_calculate)
        self.btn_calc.pack(side="right", padx=12)
        self._sync_places_state()

    # ---------- result area ----------
    def _build_result_area(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        box = ttk.LabelFrame(parent, text=" 3. Result ", padding=8)
        box.grid(row=0, column=0, sticky="nsew")
        box.columnconfigure(0, weight=1)
        box.rowconfigure(1, weight=1)

        self.lbl_result = tk.Label(box, text="Press Calculate to run forward elimination.", anchor="w",
                                   justify="left", padx=8, pady=5, background="#f0f0f0", foreground="#222222")
        self.lbl_result.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.lbl_result.bind("<Configure>", lambda e: self.lbl_result.configure(wraplength=max(200, e.width - 20)))

        self.notebook = ttk.Notebook(box)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.texts = {}
        for key, title in (("result", "Original and final matrix"), ("steps", "Row-operation steps"),
                           ("messages", "Messages")):
            frame = ttk.Frame(self.notebook)
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            text = tk.Text(frame, wrap="none", font=(self.mono, 10), height=8, state="disabled",
                           background="#fbfbfb", padx=8, pady=6)
            sy = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
            sx = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
            text.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
            text.grid(row=0, column=0, sticky="nsew")
            sy.grid(row=0, column=1, sticky="ns")
            sx.grid(row=1, column=0, sticky="ew")
            self.notebook.add(frame, text=title)
            self.texts[key] = text

        buttons = ttk.Frame(box)
        buttons.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.btn_save = ttk.Button(buttons, text="Save results...  (Ctrl+S)", command=self.on_save)
        self.btn_save.pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Copy final matrix", command=self.on_copy).pack(side="left", padx=6)
        ttk.Button(buttons, text="Clear results", command=self.clear_results).pack(side="left", padx=6)

    # =============================================================
    # The grid of cells
    # =============================================================
    def _cell_color(self, i, j):
        if (i, j) in self.invalid:
            return COLOR_INVALID
        if (i, j) in self.warn_cells:
            return COLOR_CHECK
        return COLOR_RHS if j == len(self.cells[0]) - 1 else COLOR_CELL

    def build_grid(self, rows, cols, values=None):
        """(Re)create the grid. values = rows of text (or None for an empty grid)."""
        self._bulk = True
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self.cells, self.invalid, self.warn_cells = [], {}, set()
        self.var_rows.set(str(rows))
        self.var_cols.set(str(cols))
        self._hide_review_notice()

        head_font = ("TkDefaultFont", 9)
        ttk.Label(self.grid_frame, text="").grid(row=0, column=0)
        for j in range(cols):
            col = j + 1 if j < cols - 1 else cols + 1
            if j < cols - 1:
                ttk.Label(self.grid_frame, text=f"x{j + 1}", anchor="center", font=head_font,
                          foreground=COLOR_HINT).grid(row=0, column=col)
            else:
                ttk.Label(self.grid_frame, text="RHS", anchor="center", font=("TkDefaultFont", 9, "bold"),
                          foreground=COLOR_ACCENT).grid(row=0, column=col)
        for i in range(rows):
            ttk.Label(self.grid_frame, text=f"R{i + 1}", width=4, anchor="e", font=head_font,
                      foreground=COLOR_HINT).grid(row=i + 1, column=0, padx=(2, 5))
            row_cells = []
            for j in range(cols):
                var = tk.StringVar(value=values[i][j] if values else "")
                ent = tk.Entry(self.grid_frame, textvariable=var, width=9, justify="right",
                               font=(self.mono, 11), relief="solid", borderwidth=1,
                               background=COLOR_RHS if j == cols - 1 else COLOR_CELL,
                               highlightthickness=2, highlightbackground="#c8c8c8",
                               highlightcolor=COLOR_ACCENT)
                col = j + 1 if j < cols - 1 else cols + 1
                ent.grid(row=i + 1, column=col, padx=1, pady=1, ipady=3)
                ent.bind("<Return>", lambda e, i=i, j=j: self._move(i + 1, j))
                ent.bind("<Down>", lambda e, i=i, j=j: self._move(i + 1, j))
                ent.bind("<Up>", lambda e, i=i, j=j: self._move(i - 1, j))
                ent.bind("<Control-v>", lambda e, i=i, j=j: self._paste_into(i, j))
                ent.bind("<Control-V>", lambda e, i=i, j=j: self._paste_into(i, j))
                ent.bind("<FocusIn>", lambda e, ent=ent: ent.after_idle(lambda: ent.select_range(0, "end")))
                row_cells.append((var, ent))
            self.cells.append(row_cells)
        # the blue bar that separates the coefficients from the right-hand side
        tk.Frame(self.grid_frame, width=4, background=COLOR_ACCENT).grid(
            row=0, column=cols, rowspan=rows + 1, sticky="ns", padx=4)
        for i, row_cells in enumerate(self.cells):         # watch typing, pasting, everything
            for j, (var, ent) in enumerate(row_cells):
                var.trace_add("write", lambda *a, i=i, j=j: self._on_cell_changed(i, j))
        self._bulk = False

        big = rows > BIG_ROWS or cols > BIG_COLS
        if big != self._was_big:        # change the defaults only when the matrix changes "size class"
            self.var_steps.set(not big)
            self.var_format.set("decimal" if big else "exact")
            self._sync_places_state()
            self._was_big = big
        self._refresh_validation()
        self._mark_stale()

    def fill_grid(self, matrix):
        """Show a matrix of Fractions in the grid."""
        text = [[frac_text(v) for v in row] for row in matrix]
        self.build_grid(len(text), len(text[0]), text)

    def _move(self, i, j):
        if 0 <= i < len(self.cells):
            self.cells[i][j][1].focus_set()
        return "break"

    def _on_cell_changed(self, i, j):
        self.warn_cells.discard((i, j))      # the user touched it, so it no longer "needs a check"
        self._validate_cell(i, j)
        if not self._bulk:
            self._refresh_validation()
            self._mark_stale()

    def _validate_cell(self, i, j, strict=False):
        """Update the colour/error of one cell. Empty cells are only an error when strict."""
        var, ent = self.cells[i][j]
        text = var.get()
        if text.strip() == "" and not strict:
            self.invalid.pop((i, j), None)
        else:
            _, error = parse_cell(text)
            if error:
                self.invalid[(i, j)] = error
            else:
                self.invalid.pop((i, j), None)
        ent.configure(background=self._cell_color(i, j))

    def _cell_name(self, i, j):
        cols = len(self.cells[0])
        return f"R{i + 1}, {'RHS' if j == cols - 1 else f'x{j + 1}'}"

    def _refresh_validation(self):
        if not self.invalid:
            self.lbl_validation.configure(text="")
            return
        (i, j), error = sorted(self.invalid.items())[0]
        more = len(self.invalid) - 1
        text = f"Please fix {len(self.invalid)} cell(s) (shown in red). First one: {self._cell_name(i, j)} {error}"
        if more > 0:
            text += f"  (+{more} more)"
        self.lbl_validation.configure(text=text)

    def collect_matrix(self):
        """Read the whole grid as Fractions. On any problem: mark the cells, explain, return None."""
        self.invalid = {}
        values = []
        for i, row in enumerate(self.cells):
            out = []
            for j, (var, ent) in enumerate(row):
                value, error = parse_cell(var.get())
                if error:
                    self.invalid[(i, j)] = error
                out.append(value)
            values.append(out)
        for i, row in enumerate(self.cells):
            for j, (var, ent) in enumerate(row):
                ent.configure(background=self._cell_color(i, j))
        self._refresh_validation()
        if self.invalid:
            (i, j), error = sorted(self.invalid.items())[0]
            self.cells[i][j][1].focus_set()
            messagebox.showerror(
                "Some cells need attention",
                f"{len(self.invalid)} cell(s) are empty or not valid numbers (shown in red).\n\n"
                f"First problem: {self._cell_name(i, j)} {error}\n\n"
                "Fix the red cells, or use 'Fill blanks with 0' if the empty cells should be zero.")
            return None
        return values

    # ---------- editing helpers ----------
    def on_create_grid(self):
        try:
            rows, cols = int(self.var_rows.get()), int(self.var_cols.get())
        except ValueError:
            messagebox.showerror("Grid size", "Rows and columns must be whole numbers.")
            return
        if cols < 2:
            messagebox.showerror("Grid size", "An augmented matrix needs at least 2 columns "
                                              "(the coefficients and the right-hand side).")
            return
        if not (1 <= rows <= MAX_ROWS and cols <= MAX_COLS):
            messagebox.showerror("Grid size", f"Use 1-{MAX_ROWS} rows and 2-{MAX_COLS} columns.")
            return
        self.set_status(f"Building a {rows} x {cols} grid...")
        self.update_idletasks()
        self.build_grid(rows, cols)
        self.set_status(f"Empty {rows} x {cols} grid created. Type the values, then press Calculate.")

    def on_clear(self):
        self.build_grid(len(self.cells), len(self.cells[0]))
        self.set_status("Grid cleared.")

    def on_fill_blanks(self):
        self._bulk = True
        count = 0
        for i, row in enumerate(self.cells):
            for j, (var, ent) in enumerate(row):
                if var.get().strip() == "":
                    var.set("0")
                    count += 1
                self._validate_cell(i, j)
        self._bulk = False
        self._refresh_validation()
        if count:
            self._mark_stale()
        self.set_status(f"{count} empty cell(s) filled with 0.")

    def load_example(self):
        self.build_grid(3, 4, [["2", "1", "-1", "8"], ["-3", "-1", "2", "-11"], ["-2", "1", "2", "-3"]])
        self.set_status("Example loaded.")

    def _paste_into(self, i, j):
        """Ctrl+V inside a cell: a pasted table fills the grid starting at this cell."""
        try:
            rows = parse_table_text(self.clipboard_get())
        except (tk.TclError, ValueError):
            return None                          # not a table: let Tk paste normally
        if len(rows) == 1 and len(rows[0]) == 1:
            return None
        m, n = len(self.cells), len(self.cells[0])
        clipped = False
        self._bulk = True
        for di, r in enumerate(rows):
            for dj, value in enumerate(r):
                if i + di < m and j + dj < n:
                    self.cells[i + di][j + dj][0].set(value)
                else:
                    clipped = True
        self._bulk = False
        self._refresh_validation()
        self._mark_stale()
        self.set_status("Pasted; values that did not fit in the grid were skipped." if clipped else "Pasted.")
        return "break"

    def on_paste_button(self):
        try:
            rows = parse_table_text(self.clipboard_get())
        except tk.TclError:
            messagebox.showinfo("Paste matrix", "The clipboard is empty.")
            return
        except ValueError as e:
            messagebox.showerror("Paste matrix", f"The clipboard does not contain a matrix: {e}")
            return
        if len(rows[0]) < 2:
            messagebox.showerror("Paste matrix", "An augmented matrix needs at least 2 columns.")
            return
        if len(rows) > MAX_ROWS or len(rows[0]) > MAX_COLS:
            messagebox.showerror("Paste matrix", f"The matrix is too big (maximum {MAX_ROWS} x {MAX_COLS}).")
            return
        self.build_grid(len(rows), len(rows[0]), rows)
        self.set_status(f"Pasted a {len(rows)} x {len(rows[0])} matrix. Cells in red need a fix.")

    def _on_wheel(self, event):
        """Mouse wheel scrolls the grid when the pointer is over it (Shift = sideways)."""
        w = self.winfo_containing(event.x_root, event.y_root)
        while w is not None:
            if w is self.canvas:
                if getattr(event, "num", 0) == 4:
                    step = -1
                elif getattr(event, "num", 0) == 5:
                    step = 1
                else:
                    step = -1 if event.delta > 0 else 1
                if event.state & 0x1:                      # Shift pressed
                    self.canvas.xview_scroll(step, "units")
                else:
                    self.canvas.yview_scroll(step, "units")
                return
            w = w.master

    # =============================================================
    # Reading from an image
    # =============================================================
    def _refresh_ocr_label(self, review=None):
        if review:
            self.lbl_ocr.configure(text=review, foreground="#7a4a00", background="#fff4d6")
        elif self.ocr_ok:
            self.lbl_ocr.configure(text="Image reading is available (EasyOCR and OpenCV were found).",
                                   foreground=COLOR_HINT, background=self.cget("background"))
        else:
            self.lbl_ocr.configure(
                text="Image reading is not available: EasyOCR / OpenCV are not installed. Manual entry works "
                     f"normally. To enable images, run:  {OCR_INSTALL_COMMAND}   (see Help > Reading matrices "
                     "from images).",
                foreground=COLOR_ERROR_TEXT, background="#fdeaea")

    def _hide_review_notice(self):
        self._refresh_ocr_label()

    def show_about(self):
        messagebox.showinfo(
            "What this program does",
            "This window runs the forward-elimination program (project_Dr_shreif.py).\n\n"
            "- It reduces an augmented matrix to Row Echelon Form: a leading 1 in each pivot row and zeros "
            "below each pivot.\n"
            "- It does NOT produce the Reduced Row Echelon Form and does NOT calculate the values of the "
            "variables.\n"
            "- All arithmetic uses exact fractions. 'Rounded decimals' changes only how numbers are shown.\n"
            "- A row with only zero coefficients and a non-zero right-hand side means the system is "
            "inconsistent (no solution); the program reports it.\n"
            "- Matrices read from images can contain mistakes: always check them before calculating.")

    def show_ocr_help(self):
        text = ("Reading a matrix from an image needs two optional packages: EasyOCR and OpenCV.\n\n"
                f"Install them from a terminal:\n\n    {OCR_INSTALL_COMMAND}\n\n"
                "The first time you read an image, EasyOCR also downloads its language model, which needs an "
                "internet connection and can take a while. Image reading is never perfect: always compare the "
                "matrix shown in the grid with the picture and correct any wrong value before calculating.\n\n"
                "Copy the install command to the clipboard?")
        if messagebox.askyesno("Reading matrices from images", text):
            self.clipboard_clear()
            self.clipboard_append(OCR_INSTALL_COMMAND)
            self.set_status("Install command copied to the clipboard.")

    def on_choose_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Choose the image that contains the matrix",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")])
        if path:
            self.image_path = path
            name = os.path.basename(path)
            self.lbl_image.configure(text=name if len(name) <= 30 else name[:27] + "...")
            self.set_status(f"Image chosen: {path}  -  now press 'Read matrix from image'.")

    def on_open_image(self):
        if not self.image_path:
            messagebox.showinfo("Open image", "Choose an image first.")
            return
        try:
            os.startfile(self.image_path)                    # Windows
        except AttributeError:
            webbrowser.open("file://" + os.path.abspath(self.image_path))

    def on_read_image(self):
        if self.busy:
            return
        if not self.image_path or not os.path.isfile(self.image_path):
            messagebox.showinfo("Read matrix from image", "Choose an image first (Choose image...).")
            return
        if not ocr_available():
            self.ocr_ok = False
            self._refresh_ocr_label()
            self.show_ocr_help()
            return
        rows_text, cols_text = self.var_hint_r.get().strip(), self.var_hint_c.get().strip()
        size = None
        if rows_text or cols_text:
            if rows_text.isdigit() and cols_text.isdigit() and int(rows_text) > 0 and int(cols_text) > 0:
                size = (int(rows_text), int(cols_text))
            else:
                messagebox.showerror("Size", "Type both numbers (rows and columns) as whole numbers, "
                                             "or leave both empty.")
                return
        self.core.THICKEN_INK = self.var_thicken.get()
        self.pending_warnings = set()
        self._reset_text("messages", f"Reading image: {self.image_path}\n")
        self.notebook.select(2)
        self._set_busy(True, "Reading the image... (the first time, EasyOCR loads its model; big images are "
                             "read in pieces and can take a few minutes)")
        threading.Thread(target=self._ocr_worker, args=(self.image_path, size), daemon=True).start()

    def _ocr_worker(self, path, size):
        try:
            with contextlib.redirect_stdout(QueueWriter(self.q, "messages")):
                matrix = self.core.image_to_matrix(path, size)
            self.q.put(("ocr_done", matrix))
        except ImportError:
            self.q.put(("ocr_missing", None))
        except Exception as e:                                # reading can fail in many ways
            self.q.put(("error", f"Could not read the matrix from the image:\n{e}"))

    def _on_ocr_done(self, matrix):
        self._set_busy(False)
        if len(matrix[0]) < 2:
            messagebox.showerror("Read matrix from image", "An augmented matrix needs at least 2 columns, "
                                                           "but the image gave only one.")
            return
        self.fill_grid(matrix)
        self.warn_cells = {(r - 1, c - 1) for r, c in self.pending_warnings
                           if r - 1 < len(matrix) and c - 1 < len(matrix[0])}
        for i, j in self.warn_cells:
            self.cells[i][j][1].configure(background=COLOR_CHECK)
        note = (f"Review needed: a {len(matrix)} x {len(matrix[0])} matrix was read from the image. Compare "
                "every value with the picture and correct mistakes in the grid before pressing Calculate.")
        if self.warn_cells:
            note += f" {len(self.warn_cells)} yellow cell(s): the reader was unsure about them."
        self._refresh_ocr_label(review=note)
        self.set_status(f"Image read: {len(matrix)} x {len(matrix[0])}. Please review the grid.")

    # =============================================================
    # Calculating
    # =============================================================
    def _sync_places_state(self):
        self.spin_places.configure(state="normal" if self.var_format.get() == "decimal" else "disabled")

    def on_calculate(self):
        if self.busy:
            return
        matrix = self.collect_matrix()
        if matrix is None:
            return
        m, n = len(matrix), len(matrix[0])
        if n < 2:
            messagebox.showerror("Matrix", "An augmented matrix needs at least 2 columns.")
            return
        show_steps = self.var_steps.get()
        if show_steps and (m > BIG_ROWS or n > BIG_COLS):
            show_steps = messagebox.askyesno(
                "Large matrix",
                f"The matrix is {m} x {n}. Showing every row operation can produce a huge output and be slow.\n\n"
                "Show every step anyway?\n(Choose 'No' to hide the steps; the final matrix is still shown.)")
            self.var_steps.set(show_steps)
        decimals = None
        if self.var_format.get() == "decimal":
            text = self.var_places.get().strip()
            if not (text.isdigit() and 0 <= int(text) <= 15):
                messagebox.showerror("Decimal places", "Decimal places must be a whole number from 0 to 15.")
                self.spin_places.focus_set()
                return
            decimals = int(text)

        self.result = None
        self.stale = False
        self._reset_text("result", "")
        self._reset_text("steps", "" if show_steps else
                         "The row-operation steps are hidden (the option 'Show row-operation steps' is off).\n"
                         "Any 'column skipped' notices still appear here.\n\n")
        self._set_result_banner("Calculating...", "#f0f0f0", "#222222")
        self.notebook.select(0)
        self._set_busy(True, f"Running forward elimination on a {m} x {n} matrix...")
        threading.Thread(target=self._calc_worker, args=(matrix, show_steps, decimals), daemon=True).start()

    def _calc_worker(self, matrix, show_steps, decimals):
        """Runs the program's own functions; everything they print is captured."""
        core = self.core
        try:
            core.SHOW_STEPS, core.DECIMALS = show_steps, decimals
            m, n = len(matrix), len(matrix[0])
            original_text = capture(core.print_matrix, matrix)
            self.q.put(("result", f"Original augmented matrix ({m} x {n}):\n{original_text}"))

            steps_log = []
            with contextlib.redirect_stdout(QueueWriter(self.q, "steps", keep=steps_log)):
                core.forward_elimination(matrix)          # changes the matrix in place

            final_text = capture(core.print_matrix, matrix)
            consistent = core.check_consistency(matrix)
            notes = ("Note: this is the Row Echelon Form from FORWARD elimination only. It is not the Reduced "
                     "Row Echelon Form, and the values of the variables are not calculated.\n")
            if decimals is not None:
                notes += (f"Note: numbers are rounded to {decimals} decimal places for display only; "
                          "the calculation itself was exact.\n")
            if not consistent:
                notes += ("Inconsistent system: a row has only zero coefficients but a non-zero right-hand side "
                          "(0 = b), so there is no solution.\n")
            self.q.put(("result", "========== Final augmented matrix (Row Echelon Form) ==========\n"
                        f"{final_text}{notes}"))
            self.q.put(("calc_done", {
                "shape": (m, n), "decimals": decimals, "consistent": consistent,
                "final_text": final_text, "notes": notes,
                "steps_text": "".join(steps_log) if show_steps else None,
            }))
        except Exception as e:
            self.q.put(("error", f"Something went wrong during the calculation:\n{e}"))

    def _on_calc_done(self, data):
        self._set_busy(False)
        self.result = data
        self.stale = False
        m, n = data["shape"]
        if data["consistent"]:
            self._set_result_banner(
                f"Forward elimination finished ({m} x {n}). No row of the form [0 ... 0 | b] with b not zero "
                "was found. Reminder: this is Row Echelon Form only; variable values are not calculated.",
                "#e3f4e8", COLOR_OK_TEXT)
        else:
            self._set_result_banner(
                "INCONSISTENT SYSTEM: at least one row has only zero coefficients but a non-zero right-hand "
                "side (0 = b). The system has no solution.", "#fdeaea", COLOR_ERROR_TEXT)
        self.set_status("Done.")

    def _mark_stale(self):
        """The grid no longer matches the result on screen: say so instead of leaving a green banner."""
        if self.result is not None and not self.stale:
            self.stale = True
            self._set_result_banner("The matrix was changed after this result was calculated, so the result below "
                                    "is out of date. Press Calculate to update it.", "#fff4d6", "#7a4a00")

    def _set_result_banner(self, text, background, foreground):
        self.lbl_result.configure(text=text, background=background, foreground=foreground)

    # =============================================================
    # Results: showing, copying, saving
    # =============================================================
    def _reset_text(self, key, content):
        self.buffers[key].clear()
        widget = self.texts[key]
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.configure(state="disabled")
        if content:
            self.buffers[key].append(content)

    def clear_results(self):
        if self.busy:
            return
        for key in self.texts:
            self._reset_text(key, "")
        self.result = None
        self.stale = False
        self._set_result_banner("Press Calculate to run forward elimination.", "#f0f0f0", "#222222")
        self.set_status("Results cleared.")

    def _flush_buffers(self):
        """Move queued text into the text boxes in small pieces so the window never freezes."""
        budget = 120_000
        for key, buffer in self.buffers.items():
            if not buffer or budget <= 0:
                continue
            widget = self.texts[key]
            widget.configure(state="normal")
            while buffer and budget > 0:
                piece = buffer.popleft()
                if len(piece) > budget:
                    buffer.appendleft(piece[budget:])
                    piece = piece[:budget]
                widget.insert("end", piece)
                budget -= len(piece)
            widget.configure(state="disabled")
            if key != "result":
                widget.see("end")

    def on_copy(self):
        if not self.result:
            messagebox.showinfo("Copy final matrix", "Calculate first.")
            return
        self.clipboard_clear()
        self.clipboard_append(self.result["final_text"])
        self.set_status("Final matrix copied to the clipboard.")

    def on_save(self):
        if self.busy:
            return
        if not self.result:
            messagebox.showinfo("Save results", "Calculate first, then save the result.")
            return
        have_steps = self.result["steps_text"] is not None
        dialog = tk.Toplevel(self)
        dialog.title("Save results")
        dialog.transient(self)
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=16)
        frame.pack()
        ttk.Label(frame, text="The final matrix will be saved to a text file.").pack(anchor="w")
        include = tk.BooleanVar(value=have_steps)
        check = ttk.Checkbutton(
            frame, variable=include,
            text="Also include the row-operation steps" if have_steps
            else "Row-operation steps are not available (they were hidden)")
        check.pack(anchor="w", pady=(10, 14))
        if not have_steps:
            check.state(["disabled"])
        chosen = []
        buttons = ttk.Frame(frame)
        buttons.pack(anchor="e")
        ttk.Button(buttons, text="Save...", style="Accent.TButton",
                   command=lambda: (chosen.append(include.get()), dialog.destroy())).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)
        dialog.bind("<Escape>", lambda e: dialog.destroy())
        dialog.grab_set()
        self.wait_window(dialog)
        if not chosen:
            return
        path = filedialog.asksaveasfilename(
            title="Save results", defaultextension=".txt", initialfile="result.txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        r = self.result
        m, n = r["shape"]
        fmt_line = ("exact fractions" if r["decimals"] is None else
                    f"rounded decimals, {r['decimals']} places (display only; the calculation was exact)")
        lines = ["Forward Elimination (Gaussian Elimination) - Row Echelon Form",
                 f"Matrix size: {m} x {n} (the last column is the right-hand side)",
                 f"Numbers shown as: {fmt_line}", "",
                 "Final augmented matrix:", r["final_text"].rstrip("\n"), "", r["notes"].rstrip("\n")]
        if chosen[0] and have_steps:
            lines += ["", "Row-operation steps:", "", r["steps_text"].rstrip("\n")]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as e:
            messagebox.showerror("Save results", f"The file could not be saved:\n{e}")
            return
        self.set_status(f"Saved: {path}")

    # =============================================================
    # Status, busy state and the message loop
    # =============================================================
    def set_status(self, text):
        self.var_status.set(text)

    def _set_busy(self, flag, message=None):
        self.busy = flag
        state = "disabled" if flag else "normal"
        self.btn_calc.configure(state=state)
        self.btn_read.configure(state=state)
        if flag:
            self.progress.start(12)
        else:
            self.progress.stop()
        if message:
            self.set_status(message)

    def _poll(self):
        try:
            for _ in range(400):
                kind, payload = self.q.get_nowait()
                if kind in ("result", "steps"):
                    self.buffers[kind].append(payload)
                elif kind == "messages":
                    self.buffers["messages"].append(payload)
                    for r, c in re.findall(r"row (\d+), column (\d+) had two candidates", payload):
                        self.pending_warnings.add((int(r), int(c)))
                    last = payload.strip().splitlines()[-1] if payload.strip() else ""
                    if last:
                        self.set_status(last[:160])
                elif kind == "ocr_done":
                    self._on_ocr_done(payload)
                elif kind == "ocr_missing":
                    self._set_busy(False, "Image reading is not available.")
                    self.ocr_ok = False
                    self._refresh_ocr_label()
                    self.show_ocr_help()
                elif kind == "calc_done":
                    self._on_calc_done(payload)
                elif kind == "error":
                    self._set_busy(False, "Error.")
                    self._set_result_banner("The last action failed. See the message.", "#fdeaea", COLOR_ERROR_TEXT)
                    messagebox.showerror("Error", payload)
        except queue.Empty:
            pass
        self._flush_buffers()
        self.after(30, self._poll)


# ---------------------------------------------------------------
# Start-up
# ---------------------------------------------------------------
def main():
    if sys.platform.startswith("win"):
        try:                                           # sharper text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    folder = os.path.dirname(os.path.abspath(__file__))
    path = find_core(folder)
    if path is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            APP_TITLE,
            "project_Dr_shreif.py was not found next to this file.\n\n"
            "Please locate it in the next window (or keep both files in the same folder).")
        path = filedialog.askopenfilename(title="Locate project_Dr_shreif.py",
                                          filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        root.destroy()
        if not path:
            return
    try:
        core = load_core(path)
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(APP_TITLE, f"The program file could not be loaded:\n{path}\n\n{e}")
        return
    App(core).mainloop()


if __name__ == "__main__":
    main()
