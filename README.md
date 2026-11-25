# Forward Elimination (Gaussian Elimination)

A program that reduces an **augmented matrix** to **Row Echelon Form** using forward
(Gaussian) elimination, with **exact fraction arithmetic** at every step.

It has two interfaces that share the same core:

| File | What it is | How to run |
|------|------------|------------|
| `project_Dr_shreif.py` | Command-line program | `python project_Dr_shreif.py` |
| `project_Dr_shreif_ui.py` | Desktop window (Tkinter / ttk) | `python project_Dr_shreif_ui.py` |

The GUI does **not** modify the command-line program: it simply imports and calls its
functions (`image_to_matrix`, `forward_elimination`, `check_consistency`, `print_matrix`,
`fmt`) and its settings (`SHOW_STEPS`, `DECIMALS`, `THICKEN_INK`), so the mathematics and
the OCR flow are identical in both.

> Keep both `.py` files in the same folder. The GUI looks for `project_Dr_shreif.py`
> next to itself, and will ask you to locate it if it is missing.

---

## What it does

- Reduces an augmented matrix to **Row Echelon Form**:
  - a leading `1` in each pivot row, and
  - zeros below each pivot.
- Uses Python's `fractions.Fraction`, so every calculation is **exact**
  (it prints `1/3` instead of `0.333...`).
- Chooses a clever pivot order: it prefers a row that already has `1`, then `-1`,
  then any non-zero entry — which often keeps the numbers clean and avoids fractions.
- Detects an **inconsistent system**: a row of the form `[0 ... 0 | b]` with `b ≠ 0`
  means there is no solution, and the program reports it.
- Lets you enter the matrix **by hand**, **by pasting a table**, or **by reading it
  from an image** (OCR).

## What it does NOT do

- It does **not** produce the **Reduced Row Echelon Form (RREF)**.
- It does **not** compute the actual values of the variables.
- "Rounded decimals" changes only how numbers are **displayed**; the calculation itself
  is always exact.

---

## Requirements

- **Python 3.8+** (the bundled cache was built with Python 3.13).
- The core and the GUI need **only the Python standard library**.
- **Optional — only for reading matrices from images (OCR):**

  ```bash
  pip install easyocr opencv-python
  ```

  or, using the provided file:

  ```bash
  pip install -r requirements.txt
  ```

  Notes on the optional packages:

  - `easyocr` also downloads a language model the first time you read an image, which
    needs an internet connection and can take a while.
  - `easyocr` pulls in **PyTorch**, which is a large download.
  - `opencv-python` (`cv2`) and `numpy` are used for image preprocessing and to fix
    minus signs. `numpy` is also installed automatically as a dependency of both
    packages, but it is listed explicitly because the code imports it directly.

If you only want to type matrices by hand, you can skip the optional packages entirely.

---

## Usage — command line

Run:

```bash
python project_Dr_shreif.py
```

1. Choose `1` to type the matrix, or `2` to read it from an image.
2. If typing: enter the number of rows `m`, then the number of columns `n`
   (including the right-hand-side column), then each row separated by spaces.
   Integers (`-3`), decimals (`2.5`) and fractions (`1/2`) are all accepted.
3. If reading from an image: give the image path, and optionally the matrix size as
   `rows columns` (press Enter if you don't know it). The program shows what it read
   and lets you correct any number before continuing.
4. Choose how to show numbers: exact fractions, or rounded decimals.
5. The program prints each row operation and the matrix after it, then the final
   **Row Echelon Form**.

For big matrices (more than 10 rows or 11 columns) the steps are hidden by default and
the final matrix is saved to `result.txt` in the current folder.

### Configuration (top of `project_Dr_shreif.py`)

| Setting | Default | Meaning |
|---------|---------|---------|
| `DEBUG_OCR` | `False` | Print what EasyOCR saw before any processing. |
| `THICKEN_INK` | `True` | Image preprocessing that thickens thin ink; set to `False` if OCR output looks worse. |
| `DECIMALS` | `None` | `None` = exact fractions; a number = round the display to that many places. |
| `SHOW_STEPS` | `True` | `False` does the row operations silently (useful for big matrices). |

---

## Usage — desktop window

Run:

```bash
python project_Dr_shreif_ui.py
```

- **Enter values** in the grid. The last (blue-separated, shaded) column is the
  right-hand side. Accepted values: integers, decimals with a dot, and fractions.
- **Paste a matrix** with the *Paste matrix* button, or press **Ctrl+V** inside a cell
  to paste a table starting at that cell. Rows are separated by new lines; cells by
  spaces, tabs, semicolons or `|`.
- **Read from an image** with *Choose image...* → *Read matrix from image*. The grid is
  filled and cells the OCR was unsure about are highlighted yellow — always check them.
- **Calculate** with the *Calculate* button, **Ctrl+Enter** or **F5**.
- Results appear in three tabs: *Original and final matrix*, *Row-operation steps*, and
  *Messages*. Save them to a text file with **Ctrl+S**.

### GUI limits

- Maximum matrix size: **60 rows × 61 columns** (60 unknowns + the right-hand side).
- Matrices larger than 10 rows or 11 columns are treated as "big": the steps default to
  hidden and the display defaults to rounded decimals.

---

## File summary

- `project_Dr_shreif.py` — the core: matrix input (manual + image/OCR), exact forward
  elimination, step printing, consistency check, and output.
- `project_Dr_shreif_ui.py` — the Tkinter desktop interface that wraps the core.
- `README.md` — this file.
- `requirements.txt` — the optional packages needed only for image (OCR) input.
