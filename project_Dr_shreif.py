"""
Forward Elimination (Gaussian Elimination) on an augmented matrix.
Result: Row Echelon Form (leading 1 in each pivot, zeros below each pivot).
Uses Fraction so all values stay exact (1/3 instead of 0.333...).

Input can be typed by hand OR read from an image of the matrix (EasyOCR),
followed by a confirmation step where the user can fix any wrong number.
"""

import os
import re
from decimal import Context, Decimal
from fractions import Fraction

DEBUG_OCR = False  # set to True to print what EasyOCR saw before any processing
THICKEN_INK = True  # image preprocessing; if the OCR output looks worse, try False
DECIMALS = None   # None = show exact fractions; a number = round to that many decimal places (display only)
SHOW_STEPS = True  # when False, row operations are done silently (needed for big matrices)


# ---------------------------------------------------------------
# Input functions
# ---------------------------------------------------------------
def read_positive_int(prompt):
    """Keep asking until the user enters a positive integer."""
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            print("  Please enter a number greater than 0.")
        except ValueError:
            print("  Invalid input, please enter an integer.")


def read_matrix(m, n):
    """Read an m x n matrix row by row. Accepts integers, decimals, and fractions like 1/2."""
    print(f"\nEnter the matrix row by row ({n} numbers per row, separated by spaces).")
    print("The last column is the right-hand side of the equations.\n")
    matrix = []
    for i in range(m):
        while True:
            parts = input(f"Row {i + 1}: ").split()
            if len(parts) != n:
                print(f"  You must enter exactly {n} numbers.")
                continue
            try:
                row = [Fraction(p) for p in parts]  # Fraction("1/2"), Fraction("2.5") both work
            except (ValueError, ZeroDivisionError):
                print("  Invalid number found, please try again.")
                continue
            matrix.append(row)
            break
    return matrix


# ---------------------------------------------------------------
# Image input (OCR)
# ---------------------------------------------------------------
_reader = None  # EasyOCR reader is created once (loading the model is slow)


def get_reader():
    """Create the EasyOCR reader on first use."""
    global _reader
    if _reader is None:
        import easyocr  # imported here so manual input works even without easyocr installed
        print("Loading OCR model (first time can take a while)...")
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def clean_token(text):
    """Fix common OCR mistakes: fancy minus signs, look-alike letters, stray characters."""
    for dash in ("−", "–", "—", "‒", "_", "~"):
        text = text.replace(dash, "-")
    table = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1",
                           "|": "1", "S": "5", "B": "8"})
    text = text.translate(table)
    # keep only characters that can appear in a number
    text = "".join(ch for ch in text if ch in "0123456789-/. ")
    return text.strip()


def normalize_token(tok):
    """Return a clean numeric string, or None if the token cannot be a number.
    '--2' -> '-2', '4-' -> '4', a lone '-' or '.' -> None."""
    tok = re.sub(r"^-+", "-", tok)      # several minus signs at the start become one
    tok = tok.rstrip("-./").lstrip("./")  # stray symbols at the ends
    if tok in ("", "-"):
        return None
    try:
        Fraction(tok)
    except (ValueError, ZeroDivisionError):
        return None
    return tok


def glue_minus_signs(items):
    """items = list of (y, x, h, token).
    OCR sometimes returns the minus sign as a separate box. For every lone '-':
      * if the closest number on its right already starts with '-', the minus was
        detected twice, so the lone one is dropped;
      * otherwise it is glued to that number ('-' + '3' -> '-3').
    A lone '-' with no number next to it is dropped (it is noise)."""
    items = sorted(items, key=lambda it: it[1])  # left to right: '-' comes before its number
    used, out = set(), []
    for i, (y, x, h, tok) in enumerate(items):
        if i in used:
            continue
        if tok == "-":
            best = None
            for j, (y2, x2, h2, tok2) in enumerate(items):
                if j == i or j in used or tok2 == "-":
                    continue
                same_row = abs(y2 - y) < max(h, h2) * 0.6
                close = 0 < x2 - x < max(h, h2) * 2.5
                if same_row and close and (best is None or x2 < items[best][1]):
                    best = j
            if best is not None and not items[best][3].startswith("-"):
                y2, x2, h2, tok2 = items[best]
                out.append((y2, x2, h2, "-" + tok2))
                used.add(best)
            continue  # glued, duplicate, or noise: the lone '-' itself is never kept
        out.append((y, x, h, tok))
    return out


def sign_mask(img_bgr):
    """Black-and-white mask of the ink (1 = ink, 0 = paper), used to find minus signs."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() < 127:          # dark background -> invert so ink is black on white
        bw = 255 - bw
    return (bw == 0).astype(np.uint8)


def repair_signs(boxes, ink):
    """Fix the signs of numbers using the picture itself instead of trusting the OCR.

    EasyOCR often reads '-2' as '22' (minus turned into a copy of the digit) or loses
    the minus completely. A minus sign is easy to find in the image: a small, flat,
    wide blob. So:
      1) find all such blobs ("dashes") and all digit-sized blobs ("glyphs");
      2) give every dash to the nearest number on its right  -> that number is negative;
      3) if the OCR text has two equal digits ('22') but the box holds only ONE glyph,
         the second digit was really the minus -> '-2'.
    boxes = list of [x0, y0, x1, y1, text]. Returns the same list with fixed text."""
    import cv2
    import numpy as np

    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps = [tuple(int(v) for v in stats[i][:4]) for i in range(1, n) if stats[i][4] >= 4]
    tall = [h for x, y, w, h in comps if h >= 8 and h > 1.2 * w]
    if not tall:
        return boxes
    ref_h = float(np.median(tall))          # typical digit height in this image

    dashes, glyphs = [], []
    for x, y, w, h in comps:
        if h <= 0.35 * ref_h and w >= 1.5 * h and 0.3 * ref_h <= w <= 1.3 * ref_h:
            dashes.append((x + w / 2, y + h / 2))
        elif h >= 0.5 * ref_h:
            glyphs.append((x + w / 2, y + h / 2))

    has_digit = [any(ch.isdigit() for ch in b[4]) for b in boxes]

    negative = set()
    for dx, dy in dashes:
        best = None
        for k, (x0, y0, x1, y1, text) in enumerate(boxes):
            if not has_digit[k]:
                continue
            height, cx = y1 - y0, (x0 + x1) / 2
            near = 0 < cx - dx < 1.2 * max(height, ref_h)
            same_row = y0 - 0.25 * height <= dy <= y1 + 0.25 * height
            if near and same_row and (best is None or cx - dx < best[0]):
                best = (cx - dx, k)
        if best:
            negative.add(best[1])

    fixed = []
    for k, (x0, y0, x1, y1, text) in enumerate(boxes):
        tok = text.strip()
        digits = tok.lstrip("-")
        if len(tok.split()) == 1 and digits.isdigit():
            neg = tok.startswith("-") or k in negative
            inside = sum(1 for gx, gy in glyphs if x0 <= gx <= x1 and y0 <= gy <= y1)
            if len(digits) == 2 and digits[0] == digits[1] and inside == 1:
                digits, neg = digits[0], True      # '22' with one glyph = '-2'
            text = ("-" if neg else "") + digits
        fixed.append([x0, y0, x1, y1, text])
    return fixed


TILE_SIZE = 1700  # biggest piece (in pixels) given to EasyOCR in one go; it shrinks anything larger


def blank_cuts(profile, max_len):
    """Choose where to cut a long image so every piece is at most max_len pixels.
    profile = amount of ink in every pixel row (or column). Cuts are placed in the
    middle of an empty strip, so a number is never sliced in half when possible."""
    total = len(profile)
    cuts = [0]
    while total - cuts[-1] > max_len:
        start = cuts[-1]
        lo = start + max_len // 2
        pos = None
        i = start + max_len
        while i > lo:
            if profile[i] == 0:                      # found an empty strip, find where it begins
                j = i
                while j > lo and profile[j - 1] == 0:
                    j -= 1
                pos = (i + j) // 2
                break
            i -= 1
        cuts.append(pos if pos is not None else start + max_len)
    cuts.append(total)
    return cuts


def read_in_tiles(ocr_img, ink):
    """Run EasyOCR on the image. Small images are read in one go; big ones are cut into
    tiles (so the digits keep their real size) and the boxes are moved back to full-image
    coordinates. Returns EasyOCR-style results: [(bbox, text, confidence), ...]."""
    import cv2

    height, width = ocr_img.shape[:2]
    if max(height, width) <= TILE_SIZE:
        tiles = [(0, 0, width, height)]
    else:
        xc = blank_cuts(ink.sum(axis=0), TILE_SIZE)
        yc = blank_cuts(ink.sum(axis=1), TILE_SIZE)
        tiles = [(xc[a], yc[b], xc[a + 1], yc[b + 1])
                 for b in range(len(yc) - 1) for a in range(len(xc) - 1)]
        tiles = [t for t in tiles if ink[t[1]:t[3], t[0]:t[2]].sum() > 0]  # skip empty pieces
        print(f"Big image: reading it in {len(tiles)} pieces (this can take a few minutes)...")

    margin = 20
    reader = get_reader()
    results = []
    for k, (x0, y0, x1, y1) in enumerate(tiles, start=1):
        if len(tiles) > 1:
            print(f"  piece {k}/{len(tiles)}")
        crop = ocr_img[y0:y1, x0:x1]
        white = 255 if crop.ndim == 2 else (255, 255, 255)
        crop = cv2.copyMakeBorder(crop, margin, margin, margin, margin,
                                  cv2.BORDER_CONSTANT, value=white)
        # No allowlist (it made '-3' become '33'); we clean the text ourselves in clean_token().
        found = reader.readtext(
            crop,
            low_text=0.3,        # lower = detects weaker/smaller text (like 0 or -)
            text_threshold=0.5,
            link_threshold=0.9,  # higher = less merging of neighbouring numbers
            width_ths=0.1,       # don't merge boxes that are far apart horizontally
            mag_ratio=1.5,
        )
        for bbox, text, conf in found:
            moved = [[pt[0] + x0 - margin, pt[1] + y0 - margin] for pt in bbox]
            results.append((moved, text, conf))
    return results


def ocr_items(path):
    """Run EasyOCR on the image and return a list of (y_center, x_center, height, token)."""
    import cv2
    import numpy as np

    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not open the image.")

    # Upscale small images + add white border: small characters like '0' and '-' are easier to detect
    scale = 2 if max(img.shape[:2]) < 1000 else 1   # only enlarge small images
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    ink = sign_mask(img)   # used later to find minus signs (same coordinates as the OCR image)

    ocr_img = img
    if THICKEN_INK:
        # Black & white + thicker ink: thin '-' signs become solid
        ocr_img = cv2.erode((1 - ink).astype(np.uint8) * 255, np.ones((2, 2), np.uint8), iterations=1)

    results = read_in_tiles(ocr_img, ink)
    if DEBUG_OCR:
        for bbox, text, conf in results:
            print("OCR saw:", repr(text), "conf=%.2f" % conf,
                  "x=%d y=%d w=%d h=%d" % (bbox[0][0], bbox[0][1],
                                         bbox[1][0] - bbox[0][0], bbox[2][1] - bbox[0][1]))
    if not results:
        raise ValueError("No numbers were found in the image.")

    boxes = []
    for bbox, text, conf in results:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        boxes.append([min(xs), min(ys), max(xs), max(ys), clean_token(text)])
    boxes = repair_signs(boxes, ink)

    items = []
    for x0, y0, x1, y1, text in boxes:
        y = (y0 + y1) / 2
        h = y1 - y0
        tokens = text.split()
        for i, tok in enumerate(tokens):  # if a box holds several numbers, spread them out
            xc = x0 + (x1 - x0) * (i + 0.5) / len(tokens)
            items.append((y, xc, h, tok))
    return items


def axis_centers(values, k=None, gap=None):
    """Find the centre positions of the rows (y values) or columns (x values).
    - k given: 1-D k-means with exactly k groups (most reliable).
    - otherwise: values closer than `gap` belong to the same group."""
    values = sorted(values)
    if k:
        if k > len(values):
            raise ValueError(f"Only {len(values)} numbers were found, fewer than {k} rows/columns.")
        lo, hi = values[0], values[-1]
        if k == 1:
            return [sum(values) / len(values)]
        centers = [lo + (hi - lo) * i / (k - 1) for i in range(k)]  # evenly spread start
        for _ in range(50):
            groups = [[] for _ in range(k)]
            for v in values:
                groups[min(range(k), key=lambda c: abs(centers[c] - v))].append(v)
            new = [sum(g) / len(g) if g else centers[i] for i, g in enumerate(groups)]
            if new == centers:
                break
            centers = new
        return sorted(centers)

    groups, current = [], [values[0]]
    for v in values[1:]:
        if v - current[-1] <= gap:
            current.append(v)
        else:
            groups.append(current)
            current = [v]
    groups.append(current)
    return [sum(g) / len(g) for g in groups]


def items_to_matrix(items, size=None):
    """Turn OCR items (y, x, h, token) into a matrix of Fractions.
    size = (rows, columns) if the user knows it, otherwise None (auto-detect)."""
    rows_k, cols_k = size if size else (None, None)

    items = glue_minus_signs(items)
    items = [(y, x, h, normalize_token(t)) for y, x, h, t in items]
    items = [it for it in items if it[3] is not None]   # drop junk like '^' or a lone '-'
    if not items:
        raise ValueError("No numbers were found in the image.")

    avg_h = sum(it[2] for it in items) / len(items)
    row_c = axis_centers([it[0] for it in items], rows_k, avg_h * 0.6)
    col_c = axis_centers([it[1] for it in items], cols_k, avg_h * 0.8)

    # Every number goes to the nearest (row, column) cell
    cells = {}
    for y, x, h, tok in items:
        r = min(range(len(row_c)), key=lambda k: abs(row_c[k] - y))
        c = min(range(len(col_c)), key=lambda k: abs(col_c[k] - x))
        dist = abs(row_c[r] - y) / avg_h + abs(col_c[c] - x) / avg_h
        if (r, c) in cells:
            old_dist, old_tok = cells[(r, c)]
            kept = tok if dist < old_dist else old_tok
            print(f"  Warning: row {r + 1}, column {c + 1} had two candidates "
                  f"({old_tok} and {tok}); kept {kept}. Please check it.")
            if dist < old_dist:
                cells[(r, c)] = (dist, tok)
        else:
            cells[(r, c)] = (dist, tok)

    # Empty cells become 0 (a missed '0' is the most common OCR miss)
    return [[Fraction(cells[(r, c)][1]) if (r, c) in cells else Fraction(0)
             for c in range(len(col_c))] for r in range(len(row_c))]


def image_to_matrix(path, size=None):
    """Read a matrix from an image. Missing cells (usually missed zeros) are
    filled with 0 using row/column positions; the user confirms the result afterwards."""
    return items_to_matrix(ocr_items(path), size)


def confirm_matrix(matrix):
    """Show the matrix and let the user fix wrong numbers.
    Returns True when the user accepts, False if they want to cancel."""
    while True:
        print("\nThe program read this matrix:")
        print_matrix(matrix)
        answer = input("Is it correct? (y = yes / e = edit a number / c = cancel): ").strip().lower()
        if answer == "y":
            return True
        if answer == "c":
            return False
        if answer == "e":
            parts = input("Enter: row column new_value   (example: 2 3 -5): ").split()
            if len(parts) != 3:
                print("  Please enter exactly 3 values.")
                continue
            try:
                r, c = int(parts[0]), int(parts[1])
                value = Fraction(parts[2])
                if not (1 <= r <= len(matrix) and 1 <= c <= len(matrix[0])):
                    print("  Row or column is out of range.")
                    continue
            except (ValueError, ZeroDivisionError):
                print("  Invalid input.")
                continue
            matrix[r - 1][c - 1] = value
        else:
            print("  Please type y, e, or c.")


def read_matrix_from_image():
    """Ask for an image path, read the matrix, and confirm it. Returns None on failure/cancel."""
    path = input("\nImage path: ").strip().strip('"').strip("'")
    if not os.path.isfile(path):
        print("  File not found.")
        return None
    size_text = input("Matrix size as 'rows columns' (example: 10 10). "
                      "Press Enter if you don't know: ").split()
    size = None
    if len(size_text) == 2 and all(t.isdigit() and int(t) > 0 for t in size_text):
        size = (int(size_text[0]), int(size_text[1]))
    try:
        matrix = image_to_matrix(path, size)
    except ImportError:
        print("  EasyOCR / OpenCV is not installed. Run: pip install easyocr opencv-python")
        return None
    except ValueError as e:
        print(f"  Could not read the matrix: {e}")
        return None
    if len(matrix[0]) < 2:
        print("  An augmented matrix needs at least 2 columns.")
        return None
    return matrix if confirm_matrix(matrix) else None


def read_matrix_manually():
    """Ask for the size, then read the matrix by hand."""
    m = read_positive_int("Number of rows (m): ")
    n = read_positive_int("Number of columns (n, including the RHS column): ")
    while n < 2:
        print("  An augmented matrix needs at least 2 columns.")
        n = read_positive_int("Number of columns (n, including the RHS column): ")
    return read_matrix(m, n)


def get_matrix():
    """Let the user choose between manual input and an image."""
    while True:
        print("\nHow do you want to enter the matrix?")
        print("  1) Type it manually")
        print("  2) Read it from an image")
        choice = input("Choice (1/2): ").strip()
        if choice == "1":
            return read_matrix_manually()
        if choice == "2":
            matrix = read_matrix_from_image()
            if matrix is not None:
                return matrix
            print("Let's try again.")
        else:
            print("  Please type 1 or 2.")


# ---------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------
def fmt(x):
    """Format a Fraction for display.
    Exact mode: '3' instead of '3/1', otherwise 'a/b'.
    Decimal mode (DECIMALS set): whole numbers stay as they are, other values are rounded
    to DECIMALS places, and huge values use scientific notation (1.235e+45).
    The calculation itself is always exact; only the display is rounded."""
    if DECIMALS is None:
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"

    if abs(x) >= 10 ** 9:                      # too big to show digit by digit
        value = Context(prec=12).divide(Decimal(x.numerator), Decimal(x.denominator))
        return format(value, ".3e")
    if x.denominator == 1:
        return str(x.numerator)

    rounded = round(x * 10 ** DECIMALS)        # integer, e.g. 0.33333 -> 3333 when DECIMALS = 4
    if rounded == 0:
        return "≈0"                            # not exactly zero, but too small to show
    digits = str(abs(rounded)).rjust(DECIMALS + 1, "0")
    text = digits[:-DECIMALS] + "." + digits[-DECIMALS:] if DECIMALS else digits
    text = text.rstrip("0").rstrip(".") if DECIMALS else text
    return ("-" if rounded < 0 else "") + text


def print_matrix(matrix):
    """Print the augmented matrix with aligned columns and a bar before the last column."""
    n = len(matrix[0])
    strings = [[fmt(v) for v in row] for row in matrix]
    widths = [max(len(strings[i][j]) for i in range(len(matrix))) for j in range(n)]
    for row in strings:
        left = "  ".join(row[j].rjust(widths[j]) for j in range(n - 1))
        right = row[n - 1].rjust(widths[n - 1])
        print(f"  [ {left} | {right} ]")
    print()


# ---------------------------------------------------------------
# Row operations (each one prints what it did)
# ---------------------------------------------------------------
def step(message, matrix):
    """Print one row operation and the matrix after it (only if SHOW_STEPS is on)."""
    if SHOW_STEPS:
        print(message)
        print_matrix(matrix)


def swap_rows(matrix, r1, r2):
    """Row Swap: Ri <-> Rj"""
    matrix[r1], matrix[r2] = matrix[r2], matrix[r1]
    step(f"R{r1 + 1} ↔ R{r2 + 1}", matrix)


def scale_row(matrix, r, pivot):
    """Row Scaling: divide the row by the pivot so the pivot becomes 1."""
    matrix[r] = [v / pivot for v in matrix[r]]
    if SHOW_STEPS:
        step(f"R{r + 1} → ({fmt(1 / pivot)})·R{r + 1}", matrix)


def replace_row(matrix, target, source, factor):
    """Row Replacement: R_target = R_target - factor * R_source"""
    matrix[target] = [t - factor * s for t, s in zip(matrix[target], matrix[source])]
    if SHOW_STEPS:
        sign = "-" if factor > 0 else "+"
        shown = fmt(abs(factor))
        if abs(factor).denominator != 1:
            shown = f"({shown})"
        step(f"R{target + 1} → R{target + 1} {sign} {shown}·R{source + 1}", matrix)


# ---------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------
def forward_elimination(matrix):
    """Transform the augmented matrix into Row Echelon Form, printing every step."""
    m = len(matrix)
    n = len(matrix[0])
    pivot_row = 0

    # Only coefficient columns are pivot candidates (the last column is the RHS)
    for col in range(n - 1):
        if pivot_row >= m:
            break

        # 1) Choose the pivot row (only rows from pivot_row downward can be used).
        #    Prefer a row that already has 1 in this column (no scaling needed), then a row
        #    with -1 (scaling by -1 only flips signs, so no fractions appear). Otherwise take
        #    the first row with a non-zero value.
        found = None
        for wanted in (1, -1):
            for r in range(pivot_row, m):
                if matrix[r][col] == wanted:
                    found = r
                    break
            if found is not None:
                break
        if found is None:
            for r in range(pivot_row, m):
                if matrix[r][col] != 0:
                    found = r
                    break

        if found is None:
            print(f"Column {col + 1} has only zeros below the current row → skipped.\n")
            continue

        # 2) Row Swap if the chosen row is not already in the pivot position
        if found != pivot_row:
            swap_rows(matrix, pivot_row, found)

        # 3) Row Scaling to make the pivot equal 1
        pivot = matrix[pivot_row][col]
        if pivot != 1:
            scale_row(matrix, pivot_row, pivot)

        # 4) Row Replacement to make everything below the pivot equal 0
        for r in range(pivot_row + 1, m):
            factor = matrix[r][col]
            if factor != 0:
                replace_row(matrix, r, pivot_row, factor)

        pivot_row += 1  # move to the next row and the next columns


def check_consistency(matrix):
    """A row like [0 0 0 | b] with b != 0 means the system has no solution."""
    for row in matrix:
        if all(v == 0 for v in row[:-1]) and row[-1] != 0:
            return False
    return True


def save_matrix(matrix, filename):
    """Write the matrix to a text file (one row per line, '|' before the last column)."""
    with open(filename, "w", encoding="utf-8") as f:
        for row in matrix:
            left = " ".join(fmt(v) for v in row[:-1])
            f.write(f"{left} | {fmt(row[-1])}\n")
    return os.path.abspath(filename)


def choose_number_format(big):
    """Let the user pick exact fractions or rounded decimals for everything that is displayed."""
    global DECIMALS
    default = "2" if big else "1"
    print("\nHow should the numbers be shown?")
    print("  1) Exact fractions (example: 1/3)")
    print("  2) Rounded decimals (example: 0.3333) - easier to read for big matrices")
    choice = input(f"Choice (1/2, default {default}): ").strip() or default
    if choice == "2":
        places = input("How many decimal places? (default 4): ").strip()
        DECIMALS = int(places) if places.isdigit() and 0 <= int(places) <= 15 else 4


def main():
    global SHOW_STEPS
    print("=== Forward Elimination (Gaussian Elimination) ===")
    matrix = get_matrix()

    m, n = len(matrix), len(matrix[0])
    big = m > 10 or n > 11
    choose_number_format(big)
    if big:
        print(f"\nThe matrix is {m} x {n}. Printing every step would produce a huge output.")
        SHOW_STEPS = input("Show every step anyway? (y/n, default n): ").strip().lower() == "y"

    if not big or SHOW_STEPS:
        print("\nOriginal augmented matrix:")
        print_matrix(matrix)
        print("---------- Steps ----------\n")
    else:
        print("\nWorking... (steps are hidden, this can take a little while)")

    forward_elimination(matrix)

    if big and not SHOW_STEPS:
        where = save_matrix(matrix, "result.txt")
        print(f"Done. The final matrix (Row Echelon Form) was saved to:\n  {where}")
    else:
        print("========== Final augmented matrix (Row Echelon Form) ==========")
        print_matrix(matrix)
    if DECIMALS is not None:
        print(f"Note: numbers are rounded to {DECIMALS} decimal places for display only; "
              "the calculation itself was exact.")
    if not check_consistency(matrix):
        print("Note: the system is inconsistent (a row of zeros equals a non-zero value) → no solution.")


if __name__ == "__main__":
    main()
    #C:\Users\DUBAI STORE\OneDrive\Pictures\
