# Scan-to-PDF (No OCR) — Text Alignment + Post-Trim (Multi-Folder)

This tool converts **folders of photos** (documents shot by phone/camera) into **clean PDFs**, one PDF per folder. It focuses on **paper detection** and **text alignment** (skew/shear) **without OCR**, with optional **post-trim + zoom-to-fill** to remove margins and make the paper fill the page.

## Key Features
- **Multi-folder batching**: If your input root contains `N` subfolders, you get `N` PDFs (each named `<subfolder>.pdf`).
- **Robust paper detection** using a **white-ish mask** (HSV+LAB) + contour scoring (area, brightness, rectangularity, border penalty).
- **Text-based rectification** (no perspective warp):
  - Skew (rotation) via **row projection variance maximization**.
  - **Horizontal shear** to straighten **vertical** lines (x′ = x + k·y).
  - **Vertical shear** to straighten **horizontal** lines (y′ = y + k·x).
- **Post-trim (default ON)**:
  - Re-detect paper **after** rectification to tighten margins.
  - Optional **zoom-to-fill** so paper slightly overfills the frame.
- **Safe PDF pages**: embed each page as a full-bleed JPEG with pixel→point sizing.
- **Sorting modes**: natural (default), lexicographic, file mtime, EXIF DateTimeOriginal.
- **Downscale** by longest side to control output size.
- **Debug artifacts** and optional per-image CSV.

---

## Installation

```bash
# Python 3.9+ recommended
pip install opencv-python pillow numpy reportlab
# (Optional) If you're on conda:
# conda install -c conda-forge opencv pillow numpy reportlab
```

> No OCR libraries are required.

---

## Folder Layout

```
input_root/
  FolderA/
    img_001.jpg
    img_002.jpg
  FolderB/
    1.png
    2.png
```

After running the script:

```
output_pdfs/
  FolderA.pdf
  FolderB.pdf
```

---

## Usage

```bash
# Basic: create one PDF per subfolder under input_root
python main.py -i input_root -O output_pdfs

# With debug images and a different sorting mode
python main.py -i input_root -O output_pdfs --debug --debug-dir debug_out --sort mtime

# Disable post-trim and zoom-to-fill (defaults are ON)
python main.py -i input_root -O output_pdfs --no-post-trim --no-post-trim-zoom

# Lower DPI (e.g., smaller PDFs) and control downscaling / JPEG quality
python main.py -i input_root -O output_pdfs --dpi 150 --max-side 2500 --jpeg-quality 85
```

---

## CLI Arguments

| Flag | Type | Default | Description |
|---|---|---:|---|
| `-i, --input` | str | (required) | **Input root** containing subfolders (each becomes a PDF). |
| `-O, --output-dir` | str | (required) | Output directory where PDFs are written. |
| `--ext` | str | `jpg,jpeg,png` | Comma-separated extensions to include. |
| `--sort` | choice | `natural` | Sort images per folder by `natural`, `lex`, `mtime`, or `exif`. |
| `--dpi` | int | `200` | Pixel→point conversion for page sizing. |
| `--max-side` | int | `3000` | Downscale longest side to this many pixels (0 disables). |
| `--jpeg-quality` | int | `90` | Quality for embedding pages as JPEG. |
| `--debug` | flag | `False` | Save debug overlays and intermediate images. |
| `--debug-dir` | str | `None` | Directory for debug images. |
| `--csv` | str | `None` | CSV path per folder (append your own naming if needed). |
| `--no-post-trim` | flag | `False` | **Disable** post-trim (default is ON). |
| `--no-post-trim-zoom` | flag | `False` | **Disable** zoom-to-fill (default is ON). |
| `--post-trim-pad` | int | `6` | Keep this many pixels around paper after trimming. |
| `--post-trim-zoom-over` | int | `4` | Overshoot (px) when zoom-to-fill (0–8 recommended). |

---

## How It Works (Short)
1. **Per image**: detect paper via HSV+LAB mask and contour scoring → **tight crop** (no perspective warp).
2. Build a text-emphasized binary:
   - Search for skew angle by maximizing the **variance of horizontal projections**.
   - Apply **horizontal shear** so vertical lines become vertical.
   - Apply **vertical shear** so horizontal lines become horizontal.
3. Force **portrait** orientation (rotate +90° if needed).
4. **Post-trim** (default): re-detect paper on the rectified image, crop margins, optionally **zoom-to-fill**.
5. Embed each final image as a full-bleed **PDF page**.

---

## Tuning Tips
- **Dim lighting / warm paper**: lower `LAB_L_MIN` to `150–165`, raise `AB_DEV_MAX` to `25–35`.
- **Small/remote documents**: lower `AREA_MIN_RATIO` to `0.12–0.18`.
- **Over-aggressive shears**: raise `SHEAR_MIN_ABS_DEG` to `0.3–0.5`.
- **Excess margins left**: ensure post-trim is ON and use `--post-trim-zoom --post-trim-zoom-over 6`.

---

## Troubleshooting
- **Nothing detected / black page**: Check `--ext`. Try `--sort lex` if ordering is odd.
- **Contours grabbed background edges**: increase `BORDER_PENALTY_UNIT` or tighten `AREA_MIN_RATIO`.
- **Paper still not full**: Keep `--post-trim-zoom` (default ON) and increase `--post-trim-zoom-over` to `6–8`.
- **Skew not corrected**: widen `SKEW_SEARCH_DEG` (e.g., 12–15) and/or use `SKEW_SEARCH_STEP=0.2`.
