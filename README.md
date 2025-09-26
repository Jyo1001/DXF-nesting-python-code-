# DXF Nesting Script

This repository contains a single-file DXF nesting utility that packs 2D profiles on to rectangular sheets.  The script was designed to run without external dependencies and now includes a small CLI wrapper so it can be executed directly from Linux/macOS shells as well as the original Windows workflow.

## Usage

```bash
python nest_dxf_qty_optimized_shuffle.py --folder "For waterjet cutting" --pixels-per-unit 6 --tries 4
```

### Notable options

- `--folder` – Source directory that holds DXF files and optional quantity text files.
- `--sheet WIDTH HEIGHT` – Sheet size in drawing units (defaults to 48" × 96").
- `--pixels-per-unit` – Bitmap resolution; higher values improve accuracy at the cost of runtime.
- `--tries` – Number of shuffle trials to explore alternate part orders.
- `--seed` – Fix the random seed for repeatable runs.
- `--workers` – Number of worker processes for bitmap evaluation (defaults to auto detection).
- `--allow-mirror/--no-mirror` – Control whether mirrored placements are explored.
- `--allow-hole-nesting/--forbid-hole-nesting` – Toggle part-in-hole placement.
- `--nest-mode` – Choose between the bitmap searcher and the simpler shelf fallback.

When no folder is provided the script automatically looks for the bundled `For waterjet cutting` sample directory that ships with the repository.  This makes it easy to verify the algorithm without editing constants in the file.

## Output

The script writes `nested.dxf` and a `nest_report.txt` summary to the working folder on completion.  The report documents the run parameters, sheet usage, and any skipped parts so the resulting layout can be reproduced.
