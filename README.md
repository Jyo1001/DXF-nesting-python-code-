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

## GPU acceleration

- When PyTorch with CUDA support is available, the bitmap packer automatically switches to the fastest GPU device it can access. Use `--device cuda:0` (or another device string) to explicitly select which GPU should be used, or `--device cpu` to fall back to the CPU implementation.
- The `nest_report.txt` file records whether a CUDA GPU was engaged, so you can verify that an NVIDIA device handled the workload. This is the preferred path for reducing runtime; DirectX is not required.
- On Linux and macOS the script now mirrors its Windows progress window with console updates so you can observe progress directly in the terminal. The final report also enumerates the ordering heuristics that were evaluated.

## Deepnest-inspired heuristics

After reviewing [Jack000/Deepnest](https://github.com/Jack000/Deepnest), the bitmap packer now seeds its search with additional ordering heuristics that mimic Deepnest's fitness priorities (part area, bounding box fill ratio, and perimeter). These extra seeds improve the chance of finding a dense layout within a limited number of shuffle tries while preserving the lightweight single-file implementation.
