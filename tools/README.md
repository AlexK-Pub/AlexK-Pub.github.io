# tools

Scripts that produced generated assets. None of them run when GitHub builds the site.

- `tide_figure.py`, `ctm_figure.py`: the schematics on the TIDE page.
  `python tide_figure.py out.svg` writes the light version, `--dark` the dark one. Needs numpy, matplotlib and Carlito Bold.
- `build_assets_tide.py`: makes the light/dark pairs of the attention figure and the corruption heatmaps from the poster and the paper.
- `build_assets_misc.py`: profile photo and the figures of the earlier papers (from their arXiv sources).
- `darken_figures.py`: makes a dark-theme version of a figure drawn on white (black background, white text and axes, hues kept).
  `python3 darken_figures.py in.png out-dark.webp`; add `x0,y0,x1,y1` boxes for photographs that must stay unchanged. Needs numpy, OpenCV and Pillow.
- `darken_earlier.py`: the dark versions of the plots and drawings on the power-line, dataset and smoke pages, rebuilt from the papers' source files and passed through `darken_figures.py`. Point clouds drawn with a color map keep their drawn colors (`data=True`).
- `render_equations.js`: pre-renders the Wilson-Cowan equations with KaTeX into `_includes/tide-equations.html`.

The two `build_assets_*.py` scripts and `darken_earlier.py` hold absolute paths from the machine they were written on; treat them as a record of what was done, and adjust the paths before re-running.
