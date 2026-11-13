"""Builds the TIDE page images (theme pairs) from the poster, the paper and the schematic scripts."""
import os, shutil, sys
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
RAW = "/home/claude/work/assets_raw"
OUT = "/home/claude/work/site/assets/img/tide"
MEDIA = RAW + "/pptx_v25/ppt/media"

def save_webp(im, path, w=None, q=88, lossless=False):
    if w and im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    im.save(path, "WEBP", quality=q, method=6, lossless=lossless)
    print(f"{os.path.basename(path):34s} {im.size}  {os.path.getsize(path)/1024:.0f} KB")
    return im

# 1 schematics (SVG, both themes, straight from the authors' scripts)
for src, dst in [("tide_figure_v4_light", "tide-step-light"), ("tide_figure_v4_dark", "tide-step-dark"),
                 ("ctm_figure_v4_light", "ctm-step-light"), ("ctm_figure_v4_dark", "ctm-step-dark")]:
    shutil.copy(f"{RAW}/schem/{src}.svg", f"{OUT}/{dst}.svg")
    print(dst + ".svg", os.path.getsize(f"{OUT}/{dst}.svg") // 1024, "KB")

# 2 attention figure: dark = poster asset; light = same pixels with the ink swapped back for white paper
att = Image.open(MEDIA + "/image11.png").convert("RGBA")
a = np.asarray(att).copy()
rgb, al = a[..., :3].astype(np.int16), a[..., 3]
n, lab, stats, _ = cv2.connectedComponentsWithStats((al >= 250).astype(np.uint8), connectivity=8)
protect = np.zeros(al.shape, bool)
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if w >= 300 and h >= 300:
        protect |= lab == i
print("attention: protected photo/tile components:", int(sum(1 for i in range(1, n) if stats[i][2] >= 300 and stats[i][3] >= 300)))
light = a.copy()
ink = (~protect) & (al > 0)
R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
is_red = ink & (R > G + 50) & (R > B + 50)
is_green = ink & (G > R + 40) & (G > B + 30) & ~is_red
is_dark = ink & ~is_red & ~is_green
for m, new in [(is_red, (196, 24, 26)), (is_green, (20, 128, 58)), (is_dark, (20, 26, 34))]:
    light[m, 0], light[m, 1], light[m, 2] = new
light[ink & (al < 20), 3] = 0                       # drop re-encoding fringe (garbled low-alpha edge pixels)
print("attention ink px: red", int(is_red.sum()), "green", int(is_green.sum()), "dark", int(is_dark.sum()))
def near(c, tol=40):
    return (np.abs(rgb - np.array(c, np.int16)).sum(axis=2) <= tol)
# frames (inside protected regions): poster colors back to the paper's colorblind-safe pair
for old, new in [((255, 96, 96), (215, 25, 28)), ((60, 220, 100), (26, 150, 65))]:
    m = protect & near(old, 6)
    light[m, 0], light[m, 1], light[m, 2] = new
save_webp(att, OUT + "/attention-dark.webp", w=2200, q=86)
save_webp(Image.fromarray(light, "RGBA"), OUT + "/attention-light.webp", w=2200, q=86)

# 3 ImageNet-C heatmap: light from the paper's Figure 3, dark from the poster (same numbers)
p = Image.open(RAW + "/paper/img/p-003.png").convert("RGB"); m = Image.open(RAW + "/paper/img/p-004.png").convert("L")
p.putalpha(m); bg = Image.new("RGBA", p.size, (255, 255, 255, 255)); bg.alpha_composite(p)
save_webp(bg.convert("RGB"), OUT + "/imagenet-c-light.webp", w=2400, q=90)
save_webp(Image.open(MEDIA + "/image7.jpg").convert("RGB"), OUT + "/imagenet-c-dark.webp", w=2400, q=90)

# 4 CIFAR-10-C heatmap: dark from the poster; light = luminance inverted outside the heatmap panels and color bars
d = Image.open(MEDIA + "/image8.jpg").convert("RGB")
save_webp(d, OUT + "/cifar10-c-dark.webp", w=2400, q=90)
x = np.asarray(d).astype(np.int16)
sat = x.max(axis=2) - x.min(axis=2)
H, W = sat.shape
col = (sat > 60).mean(axis=0)                       # share of clearly colored pixels per column
runs, on = [], None
for i, v in enumerate(col > 0.25):
    if v and on is None: on = i
    if not v and on is not None:
        if i - on > 25: runs.append((on, i))
        on = None
if on is not None: runs.append((on, W))
print("cifar10-c: colored column runs:", runs)
merged = []
for r in runs:                                      # cell columns -> panels (gaps between cells are a few px)
    if merged and r[0] - merged[-1][1] <= 12: merged[-1][1] = r[1]
    else: merged.append(list(r))
panels = [m for m in merged if m[1] - m[0] > 400]; bars = [m for m in merged if m[1] - m[0] <= 400]
print("cifar10-c: panels", panels, "bars", bars)
pw = panels[0][1] - panels[0][0]; gap = bars[0][0] - panels[0][1]
if len(panels) == 2 and len(bars) == 3:             # the pale diverging panel: place it from its color bar
    panels.append([bars[2][0] - gap - pw, bars[2][0] - gap])
rowp = (sat[:, panels[0][0]:panels[0][1]] > 60).mean(axis=1)
ys = np.where(rowp > 0.2)[0]; y0, y1 = int(ys.min()), int(ys.max()) + 1
rects = [[p0, y0, p1, y1] for p0, p1 in panels]
by0, by1 = H, 0
for b0, b1 in bars:                                 # all color bars share one extent; the hot ramp's pale end is unsaturated
    rp = (sat[:, b0:b1] > 60).mean(axis=1); yb = np.where(rp > 0.2)[0]
    by0, by1 = min(by0, int(yb.min())), max(by1, int(yb.max()) + 1)
for b0, b1 in bars:
    rects.append([b0, by0, b1, by1])
print("cifar10-c: protected rects:", rects)
keep = np.zeros((H, W), bool)
PAD = 6
for x0, yy0, x1, yy1 in rects:
    keep[max(0, yy0 - PAD):yy1 + PAD, max(0, x0 - PAD):x1 + PAD] = True
out = x.copy()
inv = ~keep
out[inv] = 255 - out[inv]
save_webp(Image.fromarray(out.astype(np.uint8), "RGB"), OUT + "/cifar10-c-light.webp", w=2400, q=90)
np.save("/tmp/cifar_rects.npy", np.array(rects))

# 5 poster preview + PDF
pv = Image.open("/mnt/user-data/uploads/tide-rebutal/poster/TIDE_NeurIPS2026_poster_black_v2_5_preview.png").convert("RGB")
save_webp(pv, OUT + "/poster.webp", w=2400, q=86)
shutil.copy("/mnt/user-data/uploads/tide-rebutal/poster/TIDE_NeurIPS2026_poster_black_v2_5_proof.pdf", "/home/claude/work/site/assets/files/TIDE_NeurIPS2026_poster.pdf")
print("poster pdf", os.path.getsize("/home/claude/work/site/assets/files/TIDE_NeurIPS2026_poster.pdf") // 1024, "KB")
