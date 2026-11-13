"""Dark-theme versions of figures drawn on white paper: black background, white text and axes, colors kept.

    python3 darken_figures.py SRC.png OUT.webp [x0,y0,x1,y1 ...]

Each pixel is read as ink on white paper. Its coverage is how far it is from white; the ink is what is left
after the paper is removed. The same coverage is then laid on black with the ink re-colored for a dark page:
black ink becomes white, dark colors are lifted until they read on black, and hue is kept. Light fills
(a pale pink box, a pale blue sphere) therefore turn into dark fills of the same hue, so labels on them stay legible.
Boxes given after the output path are copied unchanged (photographs inside a diagram).
"""
import sys
import cv2
import numpy as np
from PIL import Image

def darken(im, keep=(), data=False, pale=(0.30, 0.42), stroke_gain=1.7):
    """data=True: for color-mapped data (point clouds colored by height or intensity). Colored pixels are
    treated as opaque marks and keep the color they were drawn with (only colors too dark to read on black are
    lifted), so the color map still reads the same. Black and gray ink (axes, labels) is handled as usual.
    pale: how bright fills of 20% and 60% coverage become; raise it for sketches drawn in pale gray.
    stroke_gain: how strongly thin lines are restored."""
    a8 = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
    cover = 1.0 - a8.min(axis=2)                                  # 0 on white paper, 1 on full ink
    cover = np.clip((cover - 0.045) / (1.0 - 0.045), 0.0, 1.0)     # drop JPEG ringing around strokes
    c = np.maximum(cover, 1e-4)[..., None]
    ink = np.clip((a8 - (1.0 - c)) / c, 0.0, 1.0)                  # un-blend from white
    mx, mn = ink.max(axis=2), ink.min(axis=2)
    chroma = mx - mn                                               # 0 for black/gray ink, 1 for a pure hue
    hue = np.where(mx[..., None] > 1e-3, ink / np.maximum(mx, 1e-3)[..., None], 1.0)   # the ink at full brightness
    w = np.clip(chroma * 1.6, 0.0, 1.0)[..., None]                 # how much of the hue to keep
    ink2 = (1.0 - w) * 1.0 + w * hue                               # black -> white, dark red -> light red, red -> red
    lum = (ink2 * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=2)
    lift = np.clip((0.34 - lum) / np.maximum(1.0 - lum, 1e-3), 0.0, 1.0)[..., None]    # blues and deep reds: add white
    ink2 = ink2 + lift * (1.0 - ink2)
    # fills and thin strokes need opposite treatment. A morphological opening removes anything thinner than
    # 5 px, so the difference to the original marks plot lines, hairlines and small markers.
    k = np.ones((5, 5), np.uint8)
    opened = cv2.dilate(cv2.erode(cover, k), k)
    thin = np.clip((cover - opened) / np.maximum(cover, 1e-3), 0.0, 1.0)
    # fills: pale tints stay visible on black, mid tones (a yellow box) stay dark enough for white labels
    fill = np.interp(cover, [0.0, 0.2, 0.6, 1.0], [0.0, pale[0], pale[1], 1.0]).astype(np.float32)
    stroke = np.clip(cover * stroke_gain, 0.0, 1.0) ** 0.8         # thin lines lose coverage to blur: restore it
    cov2 = (1.0 - thin) * fill + thin * stroke
    out = cov2[..., None] * ink2
    if data:
        # a colored mark is opaque: anything clearly off-white gets its drawn color back; only the soft edge
        # of a mark (nearly white) is un-blended, so marks keep their shape on black
        raw = 1.0 - a8.min(axis=2)
        al = np.clip(raw / 0.30, 0.0, 1.0)[..., None]
        col = np.clip((a8 - (1.0 - al)) / np.maximum(al, 1e-4), 0.0, 1.0)
        lum_c = (col * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=2)
        lift_c = np.clip((0.22 - lum_c) / np.maximum(1.0 - lum_c, 1e-3), 0.0, 1.0)[..., None]
        col = col + lift_c * (1.0 - col)
        out = (1.0 - w) * out + w * (al * col)
    out = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    src = np.asarray(im.convert("RGB"))
    for x0, y0, x1, y1 in keep:
        out[y0:y1, x0:x1] = src[y0:y1, x0:x1]
    return Image.fromarray(out, "RGB")

if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    keep = [tuple(int(v) for v in s.split(",")) for s in sys.argv[3:]]
    darken(Image.open(src), keep).save(dst, "WEBP", quality=92, method=6)
