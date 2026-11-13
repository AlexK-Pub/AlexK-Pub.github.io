"""CTM architecture figure: one internal step, drawn on the grid of the TIDE figure (Options G and H of the canvas).

Draws CTM (Darlow et al., 2025) with the same blocks, line weight, fonts and colors as tide_figure.py, so the two
figures can sit side by side. CTM's steps 1 to 10 follow the numbering of CTM's own Figure 2. CTM has a single
population of neurons with no E/I type, so its neurons are drawn in neutral colors; dashed boxes mark the two TIDE
blocks that CTM does not have (lateral inhibition and memory).

    python ctm_figure.py ctm_option_g.svg            Option G: light theme, the palette of Option E
    python ctm_figure.py ctm_option_h.svg --dark     Option H: black poster theme with the hot ramp of Option F

Requirements and fonts are the same as for tide_figure.py (numpy, matplotlib, Carlito Bold in ./fonts).
"""
import math
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "stix"
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path

# ============================== fonts
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIRS = [os.environ.get("TIDE_FONT_DIR", ""), os.path.join(HERE, "fonts"),
             "/usr/share/fonts/truetype/crosextra", "/usr/share/fonts/truetype/carlito", "/usr/share/fonts/carlito",
             os.path.expanduser("~/.local/share/fonts"), os.path.expanduser("~/.fonts"),
             os.path.expanduser("~/Library/Fonts"), "/Library/Fonts",
             os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"),
             os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")]


def find_font(preferred, fallback):
    """First file in FONT_DIRS whose name matches `preferred` (case-insensitive), else one matching `fallback`."""
    for names, is_fallback in ((preferred, False), (fallback, True)):
        wanted = {n.lower() for n in names}
        for d in FONT_DIRS:
            if not d or not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                if name.lower() in wanted:
                    if is_fallback:
                        warnings.warn(f"Carlito not found; using {name}. Calibri has the same metrics but slightly "
                                      "different letter shapes, so the drawing will differ a little from the canvas.")
                    return os.path.join(d, name)
    sys.exit("Carlito Bold not found. Put Carlito-Bold.ttf (Google Fonts) in a 'fonts' folder next to this script, "
             "or set TIDE_FONT_DIR to the folder that holds it.")


SANSB = FontProperties(fname=find_font(["Carlito-Bold.ttf"], ["calibrib.ttf", "Calibri-Bold.ttf", "Calibri Bold.ttf"]))
MATHP = FontProperties(family="STIXGeneral")
FONTMAP = {"bold": SANSB, "math": MATHP}

# ============================== drawing primitives
INK = "#1b1b1b"
E_C, I_C = "#3f63ad", "#b23b3b"          # excitatory / inhibitory strokes (paper's Figure 1 hues)
LI_C, SY_C, MEM_C, HRF_C = "#9a4a2c", "#6657c2", "#5b8a2e", "#3f8a69"
ATTN_F, OUT_F, YEL = "#f2a65a", "#b9a8ef", "#f6d24a"

rng = np.random.default_rng(11)          # fixed seed: cell shading and hatching come out the same on every run
defs, back, mid, front = [], [], [], []  # paint order: fans and areas, shapes, text


def f(v):
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def d_of(pts, closed=False):
    s = "M" + " L".join(f"{f(x)} {f(y)}" for x, y in pts)
    return s + ("Z" if closed else "")


def densify(pts, step=5.0):
    pts = np.asarray(pts, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def trim_end(pts, L):
    pts = np.asarray(pts, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    tot = cum[-1]
    keep = cum < tot - L
    q = pts[keep]
    i = keep.sum()
    t = (tot - L - cum[i - 1]) / max(seg[i - 1], 1e-9)
    return np.vstack([q, pts[i - 1] + (pts[i] - pts[i - 1]) * t])


GRADS = {}


def grad(c0, c1, ang):
    key = (c0, c1, int(ang))
    if key not in GRADS:
        gid = f"g{len(GRADS)}"
        a = math.radians(ang)
        x1, y1 = 0.5 - 0.5 * math.cos(a), 0.5 - 0.5 * math.sin(a)
        x2, y2 = 0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a)
        defs.append(f'<linearGradient id="{gid}" x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}">'
                    f'<stop offset="0" stop-color="{c0}"/><stop offset="1" stop-color="{c1}"/></linearGradient>')
        GRADS[key] = gid
    return GRADS[key]


# cell shading families for the light theme: (light end, choices for the dark end)
FAM = {
    "gray": ("#fbfbfa", ["#b4b4b0", "#9a9a96", "#c9c9c5", "#868682"]),
    "E": ("#f5f8fe", ["#93b0e0", "#7596cf", "#adc3e9", "#6386c4"]),
    "I": ("#fef5f4", ["#e2a39d", "#d0827b", "#ebbab5", "#c26c65"]),
    "a": ("#fff7ec", ["#f3b26b", "#eea056"]),
    "z": ("#f6f3ff", ["#a593e3", "#8f7bd9", "#b9abea"]),
    "zI": ("#fbf2f6", ["#d58aa6", "#c9779a"]),
    "zB": ("#f2f4ff", ["#8f9fe3", "#7d8fd9"]),
    "m": ("#f4f9ec", ["#9cc26d", "#86b257", "#b0cf8a"]),
}


def hatch(x, y, s, op=0.35):
    """Light diagonal pencil hatching inside a cell."""
    for c in np.linspace(s * 0.35, s * 0.9, 4):
        p0, p1 = (x + c, y + s), (x + s, y + c)
        if rng.random() < 0.85:
            mid.append(f'<path d="M{f(p0[0])} {f(p0[1] - 1)} L{f(p1[0] - 1)} {f(p1[1])}" stroke="#4a4a48" '
                       f'stroke-width="0.8" stroke-linecap="round" opacity="{op}"/>')


def fan(src, dst, color, op=0.3, w=0.9):
    """All-to-all connection lines between two cell columns."""
    for a in src:
        for b in dst:
            back.append(f'<line x1="{f(a[0])}" y1="{f(a[1])}" x2="{f(b[0])}" y2="{f(b[1])}" stroke="{color}" '
                        f'stroke-width="{w}" opacity="{op}"/>')


def trace(x0, x1, yc, amp, seed, color=INK, wmin=1.4, wmax=3.8, n=240, env=None):
    """A wavy activity trace drawn as a filled ribbon of varying width."""
    r = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    y = sum(r.uniform(0.3, 1.0) * np.sin(2 * np.pi * (r.uniform(2, 9) * t + r.uniform())) for _ in range(4))
    y = y / np.abs(y).max() * amp
    if env is not None:
        y = y * env(t)
    p = np.stack([x0 + (x1 - x0) * t, yc - y], 1)
    wv = wmin + (wmax - wmin) * (0.5 + 0.5 * np.sin(2 * np.pi * (r.uniform(0.6, 1.6) * t + r.uniform())))
    d = np.gradient(p, axis=0)
    nn = np.stack([-d[:, 1], d[:, 0]], 1)
    nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-9
    poly = np.vstack([p + nn * wv[:, None] / 2, (p - nn * wv[:, None] / 2)[::-1]])
    mid.append(f'<path d="{d_of(poly, True)}" fill="{color}"/>')


def tp_d(tp, dx):
    """SVG path data for a matplotlib TextPath, flipped to SVG's downward y axis."""
    parts = []
    for verts, code in tp.iter_segments(curves=True, simplify=False):
        v = verts.reshape(-1, 2)
        if code == Path.MOVETO:
            parts.append(f"M{f(v[0, 0] + dx)} {f(-v[0, 1])}")
        elif code == Path.LINETO:
            parts.append(f"L{f(v[0, 0] + dx)} {f(-v[0, 1])}")
        elif code == Path.CURVE3:
            parts.append(f"Q{f(v[0, 0] + dx)} {f(-v[0, 1])} {f(v[1, 0] + dx)} {f(-v[1, 1])}")
        elif code == Path.CURVE4:
            parts.append(f"C{f(v[0, 0] + dx)} {f(-v[0, 1])} {f(v[1, 0] + dx)} {f(-v[1, 1])} {f(v[2, 0] + dx)} {f(-v[2, 1])}")
        elif code == Path.CLOSEPOLY:
            parts.append("Z")
    return "".join(parts)


TEXTS = []                # v3: every label's box in figure pixels, for the overlap checker (figcheck.py)


def _text(s, x, y, size, font, color, anchor="start", rot=0, op=1.0):
    """Text as outlines. Math runs get a thin same-color outline so they match the weight of the bold labels."""
    tp = TextPath((0, 0), s, size=size, prop=FONTMAP[font])
    e = tp.get_extents()
    dx = {"start": -e.x0, "middle": -(e.x0 + e.x1) / 2, "end": -e.x1}[anchor]
    tr = f"translate({f(x)} {f(y)})" + (f" rotate({rot})" if rot else "")
    sw = {"math": 0.35}.get(font, 0)
    st = f' stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"' if sw else ""
    TEXTS.append(dict(s=s, size=size, font=font, x0=x + e.x0 + dx, x1=x + e.x1 + dx, y0=y - e.y1, y1=y - e.y0))
    front.append(f'<path data-k="t" data-i="{len(TEXTS) - 1}" d="{tp_d(tp, dx)}" fill="{color}"{st} transform="{tr}" opacity="{op}"/>')
    return e.width


def rich(runs, x, y, anchor="start", gap=5):
    """Lay out (text, font, size, color) runs on one baseline."""
    ws = [TextPath((0, 0), r[0], size=r[2], prop=FONTMAP[r[1]]).get_extents().width for r in runs]
    tot = sum(ws) + gap * (len(runs) - 1)
    cx = x - (tot / 2 if anchor == "middle" else 0)
    for (s, font, size, color), w in zip(runs, ws):
        _text(s, cx, y, size, font, color)
        cx += w + gap

# ============================== themes
DARK = "--dark" in sys.argv
W, H = 1800, 1140
LW = 2.6                  # one weight for every border, outline, connector and arrow
S = 28

LIGHT_T = dict(
    bg="#ffffff", ink=INK, grey="#2f2e2b", line=INK, dash="#2f2e2b", neutral="#7d7c77", syn_panel="#5f78b8",
    E=E_C, I=I_C, LI=LI_C, SY_panel=SY_C, dec_line=SY_C, MEM=MEM_C, gate_slash=MEM_C, HRF=HRF_C,
    attn_fan="#e8902a", attn_text="#b8661a", attn_path="#d98322", kernE="#cddaf2", kernI="#f2d3d1",
    mlp_fill="#ffffff", phi_fill="#f4f4f2", pool_fill="#d07a62", data_fill="#f3f1ea", data_mtn="#b9c7a3",
    data_sun="#f2c55a", hrf_out="#cfeadc", hrf_in="#2f7d5c", attn_ell=ATTN_F, out_ell=OUT_F, ell_line=INK,
    dec_fill="#e3def8", mat_low_fill="#ffffff", mat_low_stroke="#cfcfcb", mat_stroke="#77756f",
    mat_EE="#d9ddf6", mat_EI="#e7d8f3", mat_II="#f3d8de", sel_EE="#4a5fc4", sel_EI="#8a4fc4", sel_II="#c44f6a",
    bars="#9d8be0", needle=I_C, readout=MEM_C, badge_fill=YEL, badge_line=INK, badge_text=INK,
    cell_line=INK, sil_fill="#ffffff", sil_hatch="#9a9a96", sil_line=INK, trace=INK, fam=FAM)

# Poster palette, read from TIDE_NeurIPS2026_poster_black.pptx: background #000000, text #FFFFFF, red lines and red
# text #E8383A, red header bars and tags #C81010, dark cards #1E1E1E, secondary gray #8C969B.
P_BLACK, P_WHITE, P_RED, P_RED_FILL, P_CARD, P_GRAY = "#000000", "#ffffff", "#e8383a", "#c81010", "#1e1e1e", "#8c969b"
# Hot ramp anchored on the poster red (black -> dark red -> red -> orange -> gold -> pale yellow -> white).
# Used two ways: three fixed points code the populations (E gold, mixed E-I orange, I red), and the full ramp
# shades the magnitude glyphs (decay weights, kernels, logits, memory contents, gauge, receptive fields).
H_DARK, H_ORANGE, H_GOLD, H_PALE = "#6e1617", "#ff7a1a", "#ffc933", "#fff0b8"
HOT_DEFS = [
    f'<linearGradient id="h_kE" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{H_GOLD}" stop-opacity="0"/>'
    f'<stop offset="1" stop-color="{H_GOLD}" stop-opacity="0.62"/></linearGradient>',
    f'<linearGradient id="h_kI" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{P_RED}" stop-opacity="0"/>'
    f'<stop offset="1" stop-color="{P_RED}" stop-opacity="0.7"/></linearGradient>',
    f'<linearGradient id="h_dec" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{H_DARK}"/>'
    f'<stop offset="0.45" stop-color="{P_RED}"/><stop offset="0.75" stop-color="{H_ORANGE}"/><stop offset="1" stop-color="{H_GOLD}"/></linearGradient>',
    f'<linearGradient id="h_gauge" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{P_RED}"/>'
    f'<stop offset="0.5" stop-color="{H_ORANGE}"/><stop offset="1" stop-color="{H_GOLD}"/></linearGradient>',
    f'<linearGradient id="h_sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{P_CARD}"/>'
    f'<stop offset="0.42" stop-color="{H_DARK}"/><stop offset="0.75" stop-color="{P_RED}"/><stop offset="1" stop-color="{H_ORANGE}"/></linearGradient>',
    f'<radialGradient id="h_rf" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="{H_PALE}"/>'
    f'<stop offset="0.55" stop-color="{H_GOLD}"/><stop offset="1" stop-color="{H_ORANGE}"/></radialGradient>',
]
DARK_T = dict(
    bg=P_BLACK, ink=P_WHITE, grey=P_WHITE, line=P_WHITE, dash=P_GRAY, neutral=P_GRAY, syn_panel=P_GRAY,
    E=H_GOLD, I=P_RED, LI=P_RED, SY_panel=P_GRAY, dec_line=P_WHITE, MEM=P_GRAY, gate_slash=P_WHITE, HRF=P_GRAY,
    attn_fan=P_WHITE, attn_text=P_WHITE, attn_path=P_WHITE, kernE="url(#h_kE)", kernI="url(#h_kI)",
    mlp_fill=P_BLACK, phi_fill=P_CARD, pool_fill=P_RED, data_fill="url(#h_sky)", data_mtn=P_BLACK,
    data_sun=H_GOLD, hrf_out=H_DARK, hrf_in="url(#h_rf)", attn_ell=P_CARD, out_ell=P_CARD, ell_line=P_WHITE,
    dec_fill="url(#h_dec)", mat_low_fill=P_BLACK, mat_low_stroke=P_CARD, mat_stroke=P_BLACK,
    mat_EE="#33270a", mat_EI="#361a06", mat_II="#45100f", sel_EE=H_GOLD, sel_EI=H_ORANGE, sel_II=P_RED,
    bars=P_GRAY, bar_cols=[P_RED, H_GOLD, "#a01c1e", H_ORANGE], gauge_arc="url(#h_gauge)", defs=HOT_DEFS,
    needle=P_WHITE, readout=P_WHITE, badge_fill=P_RED_FILL, badge_line=P_RED_FILL, badge_text=P_WHITE,
    # silenced cells share the black outline of the active cells, as in Option E, so they add no extra border
    cell_line=P_BLACK, sil_fill=P_CARD, sil_hatch=P_GRAY, sil_line=P_BLACK, trace=P_WHITE,
    fam={
        "gray": ("#ffffff", ["#a9a9a9", "#8f8f8f", "#bdbdbd", "#7d7d7d"]),
        "E": (H_PALE, [H_GOLD, "#f2b824", "#ffd45c", "#e6a817"]),
        "I": ("#fbd5d5", [P_RED, "#d92f31", "#ee5a5c", P_RED_FILL]),
        "a": ("#ffffff", ["#9e9e9e", "#8a8a8a"]),
        "z": ("#ffe0c2", [H_ORANGE, "#f26b10", "#ff9a4d"]),
        "zI": ("#fbd5d5", [P_RED, P_RED_FILL]),
        "zB": (H_PALE, [H_GOLD, "#f2b824"]),
        "m": (H_PALE, [P_RED, H_ORANGE, H_GOLD, "#f25a2a", "#ff9a4d"]),
    })
T = DARK_T if DARK else LIGHT_T
defs.extend(T.get("defs", []))

# ============================== shapes, all straight and drawn at one line weight


def stroke(pts, w=LW, color=None, op=1.0, closed=False, layer=None, dash=None):
    layer = mid if layer is None else layer
    pts = np.asarray(pts, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    da = f' stroke-dasharray="{dash}"' if dash else ""
    layer.append(f'<path d="{d_of(densify(pts))}" fill="none" stroke="{color or T["line"]}" stroke-width="{w}" '
                 f'stroke-linecap="round" stroke-linejoin="round" opacity="{op:.2f}"{da}/>')


def panel(x, y, w, h, color, sw=LW, r=12):
    """Block border: a rectangle with small chamfered corners."""
    k = r * 0.3
    pts = [(x + r, y), (x + w - r, y), (x + w - k, y + k), (x + w, y + r), (x + w, y + h - r), (x + w - k, y + h - k),
           (x + w - r, y + h), (x + r, y + h), (x + k, y + h - k), (x, y + h - r), (x, y + r), (x + k, y + k)]
    stroke(pts, sw, color, closed=True, layer=back)


def bar_end(p, u, color, half=7, w=LW):
    """Flat inhibitory bar at point p, across direction u."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    stroke([p + n * half, p - n * half], w, color)


HEAD = 13                 # one arrowhead size, symmetric


def text(s, x, y, size=18, font="bold", color=None, anchor="start", rot=0, op=1.0):
    return _text(s, x, y, size, font, color or T["ink"], anchor, rot, op)


def arrow(pts, w=LW, color=None, head=None, op=1.0, dash=None, layer=None):
    color = color or T["line"]
    layer = mid if layer is None else layer
    pts = np.asarray(pts, float)
    tip = pts[-1]
    back_pt = trim_end(pts, HEAD)[-1]
    u = (tip - back_pt) / (np.linalg.norm(tip - back_pt) + 1e-9)
    stroke(trim_end(pts, HEAD * 0.75), w, color, op, dash=dash, layer=layer)
    n = np.array([-u[1], u[0]])
    base = tip - u * HEAD
    layer.append(f'<path d="{d_of([tip, base + n * HEAD * 0.42, base - n * HEAD * 0.42], True)}" fill="{color}" '
                 f'stroke="{color}" stroke-width="1.2" stroke-linejoin="round" opacity="{op:.2f}"/>')



def blob(cx, cy, rx, ry, fill, w=LW, n=64, layer=None, line=None, container=False):
    """Filled ellipse with an outline. container=True marks a fill that holds a label (badge, node, ellipse), so the
    overlap checker ignores text inside it but still checks the outline."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)
    k = ' data-k="c"' if container else ""
    (mid if layer is None else layer).append(f'<path{k} d="{d_of(pts, True)}" fill="{fill}"/>')
    stroke(pts, w, line, closed=True, layer=layer)


def cell(x, y, s, fam="gray", silenced=False, w=LW):
    if silenced:
        mid.append(f'<rect x="{f(x)}" y="{f(y)}" width="{s}" height="{s}" fill="{T["sil_fill"]}"/>')
        for c in np.linspace(s * 0.2, s * 0.8, 3):
            mid.append(f'<path d="M{f(x + c)} {f(y + s - 2)} L{f(x + s - 2)} {f(y + c)}" stroke="{T["sil_hatch"]}" '
                       f'stroke-width="0.7" opacity="0.6"/>')
        line = T["sil_line"]
    else:
        c0, c1s = T["fam"][fam]
        gid = grad(c0, c1s[rng.integers(len(c1s))], rng.choice([35, 60, 90, 120, 145]))
        mid.append(f'<rect x="{f(x)}" y="{f(y)}" width="{s}" height="{s}" fill="url(#{gid})"/>')
        hatch(x, y, s)
        line = T["cell_line"]
    stroke([(x, y), (x + s, y), (x + s, y + s), (x, y + s)], w, line, closed=True)


def ortho(pts, r=10, n=8):
    """Orthogonal polyline with small rounded corners."""
    pts = [np.asarray(p, float) for p in pts]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        d1, d2 = p1 - p0, p2 - p1
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        rr = min(r, l1 / 2, l2 / 2)
        a, b = p1 - d1 / l1 * rr, p1 + d2 / l2 * rr
        out.append(a)
        for t in np.linspace(0, 1, n)[1:-1]:
            out.append((1 - t) ** 2 * a + 2 * t * (1 - t) * p1 + t ** 2 * b)
        out.append(b)
    out.append(pts[-1])
    return np.array(out)


# ============================== v3 type scale: large type for the poster (1 figure px = 0.778 pt when the figure is
# placed 19.456 in wide). Notes 24 px = 18.7 pt, titles 26 px = 20.2 pt, math 28 px. v2 used 16-20 px notes.
TITLE_SZ = 26             # block titles (capitals)
NOTE_SZ = 24              # every other label: the smallest text in the figure
MATH_SZ = 26              # math labels: STIX italic at 26 px has the x-height of Carlito Bold at 24 px
BADGE_R = 17              # callout radius (v2: 15)
BADGE_SZ = 24             # callout digits, one and two digits alike


BADGES = []               # v3: (number, x, y, radius, shape) of every callout, for figcheck.py


def callout(n, x, y):
    r = BADGE_R
    k0 = len(front)
    if n <= 10:
        blob(x, y, r, r, T["badge_fill"], w=LW, layer=front, line=T["badge_line"], container=True)
    else:
        pts = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
        front.append(f'<path data-k="c" d="{d_of(pts, True)}" fill="{T["badge_fill"]}"/>')
        stroke(pts, LW, T["badge_line"], closed=True, layer=front)
    for k in range(k0, len(front)):                      # tag the badge's fill and ring
        front[k] = front[k].replace("<path", '<path data-b="1"', 1)
    BADGES.append((n, x, y, r, "circle" if n <= 10 else "square"))
    sz = BADGE_SZ                                        # v3: 24 px for 1-13 (18.7 pt on the poster)
    text(str(n), x, y + 0.328 * sz, size=sz, font="bold", anchor="middle", color=T["badge_text"])


def title(n, s, x, y, size=TITLE_SZ):
    callout(n, x, y - 0.324 * size)
    text(s, x + BADGE_R + 10, y, size, font="bold")


def note(s, x, y, size=NOTE_SZ, anchor="start", color=None):
    text(s, x, y, size, font="bold", color=color or T["grey"], anchor=anchor)


def col(x, y, fams, silenced=()):
    out_l, out_r = [], []
    for i, fam in enumerate(fams):
        cell(x, y + i * S, S, fam, i in silenced, w=LW)
        out_l.append((x, y + (i + 0.5) * S))
        out_r.append((x + S, y + (i + 0.5) * S))
    return out_l, out_r


# ============================== CTM additions to the two themes
# CTM has one population of neurons with no E/I type, so its neurons are drawn in the neutral cell colors.
# Light (Option G): the attention side (o, S_action) keeps Option E's orange, S_out takes the output head's lavender.
# Dark (Option H): the attention side is the hot ramp's orange, S_out the poster's blue-gray.
FAM["pg"] = ("#eef0f1", ["#8c969b", "#7d878c", "#a3abaf"])
DARK_T["fam"]["pg"] = FAM["pg"]
CTM_LIGHT = dict(neuron_fan=INK, act_fan=LIGHT_T["attn_fan"], mlp_line=INK, sub_panel=LIGHT_T["neutral"],
                 mark=LIGHT_T["neutral"], absent_line="#b9b7b0", absent_text="#86847d",
                 res_back="#e6f4ec", res_front="#cfeadc", mat_base="#ecebe7", mat_outline=INK,
                 sel_action="#e8902a", sel_out="#8a72dc", act_fam="a", out_fam="z",
                 act_text=LIGHT_T["attn_text"], out_text="#6a55c8", act_path=LIGHT_T["attn_path"])
CTM_DARK = dict(neuron_fan=P_WHITE, act_fan=H_ORANGE, mlp_line=P_WHITE, sub_panel=P_GRAY,
                mark=P_GRAY, absent_line="#4f5558", absent_text=P_GRAY,
                res_back=H_DARK, res_front="url(#h_rf)", mat_base="#262626", mat_outline=P_WHITE,
                sel_action=H_ORANGE, sel_out=P_WHITE, act_fam="z", out_fam="pg",
                act_text=H_ORANGE, out_text="#c9cfd2", act_path=H_ORANGE)
T.update(CTM_DARK if DARK else CTM_LIGHT)
ACT, OUT = T["act_fam"], T["out_fam"]
NEU = ["gray"] * 10
FY = 326                  # one straight line: synapse output, pre-activation history, neuron model, new state


def absent(x, y, w, h, name, r=12):
    """Dashed outline for a TIDE block that CTM does not have."""
    k = r * 0.3
    pts = [(x + r, y), (x + w - r, y), (x + w - k, y + k), (x + w, y + r), (x + w, y + h - r), (x + w - k, y + h - k),
           (x + w - r, y + h), (x + r, y + h), (x + k, y + h - k), (x, y + h - r), (x, y + r), (x + k, y + k)]
    stroke(pts, LW, T["absent_line"], closed=True, layer=back, dash="10 7")
    note(name, x + w / 2, y + h / 2 - 4, TITLE_SZ, anchor="middle", color=T["absent_text"])
    note("Absent in CTM", x + w / 2, y + h / 2 + 28, anchor="middle", color=T["absent_text"])


# ============================== row 1: the recurrent core, left to right (y 96..524)
# (1) synapse model: a U-Net MLP maps [z(t); o(t)] to the pre-activations a(t)
panel(40, 96, 400, 428, T["syn_panel"], sw=LW, r=12)
TY = 130                  # v3: one title baseline for the row-1 blocks (v2: 133)
title(1, "SYNAPSE MODEL", 66, TY)
note("U-Net MLP, weights of either sign", 66, 160)
A_left, A_right = col(110, 200, ["gray"] * 7 + [ACT] * 3)       # post-activations, then the attention output
B_left, B_right = col(350, 200, NEU)                              # pre-activations
BN = 22
bn_y = [340 - 2 * BN + k * BN for k in range(4)]                  # bottleneck of the U-Net
for y in bn_y:
    cell(219, y, BN, "gray", w=LW)
bn_l = [(219, y + BN / 2) for y in bn_y]
bn_r = [(219 + BN, y + BN / 2) for y in bn_y]
fan(A_right[:7], bn_l, T["neuron_fan"], op=0.3)
fan(A_right[7:], bn_l, T["act_fan"], op=0.45, w=1.1)
fan(bn_r, B_left, T["neuron_fan"], op=0.3)
text("$z^{(t)}$", 124, 192, MATH_SZ, font="math", anchor="middle")
text("$a^{(t)}$", 364, 192, MATH_SZ, font="math", anchor="middle")
text("$o^{(t)}$", 98, 447, MATH_SZ, font="math", anchor="end", color=T["act_text"])
arrow([(380, FY), (494, FY)], LW, head=12)                         # pre-activations into the history

# (2)(3) neuron-level models: each neuron's private MLP reads its last M pre-activations
panel(480, 96, 600, 428, T["neutral"], sw=LW, r=12)
title(3, "NEURON-LEVEL MODELS", 506, TY)
panel(496, 150, 568, 358, T["sub_panel"], sw=LW, r=12)
callout(2, 520, 178)
note("FIFO of the last M pre-activations", 546, 184)
for i, yc in enumerate([FY - 92, FY - 46, FY, FY + 46, FY + 92]):
    trace(520, 790, yc, 10, 100 + i, color=T["trace"])
arrow([(520, 462), (790, 462)], LW, T["dash"], head=12)
note("Last M steps", 655, 490, anchor="middle")
ys_in, ys_hid = [FY - 30, FY, FY + 30], [FY - 39, FY - 13, FY + 13, FY + 39]
arrow([(796, FY), (834, FY)], LW, head=12)
for yi in ys_in:
    for yh in ys_hid:
        back.append(f'<line x1="852" y1="{yi}" x2="904" y2="{yh}" stroke="{T["mlp_line"]}" stroke-width="1.1" opacity="0.55"/>')
for yh in ys_hid:
    back.append(f'<line x1="904" y1="{yh}" x2="956" y2="{FY}" stroke="{T["mlp_line"]}" stroke-width="1.1" opacity="0.55"/>')
for (xx, ys) in [(852, ys_in), (904, ys_hid), (956, [FY])]:
    for yy in ys:
        blob(xx, yy, 8, 8, T["mlp_fill"], w=LW)
note("Per-neuron gated MLP", 912, FY - 63, anchor="middle")   # v3: 8 px right, clear of the trace ends
arrow([(966, FY), (1244, FY)], LW, head=13)

# (4) post-activations: the neuron-level models' outputs are the new state
panel(1120, 96, 280, 428, T["neutral"], sw=LW, r=12)
title(4, "POST-ACTIVATIONS", 1146, TY)
note("Neuron-model outputs", 1146, 160)                  # v3: shorter wording so it fits at 24 px
C_left, C_right = col(1246, 200, NEU)
text("$z^{(t+1)}$", 1260, 192, MATH_SZ, font="math", anchor="middle")

# TIDE's lateral inhibition has no counterpart in CTM
absent(1440, 96, 320, 428, "LATERAL INHIBITION")

# step boundary: the post-activations become the next step's input and enter the post-activation history.
# The loop leaves block 4 and re-enters block 1 on the line y = FY, level with the arrows from block 1 to block 4,
# and climbs in the gap between the blocks, so it crosses no label.
arrow(ortho([(1276, FY), (1420, FY), (1420, 44), (24, 44), (24, FY), (108, FY)]), LW, T["dash"], dash="10 7")
note("Next internal step", 722, 32, anchor="middle")
arrow([(1260, 484), (1260, 598)], LW)
note("Into post-activation history", 1272, 566)

# corridor: attention output back into the synapse model's input
arrow(ortho([(570, 680), (570, 576), (124, 576), (124, 482)]), LW, T["act_path"])

# ============================== row 2: input on the left, readout on the right (y 600..880)
# data: the input image
note("DATA", 105, 652, TITLE_SZ, anchor="middle", color=T["ink"])
frame = [(58, 670), (152, 670), (152, 760), (58, 760)]
mid.append(f'<path d="{d_of(frame, True)}" fill="{T["data_fill"]}"/>')
mid.append(f'<path d="{d_of([(64, 754), (92, 714), (108, 734), (124, 704), (148, 754)], True)}" fill="{T["data_mtn"]}"/>')
blob(132, 690, 8, 8, T["data_sun"], w=LW)
stroke(frame, LW, closed=True)
arrow([(154, 716), (198, 716)], LW, head=12)

# feature extractor: a ResNet backbone, drawn as four stages that shrink in size and grow in depth
panel(200, 600, 240, 280, T["HRF"], sw=LW, r=12)
text("FEATURE EXTRACTOR", 222, 637, 23, font="bold")     # v4: one line at 23 px (26 px needs 228 px; the block gives 210)
note("ResNet backbone", 222, 667)
x = 218
for s, n in [(46, 1), (38, 2), (30, 3), (22, 4)]:
    for j in reversed(range(n)):
        ox, oy = x + 5 * j, 744 - s / 2 - 5 * j
        mid.append(f'<rect x="{f(ox)}" y="{f(oy)}" width="{s}" height="{s}" fill="{T["res_front"] if j == 0 else T["res_back"]}"/>')
        stroke([(ox, oy), (ox + s, oy), (ox + s, oy + s), (ox, oy + s)], LW, closed=True)
    x += s + 5 * (n - 1) + 14
arrow([(442, 716), (496, 716)], LW, head=13)
note("K, V", 468, 703, anchor="middle", color=T["ink"])

# (9)(10) cross-attention: S_action gives the query over the image features; o(t) returns to the synapse model
blob(570, 716, 72, 36, T["attn_ell"], w=LW, line=T["ell_line"], container=True)
text("ATTENTION", 570, 724, NOTE_SZ, font="bold", anchor="middle")
callout(9, 509, 664)       # v3: nudged up-left so the larger badge clears the ellipse
callout(10, 640, 626)
text("$o^{(t)}$", 584, 650, MATH_SZ, font="math", color=T["act_text"])

# (8) latent representations: S_action (left half) and S_out (right half)
for c in range(12):
    cell(690 + c * 30, 700, 30, ACT if c < 6 else OUT, w=LW)
callout(8, 700, 668)
text("LATENT", 700 + BADGE_R + 10, 676, TITLE_SZ, font="bold")
arrow([(688, 715), (646, 715)], LW, head=13)
note("Query", 666, 755, anchor="middle")
text("$S_{\\mathrm{action}}$", 780, 755, MATH_SZ, font="math", anchor="middle", color=T["act_text"])
text("$S_{\\mathrm{out}}$", 915, 755, MATH_SZ, font="math", anchor="middle", color=T["out_text"])

# (5)(6)(7) synchronization: decayed running sums of pair products over two random sets of neuron pairs
panel(1120, 600, 640, 280, T["SY_panel"], sw=LW, r=12)
text("SYNCHRONIZATION", 1146, 637, TITLE_SZ, font="bold")
for i, yc in enumerate([692, 726, 760]):
    trace(1528, 1740, yc, 10, 300 + i, color=T["trace"])
    mid.append(f'<rect x="1514" y="{yc - 4}" width="8" height="8" fill="{T["mark"]}"/>')
u_ = np.linspace(0, 1, 60)
dec = np.stack([1528 + 212 * u_, 806 - 24 * np.exp(-(1 - u_) * 2.6)], 1)
back.append(f'<path d="{d_of(np.vstack([dec, [[1740, 806], [1528, 806]]]), True)}" fill="{T["dec_fill"]}" opacity="0.9"/>')
stroke(dec, LW, T["dec_line"])
note("Decayed running sums", 1634, 837, anchor="middle")
callout(5, 1734, 655)      # v3: 5 px higher so the larger badge clears the first trace
arrow([(1508, 726), (1466, 726)], LW, head=12)
MX, MY, MS = 1300, 666, 16
act_pairs = {(0, 3), (1, 8), (2, 6), (4, 9), (6, 7)}
out_pairs = {(0, 6), (1, 4), (2, 7), (3, 9), (5, 8)}
for i in range(10):
    for j in range(10):
        x, y = MX + j * MS, MY + i * MS
        if j < i:
            mid.append(f'<rect x="{x}" y="{y}" width="{MS}" height="{MS}" fill="{T["mat_low_fill"]}" stroke="{T["mat_low_stroke"]}" stroke-width="0.8"/>')
            continue
        fill = T["sel_action"] if (i, j) in act_pairs else (T["sel_out"] if (i, j) in out_pairs else T["mat_base"])
        mid.append(f'<rect x="{x}" y="{y}" width="{MS}" height="{MS}" fill="{fill}" stroke="{T["mat_stroke"]}" stroke-width="0.8"/>')
stroke([(MX, MY), (MX + 10 * MS, MY), (MX + 10 * MS, MY + 10 * MS), (MX, MY + 10 * MS)], LW, T["mat_outline"], closed=True)
callout(6, 1272, 659)      # v3: nudged so the larger badge clears the pair cells
arrow([(1296, 715), (1252, 715)], LW, head=12)
for yy, fam in [(691, ACT), (721, OUT)]:
    for c in range(4):
        cell(1176 + c * 18, yy, 18, fam, w=LW)
callout(7, 1150, 664)
note("Random pairs", 1212, 781, anchor="middle")
arrow([(1174, 715), (1052, 715)], LW, head=13)

# ============================== row 3: output (y 887..990)
# (9) output head: S_out gives the logits at every step
OX, OY = 960, 923
blob(OX, OY, 72, 34, T["out_ell"], w=LW, line=T["ell_line"], container=True)
text("OUTPUT", OX, OY + 8, NOTE_SZ, font="bold", anchor="middle")
callout(9, OX + 86, OY - 30)
arrow([(OX, 732), (OX, OY - 36)], LW, head=13)
for k, hgt in enumerate([18, 30, 12, 24]):
    x0 = OX - 136 + k * 14
    mid.append(f'<rect x="{x0}" y="{OY + 27 - hgt}" width="11" height="{hgt}" fill="{(T.get("bar_cols") or [T["bars"]] * 4)[k]}"/>')
    stroke([(x0, OY + 27), (x0, OY + 27 - hgt), (x0 + 11, OY + 27 - hgt), (x0 + 11, OY + 27)], LW)
stroke([(OX - 142, OY + 28), (OX - 74, OY + 28)], LW)
note("Logits", OX - 108, OY + 56, anchor="middle")

# TIDE's memory has no counterpart in CTM
absent(1120, 912, 640, 180, "MEMORY")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
       f'aria-label="CTM architecture, one internal step, drawn on the grid of the TIDE figure">'
       f'<defs>{"".join(defs)}</defs><rect width="{W}" height="{H}" fill="{T["bg"]}"/>'
       + "".join(back) + "".join(mid) + "".join(front) + "</svg>")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else ("ctm_option_h.svg" if DARK else "ctm_option_g.svg")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(svg)
    print(f"{len(svg) // 1024} KB -> {out}")
    if os.environ.get("FIG_DUMP"):          # v3: layers and label boxes for figcheck.py
        import json
        base = os.environ["FIG_DUMP"]
        head = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs>{"".join(defs)}</defs>'
        allel = back + mid + front
        with open(base + "_text.svg", "w", encoding="utf-8") as fh:
            fh.write(head + "".join(e for e in front if 'data-k="t"' in e) + "</svg>")
        with open(base + "_gfx.svg", "w", encoding="utf-8") as fh:
            fh.write(head + "".join(e for e in allel if 'data-k="t"' not in e and 'data-k="c"' not in e) + "</svg>")
        with open(base + "_gfxnb.svg", "w", encoding="utf-8") as fh:
            fh.write(head + "".join(e for e in allel if 'data-k="t"' not in e and 'data-k="c"' not in e
                                    and 'data-b="1"' not in e) + "</svg>")
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(dict(W=W, H=H, texts=TEXTS, badges=BADGES), fh, ensure_ascii=False, indent=0)
