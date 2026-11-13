"""Dark-theme versions of the figures of the three earlier papers (power lines, SubT dataset, smoke filtration).

Each figure is rebuilt from the paper's source file exactly as build_assets_misc.py builds the light version
(same trim, same size), then passed through darken_figures.darken and saved next to it as <name>-dark.webp.
Photographs are left alone: they are shown as they are on both themes.
Point clouds colored by a color map use data=True, and the pale-gray dust sketch gets brighter fills.
"""
import os, subprocess, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from darken_figures import darken
Image.MAX_IMAGE_PIXELS = None
SITE = "/home/claude/work/site/assets/img"
AX = "/home/claude/work/assets_raw/arxiv"

def flat(im):
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im); im = bg
    return im.convert("RGB")

def trim_white(im, pad=12, thr=248):
    a = np.asarray(im.convert("RGB")); m = (a < thr).any(axis=2)
    ys, xs = np.where(m)
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), min(im.width, xs.max() + pad), min(im.height, ys.max() + pad)))

def make(im, name, w, q=92, **kw):
    im = flat(im)
    if im.width > w: im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    light = Image.open(f"{SITE}/{name}.webp")
    assert im.size == light.size, (name, im.size, light.size)        # the pair must overlay exactly
    out = f"{SITE}/{name}-dark.webp"
    darken(im, **kw).save(out, "WEBP", quality=q, method=6)
    print(f"{name + '-dark.webp':40s} {im.size} {os.path.getsize(out) / 1024:.0f} KB")

if __name__ == "__main__":
    P = AX + "/2308.07283/Figures/"
    make(Image.open(P + "figure1.png"), "powerline/extraction", 1600, data=True)
    make(trim_white(Image.open(P + "model.png")), "powerline/catenary-model", 1400)
    make(trim_white(Image.open(P + "pl_denoised_top.png")), "powerline/lines-xy", 1400)
    make(trim_white(Image.open(P + "PL_Clusters.png")), "powerline/lines-segmented", 1400)
    subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile", P + "Pipeline_2.pdf", "/tmp/pl_pipeline"], check=True)
    make(trim_white(Image.open("/tmp/pl_pipeline.png"), pad=16), "powerline/pipeline", 1800)

    S = AX + "/2304.14520/Images/"
    make(trim_white(Image.open(S + "Fig2_isometric_Components2.png")), "subt/components", 1100)
    make(trim_white(Image.open(S + "Fig2_Side_and_Distances2.png")), "subt/fov-side", 1100)
    make(trim_white(Image.open(S + "Fig3_radar_topview.png")), "subt/radar-top", 900)
    make(trim_white(Image.open(S + "Fig5_mjolk_pcd.png")), "subt/map", 1600, data=True)
    make(trim_white(Image.open(S + "Odom_LiDARs.png")), "subt/odometry", 1300)

    F = AX + "/2308.07264/figures/"
    make(trim_white(Image.open(F + "dust-drone-concept.png"), pad=6), "smoke/lidar-in-dust", 940, pale=(0.62, 0.74), stroke_gain=3.0)
    make(trim_white(Image.open(F + "Figure1_Right_Dusty.png")), "smoke/pcl-raw", 1300, data=True)
    make(trim_white(Image.open(F + "Figure1_Left_Filtered.png")), "smoke/pcl-filtered", 1300, data=True)
    make(trim_white(Image.open(F + "Figure1_dust.png")), "smoke/pcl-dust", 1300, data=True)
    make(trim_white(Image.open(F + "Spot_Test2_Raw_1.png")), "smoke/test2-raw", 1500)
    make(trim_white(Image.open(F + "Spot_Test2_Filt_Smoke_1.png")), "smoke/test2-filtered", 1500)
    subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile", F + "Pipeline_Temp.pdf", "/tmp/smoke_pipeline"], check=True)
    sp = trim_white(Image.open("/tmp/smoke_pipeline.png"), pad=16)
    ink = (np.asarray(sp.convert("L")) < 200).any(axis=1)             # keep the diagram, not the scratch shapes parked below it
    gap, end, started = 0, sp.height, False
    for i, v in enumerate(ink):
        if v: started, gap = True, 0
        elif started:
            gap += 1
            if gap > 60: end = i - gap + 20; break
    make(sp.crop((0, 0, sp.width, end)), "smoke/pipeline", 1800)
