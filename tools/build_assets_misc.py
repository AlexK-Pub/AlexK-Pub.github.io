"""Profile photo, social card, and figures of the three earlier papers."""
import os, subprocess
import numpy as np
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
SITE = "/home/claude/work/site/assets/img"
AX = "/home/claude/work/assets_raw/arxiv"

def webp(im, path, w=1600, q=86):
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im); im = bg
    im = im.convert("RGB")
    if im.width > w: im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    im.save(path, "WEBP", quality=q, method=6)
    print(f"{path.replace(SITE,''):52s} {im.size} {os.path.getsize(path)/1024:.0f} KB")

def trim_white(im, pad=12, thr=248):
    a = np.asarray(im.convert("RGB")); m = (a < thr).any(axis=2)
    ys, xs = np.where(m)
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), min(im.width, xs.max() + pad), min(im.height, ys.max() + pad)))

# profile: square crop around the face
p = Image.open("/mnt/user-data/uploads/a00f8114-image.png").convert("RGB")
print("profile src", p.size)
sq = p.crop((90, 0, 90 + 640, 640))
for w in (640, 320):
    sq.resize((w, w), Image.LANCZOS).save(f"{SITE}/profile/alexander-kyuroson-{w}.jpg", quality=88, optimize=True)
    print("profile", w, os.path.getsize(f"{SITE}/profile/alexander-kyuroson-{w}.jpg") // 1024, "KB")

# power lines (arXiv:2308.07283)
P = AX + "/2308.07283/Figures/"
webp(Image.open(P + "figure1.png"), SITE + "/powerline/extraction.webp")
webp(Image.open(P + "drone_snow_text.png"), SITE + "/powerline/inspection-flight.webp", w=1400)
webp(Image.open(P + "drone.png"), SITE + "/powerline/platform.webp", w=1100)
webp(Image.open(P + "raw_data3.png"), SITE + "/powerline/point-cloud.webp", w=1400)
webp(trim_white(Image.open(P + "model.png")), SITE + "/powerline/catenary-model.webp", w=1400)
webp(trim_white(Image.open(P + "pl_denoised_top.png")), SITE + "/powerline/lines-xy.webp", w=1400)
webp(trim_white(Image.open(P + "PL_Clusters.png")), SITE + "/powerline/lines-segmented.webp", w=1400)
subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile", P + "Pipeline_2.pdf", "/tmp/pl_pipeline"], check=True)
webp(trim_white(Image.open("/tmp/pl_pipeline.png"), pad=16), SITE + "/powerline/pipeline.webp", w=1800, q=90)

# SubT dataset (arXiv:2304.14520)
S = AX + "/2304.14520/Images/"
webp(Image.open(S + "Fig1.png"), SITE + "/subt/tunnel-vapor.webp")
webp(Image.open(S + "Fig2_robot_in_fog_edited3.jpg"), SITE + "/subt/platform.webp", w=1128)
webp(trim_white(Image.open(S + "Fig2_isometric_Components2.png")), SITE + "/subt/components.webp", w=1100)
webp(trim_white(Image.open(S + "Fig2_Side_and_Distances2.png")), SITE + "/subt/fov-side.webp", w=1100)
webp(trim_white(Image.open(S + "Fig3_radar_topview.png")), SITE + "/subt/radar-top.webp", w=900)
for n, o in [("dark_return.jpg", "ir-dark"), ("execess_smoke_static.jpg", "ir-vapor"), ("smoke_human_static.jpg", "ir-safety-clothing")]:
    im = Image.open(S + n); w, h = im.size
    im = im.crop((8, 30, w - 8, h - 22))                                              # drop the viewer window chrome
    g = np.asarray(im.convert("L")).mean(axis=1); rows = np.where(g > 14)[0]          # and its black letterbox bars
    webp(im.crop((0, int(rows.min()), im.width, int(rows.max()) + 1)), SITE + f"/subt/{o}.webp", w=900)
webp(trim_white(Image.open(S + "Fig5_mjolk_pcd.png")), SITE + "/subt/map.webp", w=1600)
webp(trim_white(Image.open(S + "Odom_LiDARs.png")), SITE + "/subt/odometry.webp", w=1300)

# smoke filtration (arXiv:2308.07264)
F = AX + "/2308.07264/figures/"
webp(trim_white(Image.open(F + "dust-drone-concept.png"), pad=6), SITE + "/smoke/lidar-in-dust.webp", w=940)
webp(trim_white(Image.open(F + "Figure1_Right_Dusty.png")), SITE + "/smoke/pcl-raw.webp", w=1300)
webp(trim_white(Image.open(F + "Figure1_Left_Filtered.png")), SITE + "/smoke/pcl-filtered.webp", w=1300)
webp(trim_white(Image.open(F + "Figure1_dust.png")), SITE + "/smoke/pcl-dust.webp", w=1300)
webp(Image.open(F + "dustTunnel1.PNG"), SITE + "/smoke/tunnel-1.webp", w=900)
webp(Image.open(F + "dustTunnel2.PNG"), SITE + "/smoke/tunnel-2.webp", w=900)
webp(trim_white(Image.open(F + "Spot_Test2_Raw_1.png")), SITE + "/smoke/test2-raw.webp", w=1500)
webp(trim_white(Image.open(F + "Spot_Test2_Filt_Smoke_1.png")), SITE + "/smoke/test2-filtered.webp", w=1500)
subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile", F + "Pipeline_Temp.pdf", "/tmp/smoke_pipeline"], check=True)
sp = trim_white(Image.open("/tmp/smoke_pipeline.png"), pad=16)
ink = (np.asarray(sp.convert("L")) < 200).any(axis=1)                                # keep the diagram, not the scratch shapes parked below it
y, gap, end = 0, 0, sp.height
started = False
for i, v in enumerate(ink):
    if v: started, gap = True, 0
    elif started:
        gap += 1
        if gap > 60: end = i - gap + 20; break
webp(sp.crop((0, 0, sp.width, end)), SITE + "/smoke/pipeline.webp", w=1800, q=90)
