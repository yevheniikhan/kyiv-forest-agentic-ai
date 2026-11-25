"""Animation: Sentinel-2 scene -> layer stack -> per-pixel feature vector -> model -> biomass map.
Frames are drawn with PIL and encoded with ffmpeg (H.264, 1920x1080, 25 fps).
Text is sized for a video embedded about 7.5 inches wide on a slide (1 px is about 0.29 pt)."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 25
BG = (251, 250, 247); DARK = (12, 58, 52); TEAL = (20, 82, 74); OCHRE = (154, 107, 47); GREY = (90, 101, 96); TXT = (22, 33, 29)
F = "/usr/share/fonts/truetype/crosextra/"
fC = ImageFont.truetype(F + "Carlito-Bold.ttf", 62)      # scene caption
fL = ImageFont.truetype(F + "Carlito-Regular.ttf", 44)   # labels
fLb = ImageFont.truetype(F + "Carlito-Bold.ttf", 46)
fN = ImageFont.truetype(F + "Carlito-Bold.ttf", 58)      # numbers in boxes

s2 = Image.open("fig/s2.png").convert("RGB"); agbm = Image.open("fig/agb_map.png").convert("RGB")
fx0, fy0, fs = 0.25, 0.30, 0.28


def crop(im):
    w, h = im.size
    return im.crop((int(fx0 * w), int(fy0 * h), int((fx0 + fs) * w), int(fy0 * h + fs * w)))


N = 24                                   # pixels per side in the schematic grid
S = 800                                  # size of the square image on screen
rgb = np.asarray(crop(s2).resize((N, N), Image.BOX)).astype(float)
agb = np.asarray(crop(agbm).resize((N, N), Image.NEAREST)).astype(float)
big_rgb = crop(s2).resize((S, S), Image.LANCZOS)
map_cells = Image.fromarray(agb.astype(np.uint8)).resize((S, S), Image.NEAREST)

r, g, b = rgb[..., 0] / 255, rgb[..., 1] / 255, rgb[..., 2] / 255
exg = np.clip((2 * g - r - b) * 3 + .4, 0, 1)
bright = np.clip((r + g + b) / 3 * 1.6, 0, 1)
LAYERS = [("B2 синій", b, (70, 110, 170)), ("B3 зелений", g, (70, 150, 90)), ("B4 червоний", r, (180, 70, 60)),
          ("B5 червоний край", exg, (150, 70, 120)), ("B8 ближній ІЧ", exg ** .7, (120, 60, 60)),
          ("B11, B12 SWIR", 1 - exg * .8, (150, 110, 50)), ("VV, VH радар", np.clip(1 - bright, 0, 1), (90, 90, 90)),
          ("рельєф", np.tile(np.linspace(.3, .7, N), (N, 1)), (110, 95, 70))]
LS = 500                                 # layer tile size before projection


def layer_img(arr, tint, size=LS):
    a = np.clip(arr, 0, 1)
    im = np.zeros((N, N, 3))
    for k in range(3):
        im[..., k] = 245 * (1 - a) + tint[k] * a
    img = Image.fromarray(im.astype(np.uint8)).resize((size, size), Image.NEAREST)
    d = ImageDraw.Draw(img)
    for i in range(N + 1):
        p = min(int(i * size / N), size - 1)
        d.line([(p, 0), (p, size)], fill=(255, 255, 255), width=1)
        d.line([(0, p), (size, p)], fill=(255, 255, 255), width=1)
    return img


def iso(img, alpha=238):
    s = img.size[0]
    A = np.array([[0.85, 0.40], [-0.22, 0.40]]); Ai = np.linalg.inv(A); off = np.array([0, 0.22 * s])
    co = (Ai[0, 0], Ai[0, 1], -Ai[0, 0] * off[0] - Ai[0, 1] * off[1], Ai[1, 0], Ai[1, 1], -Ai[1, 0] * off[0] - Ai[1, 1] * off[1])
    rgba = img.convert("RGBA"); rgba.putalpha(alpha)
    return rgba.transform((int(s * 1.25), int(s * 0.62)), Image.AFFINE, co, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))


def iso_pt(u, v, s, ox, oy):
    return ox + 0.85 * u + 0.40 * v, oy + 0.22 * s - 0.22 * u + 0.40 * v


ISO = [iso(layer_img(arr, tint)) for _, arr, tint in LAYERS]
IH = ISO[0].size[1]


def ease(t):
    t = min(max(t, 0), 1)
    return t * t * (3 - 2 * t)


def caption(d, txt):
    d.text((80, 56), txt, font=fC, fill=TXT)


PAL = np.array([(232, 217, 168), (185, 207, 150), (127, 174, 134), (63, 125, 102), (20, 82, 74), (10, 47, 42)], float)
cands = [(i, j) for i in range(N) for j in range(N) if agb[i, j].sum() < 600]
PY, PX = cands[len(cands) // 2]
kpal = int(np.argmin(((PAL - agb[PY, PX]) ** 2).sum(1)))
lo, hi = kpal * 50, kpal * 50 + 50
vec = [float(np.clip(arr[PY, PX], 0.08, 1)) for _, arr, _ in LAYERS]

GAP = 76                                 # vertical distance between layers when the stack is open
TOP = 170                                # top of the drawing area
STACK_H = IH + 7 * GAP
T1, T2, T3, T4 = 3.0, 8.0, 14.0, 20.0
TOT = int(23.5 * FPS)
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "out/layers_pixels_ai.mp4"], stdin=subprocess.PIPE)


def draw_stack(im, d, ox, k=1.0, labels=0.0, mark=False):
    base = TOP + STACK_H - IH
    prev = None
    for i, (name, _, _) in enumerate(LAYERS):
        y = int(base - i * (10 + (GAP - 10) * k))
        im.paste(ISO[i], (ox, y), ISO[i])
        if labels > 0:
            col = tuple(int(BG[c] + (TXT[c] - BG[c]) * labels) for c in range(3))
            d.text((ox + ISO[i].size[0] + 40, y + IH // 2 - 28), name, font=fL, fill=col)
        if mark:
            cs = LS / N
            pts = [iso_pt(PX * cs, PY * cs, LS, ox, y), iso_pt((PX + 1) * cs, PY * cs, LS, ox, y),
                   iso_pt((PX + 1) * cs, (PY + 1) * cs, LS, ox, y), iso_pt(PX * cs, (PY + 1) * cs, LS, ox, y)]
            cx = sum(p[0] for p in pts) / 4; cy = sum(p[1] for p in pts) / 4
            d.polygon(pts, fill=OCHRE)
            if prev:
                d.line([(cx, cy), prev], fill=OCHRE, width=5)
            d.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], fill=OCHRE)
            prev = (cx, cy)


for fi in range(TOT):
    t = fi / FPS
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    if t < T1:
        caption(d, "1. Знімок складається з пікселів")
        x0, y0 = (W - S) // 2, TOP + 40
        im.paste(big_rgb, (x0, y0))
        a = int(255 * ease((t - 1.2) / 1.2))
        if a > 0:
            mask = Image.new("L", (S, S), 0); md = ImageDraw.Draw(mask)
            for i in range(N + 1):
                p = min(int(i * S / N), S - 1)
                md.line([(p, 0), (p, S)], fill=a, width=2); md.line([(0, p), (S, p)], fill=a, width=2)
            im.paste(Image.new("RGB", (S, S), (255, 255, 255)), (x0, y0), mask)
    elif t < T2:
        caption(d, "2. Кожен піксель має шари ознак")
        k = ease((t - T1) / 3.2)
        draw_stack(im, d, 430, k=k, labels=ease((k - 0.6) / 0.4))
    elif t < T3:
        caption(d, "3. Модель перетворює вектор ознак пікселя на біомасу")
        draw_stack(im, d, 60, mark=True)
        k = ease((t - T2 - 0.4) / 1.6)
        vx, vy = 800, TOP + 150
        d.text((vx, vy - 80), "вектор ознак", font=fLb, fill=TEAL)
        for i in range(len(LAYERS)):
            yy = vy + i * 70
            d.rectangle([vx, yy, vx + int(330 * vec[i] * k), yy + 44], fill=TEAL)
        k2 = ease((t - T2 - 2.4) / 1.0)
        mx, my = 1330, TOP + 190
        if k2 > 0:
            d.line([(vx + 360, my + 110), (vx + 360 + int((mx - vx - 380) * k2), my + 110)], fill=GREY, width=6)
        if k2 >= 1:
            d.rectangle([mx, my, mx + 500, my + 220], fill=DARK)
            d.text((mx + 40, my + 40), "Random Forest", font=fN, fill=BG)
            d.text((mx + 40, my + 125), "ансамбль дерев", font=fL, fill=(185, 204, 195))
        k3 = ease((t - T2 - 3.8) / 1.0)
        if k3 > 0:
            d.line([(mx + 250, my + 220), (mx + 250, my + 220 + int(90 * k3))], fill=GREY, width=6)
        if k3 >= 1:
            d.rectangle([mx, my + 310, mx + 500, my + 530], fill=OCHRE)
            d.text((mx + 40, my + 345), "біомаса пікселя", font=fL, fill=BG)
            d.text((mx + 40, my + 415), f"{lo}–{hi} т/га", font=fN, fill=BG)
    elif t < T4:
        caption(d, "4. З оцінок пікселів складається карта")
        x0, y0 = 200, TOP + 40
        im.paste(big_rgb, (x0, y0))
        n_done = int(N * N * ease((t - T3) / 5.2))
        cell = S / N
        full_rows, rest = divmod(n_done, N)
        if full_rows:
            im.paste(map_cells.crop((0, 0, S, int(full_rows * cell))), (x0, y0))
        if rest:
            im.paste(map_cells.crop((0, int(full_rows * cell), int(rest * cell), int((full_rows + 1) * cell))), (x0, y0 + int(full_rows * cell)))
        if n_done < N * N:
            d.rectangle([x0 + rest * cell, y0 + full_rows * cell, x0 + (rest + 1) * cell, y0 + (full_rows + 1) * cell], outline=OCHRE, width=6)
        d.text((x0 + S + 110, y0 + 240), "оцінено пікселів", font=fL, fill=GREY)
        d.text((x0 + S + 110, y0 + 310), f"{n_done} з {N * N}", font=fN, fill=TXT)
    else:
        caption(d, "5. Карта надземної біомаси, 30 м")
        mh = 800; mw = int(mh * agbm.size[0] / agbm.size[1])
        x0, y0 = 420, TOP + 40
        im.paste(agbm.resize((mw, mh), Image.LANCZOS), (x0, y0))
        lx = x0 + mw + 80
        for k_, c in enumerate(PAL.astype(int)):
            d.rectangle([lx, y0 + 80 + (5 - k_) * 100, lx + 44, y0 + 180 + (5 - k_) * 100], fill=tuple(c))
        d.text((lx + 70, y0 + 60), "300+ т/га", font=fL, fill=GREY)
        d.text((lx + 70, y0 + 640), "0", font=fL, fill=GREY)
    for edge in (T1, T2, T3, T4):
        dt = abs(t - edge)
        if dt < 0.25:
            im = Image.blend(im, Image.new("RGB", (W, H), BG), 1 - dt / 0.25)
    proc.stdin.write(im.tobytes())
proc.stdin.close(); proc.wait()
print("ok")
