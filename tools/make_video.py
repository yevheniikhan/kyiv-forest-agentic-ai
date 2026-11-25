# Requires fig/s2.png and fig/agb_map.png (UTM 36N previews from the demonstration run).
"""Animation: Sentinel-2 scene -> layer stack -> per-pixel feature vector -> model -> biomass map.
Frames drawn with PIL, encoded with ffmpeg (H.264, 1920x1080, 25 fps)."""
import subprocess, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1920, 1080, 25
BG = (251, 250, 247); DARK = (12, 58, 52); TEAL = (20, 82, 74); OCHRE = (154, 107, 47); GREY = (90, 101, 96); TXT = (22, 33, 29)
F = "/usr/share/fonts/truetype/dejavu/"
fT = ImageFont.truetype(F + "DejaVuSerif-Bold.ttf", 50)
fH = ImageFont.truetype(F + "DejaVuSans-Bold.ttf", 30)
fB = ImageFont.truetype(F + "DejaVuSans.ttf", 26)
fS = ImageFont.truetype(F + "DejaVuSans.ttf", 21)
fM = ImageFont.truetype(F + "DejaVuSansMono.ttf", 22)

s2 = Image.open("fig/s2.png").convert("RGB"); agbm = Image.open("fig/agb_map.png").convert("RGB")
fx0, fy0, fs = 0.25, 0.30, 0.28


def crop(im):
    w, h = im.size
    return im.crop((int(fx0 * w), int(fy0 * h), int((fx0 + fs) * w), int(fy0 * h + fs * w)))


N = 24  # pixels per side in the schematic grid
rgb = np.asarray(crop(s2).resize((N, N), Image.BOX)).astype(float)
agb = np.asarray(crop(agbm).resize((N, N), Image.NEAREST)).astype(float)
big_rgb = crop(s2).resize((560, 560), Image.LANCZOS)

# layers: three real RGB channels, then derived layers shown schematically
r, g, b = rgb[..., 0] / 255, rgb[..., 1] / 255, rgb[..., 2] / 255
exg = np.clip((2 * g - r - b) * 3 + .4, 0, 1)
bright = np.clip((r + g + b) / 3 * 1.6, 0, 1)
LAYERS = [("B2 синій", b, (70, 110, 170)), ("B3 зелений", g, (70, 150, 90)), ("B4 червоний", r, (180, 70, 60)),
          ("B5 червоний край", exg, (150, 70, 120)), ("B8 ближній ІЧ", exg ** .7, (120, 60, 60)),
          ("B11–B12 SWIR", 1 - exg * .8, (150, 110, 50)), ("VV, VH радар", np.clip(1 - bright, 0, 1), (90, 90, 90)),
          ("рельєф", np.tile(np.linspace(.3, .7, N), (N, 1)), (110, 95, 70))]


def layer_img(arr, tint, size=360, grid=True):
    a = np.clip(arr, 0, 1)
    im = np.zeros((N, N, 3))
    for k in range(3):
        im[..., k] = 245 * (1 - a) + tint[k] * a
    img = Image.fromarray(im.astype(np.uint8)).resize((size, size), Image.NEAREST)
    if grid:
        d = ImageDraw.Draw(img)
        for i in range(N + 1):
            p = int(i * size / N)
            d.line([(p, 0), (p, size)], fill=(255, 255, 255), width=1)
            d.line([(0, p), (size, p)], fill=(255, 255, 255), width=1)
    return img


def iso(img, alpha=255):
    """Project a square image to an isometric parallelogram (RGBA)."""
    s = img.size[0]
    out_w, out_h = int(s * 1.25), int(s * 0.62)
    # map output (x, y) -> input (u, v): parallelogram with top vertex skew
    # x = 0.85u + 0.4v, y = 0.22u - 0.0 + 0.4v  -> solve inverse
    A = np.array([[0.85, 0.40], [-0.22, 0.40]])
    Ai = np.linalg.inv(A)
    off = np.array([0, 0.22 * s])
    coeffs = (Ai[0, 0], Ai[0, 1], -Ai[0, 0] * off[0] - Ai[0, 1] * off[1],
              Ai[1, 0], Ai[1, 1], -Ai[1, 0] * off[0] - Ai[1, 1] * off[1])
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba.transform((out_w, out_h), Image.AFFINE, coeffs, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))


def iso_pt(u, v, s, ox, oy):
    return ox + 0.85 * u + 0.40 * v, oy + 0.22 * s - 0.22 * u + 0.40 * v


def ease(t):
    t = min(max(t, 0), 1)
    return t * t * (3 - 2 * t)


def header(d, title, sub=None):
    d.text((90, 60), title, font=fT, fill=TXT)
    if sub:
        d.text((92, 128), sub, font=fB, fill=GREY)
    d.line([(90, 178), (W - 90, 178)], fill=(201, 199, 191), width=2)
    d.text((90, H - 60), "Схематична візуалізація: шари B2–B4 – реальні канали Sentinel-2; інші шари показано умовно.", font=fS, fill=GREY)


PX, PY = 9, 14  # highlighted pixel
TOT = 22 * FPS
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "out/layers_pixels_ai.mp4"], stdin=subprocess.PIPE)

vec_vals = [float(np.clip(arr[PY, PX], 0, 1)) for _, arr, _ in LAYERS]
PAL = np.array([(232, 217, 168), (185, 207, 150), (127, 174, 134), (63, 125, 102), (20, 82, 74), (10, 47, 42)], float)
# choose a forest pixel and read its class from the map colour ramp (0-300 Mg/ha in six steps)
cands = [(i, j) for i in range(N) for j in range(N) if agb[i, j].sum() < 600]
PY, PX = cands[len(cands) // 2]
kpal = int(np.argmin(((PAL - agb[PY, PX]) ** 2).sum(1)))
agb_val = kpal * 50 + 25

for fi in range(TOT):
    t = fi / FPS
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    if t < 3.0:  # scene 1: the image
        header(d, "Знімок Sentinel-2: кожен піксель 10–20 м", "Ділянка Боярської ЛДС, червень–серпень 2022, UTM 36N")
        k = ease(t / 1.2)
        small = big_rgb.resize((int(560 * (0.85 + .15 * k)),) * 2)
        im.paste(small, (W // 2 - small.size[0] // 2, 230 + (560 - small.size[1]) // 2))
        if t > 1.6:  # pixel grid appears
            gi = layer_img(np.zeros((N, N)), (0, 0, 0), 560, grid=True).convert("RGBA")
            a = int(255 * ease((t - 1.6) / 1.0))
            mask = Image.new("L", (560, 560), 0); md = ImageDraw.Draw(mask)
            for i in range(N + 1):
                p = int(i * 560 / N); md.line([(p, 0), (p, 560)], fill=a, width=1); md.line([(0, p), (560, p)], fill=a, width=1)
            im.paste(Image.new("RGB", (560, 560), (255, 255, 255)), (W // 2 - 280, 230), mask)
    elif t < 8.0:  # scene 2: split into layers
        header(d, "Піксель розкладається на шари ознак", "спектральні канали, радар, рельєф")
        k = ease((t - 3.0) / 3.5)
        ox, base = 640, 900
        for i, (name, arr, tint) in enumerate(LAYERS):
            li = iso(layer_img(arr, tint), 235)
            y = base - i * (12 + 62 * k) - 140
            im.paste(li, (ox, int(y)), li)
            if k > .6:
                a = ease((k - .6) / .4)
                col = tuple(int(BG[c] + (TXT[c] - BG[c]) * a) for c in range(3))
                d.text((ox + li.size[0] + 30, int(y) + 70), name, font=fB, fill=col)
    elif t < 13.0:  # scene 3: pixel column -> vector -> model -> value
        header(d, "Для кожного пікселя модель отримує вектор ознак", "random forest навчено на відбитках GEDI L4A")
        ox, base = 300, 900
        sz = 360
        for i, (name, arr, tint) in enumerate(LAYERS):
            li = iso(layer_img(arr, tint), 235)
            y = base - i * 74 - 140
            im.paste(li, (ox, int(y)), li)
            # highlight pixel cell on each layer
            cs = sz / N
            pts = [iso_pt(PX * cs, PY * cs, sz, ox, y), iso_pt((PX + 1) * cs, PY * cs, sz, ox, y),
                   iso_pt((PX + 1) * cs, (PY + 1) * cs, sz, ox, y), iso_pt(PX * cs, (PY + 1) * cs, sz, ox, y)]
            cx_ = sum(p[0] for p in pts) / 4; cy_ = sum(p[1] for p in pts) / 4
            d.polygon(pts, fill=OCHRE, outline=OCHRE)
            d.ellipse([cx_ - 9, cy_ - 9, cx_ + 9, cy_ + 9], outline=OCHRE, width=4)
            if i > 0:
                d.line([(cx_, cy_), (prev[0], prev[1])], fill=OCHRE, width=3)
            prev = (cx_, cy_)
        k = ease((t - 8.3) / 1.5)
        # feature vector bars
        vx, vy = 900, 330
        d.text((vx, vy - 50), "вектор ознак пікселя", font=fH, fill=TEAL)
        for i, (name, arr, tint) in enumerate(LAYERS):
            yy = vy + i * 46
            d.text((vx, yy), name, font=fS, fill=GREY)
            d.rectangle([vx + 230, yy + 4, vx + 230 + int(220 * vec_vals[i] * k), yy + 30], fill=TEAL)
        k2 = ease((t - 10.0) / 1.0)
        if k2 > 0:
            mx, my = 1430, 420
            d.line([(vx + 470, 500), (vx + 470 + int(60 * k2), 500)], fill=GREY, width=4)
            d.rounded_rectangle([mx, my, mx + 300, my + 170], radius=18, fill=DARK)
            d.text((mx + 30, my + 30), "Random forest", font=fH, fill=(251, 250, 247))
            d.text((mx + 30, my + 85), "300 дерев рішень", font=fB, fill=(185, 204, 195))
        k3 = ease((t - 11.3) / 1.0)
        if k3 > 0:
            d.line([(1580, 595), (1580, 595 + int(60 * k3))], fill=GREY, width=4)
            d.rounded_rectangle([1430, 670, 1730, 790], radius=18, fill=OCHRE)
            d.text((1460, 690), "AGB пікселя", font=fB, fill=(251, 250, 247))
            d.text((1460, 730), f"{agb_val - 25}–{agb_val + 25} Мг/га", font=fH, fill=(251, 250, 247))
    elif t < 19.0:  # scene 4: pixel-by-pixel map fill
        header(d, "Модель обходить усі пікселі і будує карту біомаси", "кожен піксель оцінюється окремо, далі – суми і перевірка")
        S = 560
        x0, y0 = 300, 250
        im.paste(big_rgb, (x0, y0))
        n_done = int(N * N * ease((t - 13.0) / 5.0))
        cell = S / N
        mapcells = Image.fromarray(agb.astype(np.uint8)).resize((S, S), Image.NEAREST)
        for idx in range(n_done):
            i, j = divmod(idx, N)
            box = (int(j * cell), int(i * cell), int((j + 1) * cell), int((i + 1) * cell))
            im.paste(mapcells.crop(box), (x0 + box[0], y0 + box[1]))
        if n_done < N * N:
            i, j = divmod(n_done, N)
            d.rectangle([x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell, y0 + (i + 1) * cell], outline=OCHRE, width=4)
        d.text((x0 + S + 120, 330), f"оброблено пікселів: {n_done} з {N * N}", font=fM, fill=TXT)
        d.text((x0 + S + 120, 390), "на всій ділянці: ≈ 2,2 млн пікселів 30 м", font=fB, fill=GREY)
        d.text((x0 + S + 120, 450), "перевірки після кроку: діапазон 0–600 Мг/га,", font=fB, fill=GREY)
        d.text((x0 + S + 120, 490), "частка пропусків, просторова крос-валідація", font=fB, fill=GREY)
    else:  # scene 5: final map
        header(d, "Результат: карта надземної біомаси, 30 м", "середня 101 Мг/га; рішення про використання ухвалює дослідник")
        full = agbm.resize((560, 582))
        im.paste(full, (W // 2 - 280, 235))
        pal = [(232, 217, 168), (185, 207, 150), (127, 174, 134), (63, 125, 102), (20, 82, 74), (10, 47, 42)]
        for k_, c in enumerate(pal):
            d.rectangle([W // 2 + 320, 760 - k_ * 70, W // 2 + 345, 830 - k_ * 70], fill=c)
        d.text((W // 2 + 360, 410), "300+ Мг/га", font=fS, fill=GREY); d.text((W // 2 + 360, 805), "0", font=fS, fill=GREY)
    # fade transitions between scenes
    for edge in (3.0, 8.0, 13.0, 19.0):
        dt = abs(t - edge)
        if dt < 0.25:
            im = Image.blend(im, Image.new("RGB", (W, H), BG), 1 - dt / 0.25)
    proc.stdin.write(im.tobytes())
proc.stdin.close(); proc.wait()
# poster frame for the slide
print("ok")
