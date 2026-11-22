"""Цветокоррекция под настроение: делает все фото дня (и дневника) в одном «вайбе».
Запуск:  python mood_grade.py winter_dreamy out_folder photo1.jpg photo2.jpg photo3.jpg
Из кода: grade_set(["a.jpg", "b.jpg"], "winter_dreamy", "out_folder")
"""
import os, sys
import cv2
import numpy as np

# Каждый «look» — набор параметров. Меняйте и добавляйте свои настроения здесь.
#  temp: тепло(+)/холод(-) | bright: яркость | contrast | sat: насыщенность
#  shadows/highlights: тонирование теней и света (R, G, B) | lift: «выцветшие» чёрные
#  bloom: свечение ярких мест | vignette: затемнение углов | grain: плёночное зерно
LOOKS = {
    "calm_warm":     dict(temp=.04,  bright=1.03, contrast=.98, sat=1.10, shadows=(.03, .015, 0),  highlights=(.05, .03, -.02), lift=.03, bloom=.15, vignette=.30, grain=.030),
    "energetic":     dict(temp=0,    bright=1.00, contrast=1.25, sat=1.35, shadows=(-.02, 0, .04), highlights=(.04, .02, 0),     lift=0,   bloom=.05, vignette=.20, grain=.015),
    "tense_cold":    dict(temp=-.05, bright=.95,  contrast=1.20, sat=.70,  shadows=(-.02, .02, .07), highlights=(-.01, .02, .04), lift=.02, bloom=0,   vignette=.40, grain=.040),
    "melancholic":   dict(temp=-.02, bright=.92,  contrast=.90,  sat=.60,  shadows=(0, .02, .05),  highlights=(.02, .02, .03),   lift=.06, bloom=.10, vignette=.40, grain=.050),
    "joyful":        dict(temp=.03,  bright=1.08, contrast=1.05, sat=1.30, shadows=(.02, 0, .03),  highlights=(.05, .04, 0),     lift=.02, bloom=.12, vignette=.12, grain=.020),
    "dreamy":        dict(temp=0,    bright=1.08, contrast=.88,  sat=.95,  shadows=(.03, 0, .05),  highlights=(.05, .03, .04),   lift=.05, bloom=.30, vignette=.30, grain=.030),
    "winter_dreamy": dict(temp=-.02, bright=1.05, contrast=.92,  sat=1.00, shadows=(-.01, .02, .07), highlights=(.07, .04, -.01), lift=.04, bloom=.28, vignette=.35, grain=.030),
    "cozy_winter":   dict(temp=.04,  bright=1.03, contrast=.95,  sat=1.10, shadows=(0, .015, .05), highlights=(.07, .04, -.02),  lift=.03, bloom=.20, vignette=.35, grain=.035),
}


def _lum(x):
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def unify(images, strength_light=0.35, strength_color=0.7):
    """Сближает фото между собой: цветовой оттенок (сильно) и яркость (слабо),
    чтобы переход от дня к ночи сохранялся, но общий цвет был одним."""
    labs = [cv2.cvtColor(i, cv2.COLOR_RGB2LAB) for i in images]
    means = [l.reshape(-1, 3).mean(axis=0) for l in labs]
    group = np.mean(means, axis=0)
    out = []
    for lab, m in zip(labs, means):
        lab = lab.copy()
        lab[..., 0] += (group[0] - m[0]) * strength_light
        lab[..., 1] += (group[1] - m[1]) * strength_color
        lab[..., 2] += (group[2] - m[2]) * strength_color
        out.append(np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1))
    return out


def apply_look(rgb, look, seed=0):
    x = rgb.copy()
    H, W = x.shape[:2]
    x[..., 0] += look["temp"]; x[..., 2] -= look["temp"]                     # тепло / холод
    x = np.clip(x, 0, 1) ** (1 / look["bright"])                              # яркость
    x = np.clip((x - .5) * look["contrast"] + .5, 0, 1)                       # контраст
    l = _lum(x)[..., None]
    x = np.clip(l + (x - l) * look["sat"], 0, 1)                              # насыщенность
    l = _lum(x)[..., None]
    x = x + np.array(look["shadows"], np.float32) * (1 - l) ** 2 + np.array(look["highlights"], np.float32) * l ** 2  # тонирование
    x = np.clip(x * (1 - look["lift"]) + look["lift"], 0, 1)                  # выцветшие чёрные
    if look["bloom"] > 0:                                                     # свечение ярких мест
        hi = np.clip(x - .65, 0, 1) / .35
        glow = cv2.GaussianBlur(hi, (0, 0), max(3, W * 0.012))
        x = 1 - (1 - x) * (1 - np.clip(glow * look["bloom"] * 1.5, 0, 1))
    if look["vignette"] > 0:                                                  # виньетка
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H * .45) / (H / 2)) ** 2)
        x = x * (1 - look["vignette"] * np.clip(d - .5, 0, 1) ** 2 * 1.6)[..., None]
    if look["grain"] > 0:                                                     # зерно
        n = np.random.default_rng(seed).normal(0, look["grain"], (H, W, 1)).astype(np.float32)
        x = x + 0.6 * n * (1 - np.abs(x - .5) * 1.2)
    return np.clip(x, 0, 1)


def grade_set(paths, mood, out_dir, unify_photos=True):
    look = LOOKS.get(mood, LOOKS["calm_warm"])
    os.makedirs(out_dir, exist_ok=True)
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 for p in paths]
    if unify_photos and len(imgs) > 1:
        imgs = unify(imgs)
    out = []
    for i, (p, im) in enumerate(zip(paths, imgs)):
        g = apply_look(im, look, seed=i)
        fn = os.path.join(out_dir, "graded_" + os.path.basename(p))
        cv2.imwrite(fn, cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        out.append(fn)
    return out


if __name__ == "__main__":
    print(grade_set(sys.argv[3:], sys.argv[1], sys.argv[2]))
