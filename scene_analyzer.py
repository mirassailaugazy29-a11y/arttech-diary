"""Автоматический разбор фото: находит огни, людей и (через ИИ) всё, что может двигаться.
Запуск:  python scene_analyzer.py photo.jpg out_folder
Результат в out_folder: scene.json, background.jpg (фон без людей), person_0.png ... (люди отдельными слоями)
"""
import os, sys, json, base64
import cv2
import numpy as np


# ---------- 1. ОГНИ: яркие маленькие пятна ----------
def _blobs(mask, img, gray, W, H, max_n, min_area, max_area):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    items = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x, y = cent[i]
        b, g, r = img[lab == i].mean(axis=0)
        peak = int(gray[lab == i].max())
        items.append({"x": float(x), "y": float(y), "area": area, "warm": (r - b) > 10, "peak": peak,
                      "_score": peak + area})
    items.sort(key=lambda d: -d["_score"])
    return items[:max_n]


def find_lights(img, max_n=70):
    """Огни: яркие пятна, заметно светлее окружения. Тёплые ночью мерцают как огни, остальные дают блики."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    night = np.median(gray) < 110                      # ночное фото или дневное
    k = max(15, int(W * 0.04)) | 1
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask = ((tophat > (28 if night else 50)) & (gray > (185 if night else 245))).astype(np.uint8)
    out = []
    for d in _blobs(mask, img, gray, W, H, 400, 2, W * H * 0.002):
        if d["peak"] < (230 if night else 250):        # настоящий источник света «пережжён» до белого
            continue
        if len(out) >= (max_n if night else 20):
            break
        out.append({"x": round(d["x"] / W * 100, 1), "y": round(d["y"] / H * 100, 1),
                    "size": round(float(min(8, max(2.8, (d["area"] ** 0.5) / W * 100 * 2.2))), 1),
                    "kind": "glow" if (d["warm"] and night) else "spark"})
    return out


def find_glints(img, max_n=16):
    """Мелкие блестки на снегу и инее: светлые точки на светлом фоне. Мерцают быстро и мелко."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = max(7, int(W * 0.015)) | 1
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask = ((tophat > 18) & (gray > 170)).astype(np.uint8)
    items = _blobs(mask, img, gray, W, H, 400, 1, W * H * 0.0004)
    rng = np.random.default_rng(1)
    rng.shuffle(items)                                 # берём случайные, чтобы блестки были по всему кадру
    return [{"x": round(d["x"] / W * 100, 1), "y": round(d["y"] / H * 100, 1),
             "size": round(float(min(2.4, max(1.2, (d["area"] ** 0.5) / W * 100 * 2))), 1), "kind": "glint"}
            for d in items[:max_n]]


def find_stars(img, max_n=26):
    """Звёзды: тёмное синее небо вверху ночного фото."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.median(gray) >= 110:
        return []
    top = int(H * 0.45)
    b, r = img[:top, :, 0].astype(np.int16), img[:top, :, 2].astype(np.int16)
    sky = ((b > r + 12) & (gray[:top] < 95)).astype(np.uint8)
    sky = cv2.erode(sky, np.ones((15, 15), np.uint8))          # только открытое небо, не ветки
    ys, xs = np.nonzero(sky)
    if len(xs) < 50:
        return []
    rng = np.random.default_rng(2)
    idx = rng.choice(len(xs), min(max_n, len(xs)), replace=False)
    return [{"x": round(float(xs[i]) / W * 100, 1), "y": round(float(ys[i]) / H * 100, 1),
             "size": round(float(0.9 + rng.random() * 1.0), 1), "kind": "star"} for i in idx]


# ---------- 2. ЛЮДИ: rembg, если установлен, иначе встроенный детектор OpenCV ----------
def person_mask(img):
    H, W = img.shape[:2]
    try:
        from rembg import remove, new_session
        rgba = remove(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), session=new_session("u2net_human_seg"))
        return (np.asarray(rgba)[..., 3] > 40).astype(np.uint8), "rembg"
    except Exception:
        pass
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    s = 2.0 if W < 1000 else 1.0
    big = cv2.resize(img, None, fx=s, fy=s)
    rects, weights = hog.detectMultiScale(big, winStride=(8, 8), padding=(8, 8), scale=1.05)
    mask = np.zeros((H, W), np.uint8)
    for (x, y, rw, rh), wt in zip(rects, np.ravel(weights)):
        if wt < 0.5:
            continue
        x, y, rw, rh = [int(v / s) for v in (x, y, rw, rh)]
        gm = np.zeros((H, W), np.uint8)
        cv2.grabCut(img, gm, (x, y, rw, rh), np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_RECT)
        mask |= ((gm == 1) | (gm == 3)).astype(np.uint8)
    return mask, "hog"


def extract_people(img, out):
    H, W = img.shape[:2]
    mask, how = person_mask(img)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    persons, gone = [], np.zeros((H, W), np.uint8)
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in stats[i]]
        if a < H * W * 0.0004 or h > H * 0.6:          # слишком мелкое или слишком крупное (портрет) — не двигаем
            continue
        comp = (lab == i).astype(np.uint8)
        alpha = cv2.GaussianBlur(cv2.dilate(comp, np.ones((3, 3), np.uint8)).astype(np.float32), (3, 3), 0)
        x0, y0, x1, y1 = max(0, x - 2), max(0, y - 2), min(W, x + w + 2), min(H, y + h + 2)
        sprite = np.dstack([img[y0:y1, x0:x1], (alpha[y0:y1, x0:x1] * 255).astype(np.uint8)])
        fn = f"person_{len(persons)}.png"
        cv2.imwrite(os.path.join(out, fn), sprite)
        persons.append({"sprite": fn, "box": [x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H]})
        gone |= comp
    if gone.any():                                     # дорисовываем фон на месте людей
        k = max(5, int(W * 0.012)) | 1
        img = cv2.inpaint(img, cv2.dilate(gone, np.ones((k, k), np.uint8)) * 255, 5, cv2.INPAINT_TELEA)
    return persons, img, how


# ---------- 3. ВСЁ ОСТАЛЬНОЕ: ИИ-зрение подсказывает, что может двигаться ----------
PROMPT = """Посмотри на фото и верни ТОЛЬКО JSON без пояснений и без ```:
{"elements":[{"label":"что это","effect":"sway|flicker|drift|ripple|flutter","box":[x0,y0,x1,y1],"strength":0.0-1.0}],
 "vanishing_point":[x,y]}
Координаты в долях от 0 до 1 (x вправо, y вниз). effect: sway — деревья, ветки, трава; flicker — огонь, свечи, экраны;
drift — дым, пар, облака, туман; ripple — вода; flutter — флаги, шторы, ткань, волосы.
Не добавляй людей и маленькие точечные огни, их находит другой код. Не больше 8 элементов, только то, что реально видно.
vanishing_point — куда уходит дорога или улица (куда люди идут вглубь кадра)."""


def ask_llm(path):
    try:
        import anthropic
    except ImportError:
        return {}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {}
    img = cv2.imread(path)
    s = 1200 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s)
    data = base64.standard_b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()
    try:
        msg = anthropic.Anthropic().messages.create(
            model="claude-sonnet-5-5", max_tokens=1000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": PROMPT}]}])
        raw = msg.content[0].text
        return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception as e:
        print("ИИ-разбор пропущен:", e)
        return {}


# ---------- главная функция ----------
def analyze(path, out):
    os.makedirs(out, exist_ok=True)
    img = cv2.imread(path)
    H, W = img.shape[:2]
    persons, bg, how = extract_people(img, out)
    cv2.imwrite(os.path.join(out, "background.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 88])
    llm = ask_llm(path)
    scene = {"size": [W, H], "background": "background.jpg", "detector": how,
             "persons": persons, "lights": find_lights(img),
             "effects": llm.get("elements", []), "vanishing_point": llm.get("vanishing_point", [0.5, 0.45])}
    with open(os.path.join(out, "scene.json"), "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False, indent=2)
    return scene


if __name__ == "__main__":
    s = analyze(sys.argv[1], sys.argv[2])
    print(f"людей: {len(s['persons'])} ({s['detector']}), огней: {len(s['lights'])}, эффектов: {len(s['effects'])}")
