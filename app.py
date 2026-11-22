import os, re, json, base64, datetime, uuid, shutil, glob
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from mood_grade import grade_set, LOOKS
from scene_analyzer import find_lights, find_glints, find_stars

BASE = os.path.dirname(os.path.abspath(__file__))
DAYS = os.path.join(BASE, "days")
TMP = os.path.join(BASE, "tmp")
os.makedirs(DAYS, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

app = Flask(__name__, static_folder=None)
MOODS = list(LOOKS)
MODEL = "claude-sonnet-5-5"

# ИИ необязателен: если задан ключ ANTHROPIC_API_KEY, используем Claude, иначе работает бесплатный режим по правилам
client = None
if os.environ.get("ANTHROPIC_API_KEY"):
    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        client = None

SYSTEM = f"""Ты — модуль анализа дня для личного дневника атмосферы.
Тебе дают запись о дне, фото и описание песни. Верни ТОЛЬКО JSON без пояснений и без ```:
{{
  "title": "название дня, 3-7 слов, по-русски, спокойный тон, без точки",
  "mood": одно из {MOODS},
  "energy": число от 0 до 1,
  "season": "весна|лето|осень|зима|неизвестно",
  "time_of_day": "утро|день|вечер|ночь|неизвестно",
  "palette": ["#hex", "#hex", "#hex"],
  "caption": "короткая подпись, до 10 слов, от первого лица",
  "reason": "1-2 предложения: почему выбрано такое настроение (текст, фото, песня)"
}}
Песня обязательно должна влиять на mood и energy. Не ставь диагнозов, описывай только атмосферу."""


def analyze_claude(text, song, paths):
    content = []
    for p in paths:
        data = base64.standard_b64encode(open(p, "rb").read()).decode()
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
    content.append({"type": "text", "text": f"Запись о дне:\n{text}\n\nПесня дня: {song}"})
    msg = client.messages.create(model=MODEL, max_tokens=1000, system=SYSTEM,
                                 messages=[{"role": "user", "content": content}])
    raw = msg.content[0].text
    r = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    r["engine"] = "claude"
    return r


# ---------- Бесплатный режим: настроение по словам, песне и цветам фото ----------
KW = {
    "tense_cold": ["дедлайн", "срочно", "офис", "работ", "проект", "хакатон", "сдач", "код", "экзамен", "устал", "бессонн", "electronic", "techno", "edm", "электрон", "техно"],
    "calm_warm": ["уют", "кофе", "какао", "чай", "тепл", "дом", "разговарив", "парк", "книг", "камин", "осен", "jazz", "джаз", "sinatra", "acoustic", "lofi", "пиано", "piano"],
    "energetic": ["танц", "бег", "вечеринк", "концерт", "быстр", "драйв", "спорт", "клуб", "поход", "скорост", "рок", "rock", "metal", "рэп", "rap", "punk", "dance"],
    "melancholic": ["грустн", "дожд", "одинок", "скуча", "тоск", "пасмурн", "серый", "плакал", "blues", "блюз", "sad"],
    "joyful": ["рад", "счастл", "смех", "смеял", "праздник", "день рождения", "друзь", "море", "лето", "пляж", "весел", "улыб", "подарок"],
    "dreamy": ["мечт", "звезд", "туман", "тихо", "сон", "облак", "закат", "рассвет", "ambient", "dream"],
    "winter_dreamy": ["снег", "зим", "мороз", "метел", "новый год", "рождеств", "гирлянд", "ёлк", "елк", "snow", "christmas", "иней"],
}
ENERGY = {"calm_warm": .3, "energetic": .9, "tense_cold": .7, "melancholic": .2, "joyful": .8, "dreamy": .25, "winter_dreamy": .25, "cozy_winter": .3}
CAPTION = {
    "calm_warm": "Тепло, тихо и никуда не хочется", "energetic": "Быстро, ярко и без остановки",
    "tense_cold": "Ночь, тишина и один дедлайн", "melancholic": "Тихо, немного грустно и всё равно красиво",
    "joyful": "Хороший день, который хочется повторить", "dreamy": "Как будто сон, из которого не хочется выходить",
    "winter_dreamy": "Снег идёт, огни горят, и никуда не хочется", "cozy_winter": "За окном снег, а внутри тепло"}
ADJ = {"calm_warm": "Тёплый", "energetic": "Яркий", "tense_cold": "Холодный", "melancholic": "Тихий", "joyful": "Радостный",
       "dreamy": "Мечтательный", "winter_dreamy": "Зимний", "cozy_winter": "Уютный зимний"}
PLACES = [("кофейн", "в кофейне"), ("кафе", "в кафе"), ("офис", "в офисе"), ("парк", "в парке"), ("пруд", "у пруда"), ("лес", "в лесу"),
          ("горах", "в горах"), ("горы", "в горах"), ("море", "у моря"), ("пляж", "на пляже"), ("город", "в городе"), ("улиц", "на улице"),
          ("двор", "во дворе"), ("дом", "дома")]
WITH = [("друзь", "с друзьями"), ("девушк", "с девушкой"), ("парн", "с парнем"), ("семь", "с семьёй"), ("коллег", "с командой"), ("команд", "с командой"), ("подруг", "с подругой")]


def photo_stats(imgs):
    v = s = w = 0.0
    pix = []
    for im in imgs:
        sm = cv2.resize(im, (64, 64))
        hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
        v += hsv[..., 2].mean(); s += hsv[..., 1].mean(); w += float(sm[..., 2].mean()) - float(sm[..., 0].mean())
        pix.append(sm.reshape(-1, 3))
    n = len(imgs)
    pts = np.vstack(pix).astype(np.float32)
    _, lab, cent = cv2.kmeans(pts, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(lab.ravel(), minlength=3))
    palette = ["#%02X%02X%02X" % (int(cent[i][2]), int(cent[i][1]), int(cent[i][0])) for i in order]
    return v / n, s / n, w / n, palette


def analyze_free(text, song, imgs):
    tl, sl = text.lower(), song.lower()
    v, s, w, palette = photo_stats(imgs)
    score = {m: 0.0 for m in MOODS}
    hits = []
    for m, kws in KW.items():
        for k in kws:
            if k in tl: score[m] += 2; hits.append(k)
            if k in sl: score[m] += 2; hits.append(k)
    if v < 90: score["tense_cold" if w < 0 else "calm_warm"] += 1      # тёмное фото: холодное или с тёплыми огнями
    if v > 150 and s > 110: score["joyful"] += 1                          # светлое и яркое
    if s < 60: score["melancholic"] += .5                                 # бледные цвета
    if v > 140 and w < -10 and s < 110: score["winter_dreamy"] += .5      # светлое, холодное, как снег
    mood = max(score, key=score.get) if max(score.values()) > 0 else "calm_warm"
    if score["winter_dreamy"] >= 2 and score["calm_warm"] >= 2 and mood in ("calm_warm", "winter_dreamy"):
        mood = "cozy_winter"
    season = "неизвестно"
    for k, sn in [("зим", "зима"), ("снег", "зима"), ("мороз", "зима"), ("лето", "лето"), ("летом", "лето"), ("летн", "лето"), ("пляж", "лето"), ("осен", "осень"), ("листь", "осень"), ("листв", "осень"), ("листопад", "осень"), ("весн", "весна"), ("цветущ", "весна"), ("сакур", "весна")]:
        if k in tl: season = sn; break
    if "winter" in mood: season = "зима"
    if re.search("утр", tl): tod = "утро"
    elif re.search("ночь|ночью", tl): tod = "ночь"
    elif re.search("вечер", tl): tod = "вечер"
    elif re.search("днем|днём|полдень", tl): tod = "день"
    else: tod = "ночь" if v < 70 else ("вечер" if v < 110 else "день")
    place = next((p for k, p in PLACES if k in tl), "")
    comp = next((c for k, c in WITH if k in tl), "")
    sp = {"зима": "зимой", "лето": "летом", "осень": "осенью", "весна": "весной"}.get(season, "")
    title = " ".join(x for x in [tod.capitalize(), place, comp, sp] if x)
    if len(title.split()) < 3:
        base = " ".join(x for x in [tod.capitalize(), place, comp] if x) if "winter" in mood else title   # без повтора «зимний… зимой»
        title = ADJ[mood] + " " + base[0].lower() + base[1:]
    reason = ("Режим без ИИ: настроение определено по словам в записи и песне (" + ", ".join(dict.fromkeys(hits))[:80] + ") и по цветам фото."
              if hits else "Режим без ИИ: настроение определено по яркости и цветам фото.")
    return {"title": title, "mood": mood, "energy": ENERGY[mood], "season": season, "time_of_day": tod,
            "palette": palette, "caption": CAPTION[mood], "reason": reason, "engine": "rules"}


# Страница и картинки ищутся в папке static, а если её нет (например, файлы загружены на GitHub без папок),
# то рядом с app.py. Так проект запускается при любой раскладке. Отдаются только безопасные типы файлов.
SAFE_EXT = (".html", ".js", ".css", ".jpg", ".jpeg", ".png", ".webp")


def _static_dir(name):
    for d in (os.path.join(BASE, "static"), BASE):
        if os.path.isfile(os.path.join(d, name)):
            return d
    return None


@app.get("/")
def index():
    d = _static_dir("index.html")
    if d is None:
        return "Не найден index.html: положите его в папку static или рядом с app.py", 404
    return send_from_directory(d, "index.html")


@app.get("/static/<path:p>")
def static_files(p):
    name = os.path.basename(p)
    if not name.lower().endswith(SAFE_EXT):
        return "Not found", 404
    full = os.path.join(BASE, "static", p)
    if os.path.isfile(full) and os.path.realpath(full).startswith(os.path.realpath(os.path.join(BASE, "static"))):
        return send_from_directory(os.path.join(BASE, "static"), p)
    d = _static_dir(name)
    return send_from_directory(d, name) if d else ("Not found", 404)


@app.get("/days/<path:p>")
def day_files(p):
    return send_from_directory(DAYS, p)


@app.get("/tmp/<path:p>")
def tmp_files(p):
    return send_from_directory(TMP, p)


@app.post("/api/analyze")
def analyze():
    text = request.form.get("text", "")
    song = request.form.get("song", "")
    sid = uuid.uuid4().hex[:10]
    d = os.path.join(TMP, sid)
    os.makedirs(d)
    paths = []
    for i, f in enumerate(request.files.getlist("photos")[:5]):
        p = os.path.join(d, f"p{i}.jpg")
        f.save(p)
        paths.append(p)
    if not paths:
        return jsonify({"error": "Нужно хотя бы одно фото"}), 400
    imgs = [cv2.imread(p) for p in paths]
    result = None
    if client:
        try:
            result = analyze_claude(text, song, paths)
        except Exception as e:
            print("Claude не сработал, включаю бесплатный режим:", e)
    if result is None:
        result = analyze_free(text, song, imgs)
    if result.get("mood") not in MOODS:
        result["mood"] = "calm_warm"
    try:
        graded = grade_set(paths, result["mood"], d)
        result["graded"] = True
    except Exception as e:
        print("Цветокоррекция пропущена:", e)
        graded = paths
        result["graded"] = False
    scenes = []                                       # огни для каждого фото: найдены автоматически
    for im in imgs:
        sc = {"w": int(im.shape[1]), "h": int(im.shape[0]), "lights": [],
              "day": bool(np.median(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) >= 110)}
        try:
            sc["lights"] = find_lights(im) + find_glints(im) + find_stars(im)
        except Exception as e:
            print("Огни не найдены:", e)
        scenes.append(sc)
    result["scenes"] = scenes
    tl = text.lower()                                 # погода и огонь по словам в записи
    result["weather"] = "rain" if re.search("дожд|ливень|морос|капл", tl) else ("snow" if re.search("снег|метел|снегопад", tl) else "")
    result["fire"] = bool(re.search("свеч|камин|костер|костёр|огонь|огня|огнём|очаг", tl))
    result["session"] = sid
    result["photos"] = [f"/tmp/{sid}/{os.path.basename(g)}" for g in graded]
    return jsonify(result)


@app.post("/api/save")
def save():
    result = json.loads(request.form["result"])
    sid = str(result.pop("session", ""))
    result.pop("photos", None)
    if not re.fullmatch(r"[0-9a-f]{10}", sid):
        return jsonify({"error": "Нет данных сессии, создайте воспоминание заново"}), 400
    src = os.path.join(TMP, sid)
    date = datetime.date.today().isoformat()
    safe = re.sub(r"[^\w\- ]", "", result.get("title", "день"), flags=re.UNICODE).strip() or "день"
    folder, n = f"{date} {safe}", 2
    while os.path.exists(os.path.join(DAYS, folder)):
        folder = f"{date} {safe} {n}"
        n += 1
    path = os.path.join(DAYS, folder)
    os.makedirs(path)
    files = sorted(glob.glob(os.path.join(src, "graded_p*.jpg"))) or sorted(glob.glob(os.path.join(src, "p*.jpg")))
    for i, f in enumerate(files):
        shutil.copy(f, os.path.join(path, f"photo_{i}.jpg"))
    audio = request.files.get("audio")
    if audio and audio.filename:
        ext = os.path.splitext(audio.filename)[1] or ".mp3"
        audio.save(os.path.join(path, "song" + ext))
    data = {**result, "text": request.form.get("text", ""), "song": request.form.get("song", ""), "date": date}
    with open(os.path.join(path, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return jsonify({"folder": folder})


@app.get("/api/days")
def days():
    out = []
    for name in sorted(os.listdir(DAYS), reverse=True):
        p = os.path.join(DAYS, name, "result.json")
        if os.path.isfile(p):
            files = os.listdir(os.path.join(DAYS, name))
            with open(p, encoding="utf-8") as fh:
                result = json.load(fh)
            out.append({"folder": name, "result": result,
                        "photos": sorted(f for f in files if f.startswith("photo_")),
                        "audio": next((f for f in files if f.startswith("song.")), None)})
    return jsonify(out)


if __name__ == "__main__":
    print("Движок анализа:", "Claude" if client else "бесплатный режим без ИИ")
    app.run(debug=True, port=5000)
