/* Строит анимированную сцену из scene.json (его делает scene_analyzer.py).
   Использование:
     <div class="scene" style="position:relative;width:100%;aspect-ratio:W/H"></div>
     renderScene(document.querySelector('.scene'), scene, '/days/папка_дня/');
   Контейнер должен иметь position:relative и пропорции фото (W/H из scene.size). */
(function () {
  var CSS = ".rs-cam{position:absolute;inset:0;transform-origin:50% 60%;animation:rs-cam 20s ease-in-out infinite alternate}" +
    "@keyframes rs-cam{from{transform:scale(1.04)}to{transform:scale(1.15) translate(-1.8%,-1.6%)}}" +
    ".rs-l{position:absolute;inset:0;width:100%;height:100%;display:block}" +
    ".rs-glow,.rs-spark{position:absolute;aspect-ratio:1;border-radius:50%;mix-blend-mode:screen;transform:translate(-50%,-50%)}" +
    ".rs-glow{background:radial-gradient(circle,rgba(255,228,170,.95),rgba(255,170,70,.28) 45%,transparent 72%);animation:rs-tw 3s ease-in-out infinite}" +
    ".rs-spark{background:radial-gradient(circle,#fff,rgba(190,220,255,.45) 40%,transparent 70%);animation:rs-sp 2.4s ease-in-out infinite}" +
    "@keyframes rs-tw{0%,100%{opacity:.2;scale:.8}40%{opacity:1;scale:1.15}70%{opacity:.45;scale:.95}}" +
    "@keyframes rs-sp{0%,100%{opacity:0;scale:.4}50%{opacity:1;scale:1.2}}" +
    ".rs-walker{position:absolute;transform-origin:50% 100%;animation:rs-walk 16s linear infinite}" +
    "@keyframes rs-walk{0%{transform:translate(0,0) scale(1);opacity:0}6%{opacity:1}84%{opacity:1}100%{transform:translate(var(--dx),var(--dy)) scale(.55);opacity:0}}" +
    ".rs-bob{width:100%;height:100%;transform-origin:50% 100%;animation:rs-step .64s ease-in-out infinite alternate}" +
    "@keyframes rs-step{from{transform:translateY(0) rotate(-1.4deg)}to{transform:translateY(-3.5%) rotate(1.4deg)}}" +
    ".rs-bob img{width:100%;height:100%;display:block}" +
    /* эффекты по областям: --s это сила от 0 до 1 */
    ".rs-sway{animation:rs-sway 4.5s ease-in-out infinite alternate}@keyframes rs-sway{from{transform:rotate(calc(var(--s)*-1.2deg))}to{transform:rotate(calc(var(--s)*1.2deg))}}" +
    ".rs-flicker{animation:rs-fl 2.2s ease-in-out infinite}@keyframes rs-fl{0%,100%{filter:brightness(1)}30%{filter:brightness(calc(1 + var(--s)*.35))}60%{filter:brightness(calc(1 - var(--s)*.15))}}" +
    ".rs-drift{animation:rs-dr 9s ease-in-out infinite}@keyframes rs-dr{0%{opacity:0;transform:translate(0,0)}50%{opacity:.8}100%{opacity:0;transform:translate(calc(var(--s)*2%),calc(var(--s)*-6%))}}" +
    ".rs-ripple{animation:rs-rp 3s ease-in-out infinite alternate}@keyframes rs-rp{from{transform:translateX(calc(var(--s)*-.6%))}to{transform:translateX(calc(var(--s)*.6%)) scaleY(1.01)}}" +
    ".rs-flutter{animation:rs-fu 1.6s ease-in-out infinite alternate}@keyframes rs-fu{from{transform:rotate(calc(var(--s)*-2.5deg))}to{transform:rotate(calc(var(--s)*2.5deg))}}" +
    "@media (prefers-reduced-motion:reduce){.rs-cam,.rs-glow,.rs-spark,.rs-walker,.rs-bob,.rs-l{animation:none}}";

  function el(tag, cls, parent) { var e = document.createElement(tag); if (cls) e.className = cls; if (parent) parent.appendChild(e); return e; }
  function pct(v) { return Math.round(v * 1000) / 10; }

  window.renderScene = function (host, scene, base) {
    if (!document.getElementById("rs-style")) { var st = el("style", "", document.head); st.id = "rs-style"; st.textContent = CSS; }
    host.innerHTML = "";
    var cam = el("div", "rs-cam", host), W = scene.size[0], H = scene.size[1];

    var bg = el("img", "rs-l", cam); bg.src = base + scene.background; bg.alt = "";

    // эффекты по областям: копия фона, видна только внутри размытой маски вокруг объекта
    (scene.effects || []).forEach(function (e) {
      var b = e.box, w = b[2] - b[0], h = b[3] - b[1], cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
      var im = el("img", "rs-l rs-" + e.effect, cam); im.src = base + scene.background; im.alt = "";
      var m = "radial-gradient(ellipse " + pct(w / 2 * 1.2) + "% " + pct(h / 2 * 1.2) + "% at " + pct(cx) + "% " + pct(cy) + "%, #000 45%, transparent 100%)";
      im.style.webkitMaskImage = im.style.maskImage = m;
      im.style.setProperty("--s", e.strength == null ? 0.6 : e.strength);
      im.style.transformOrigin = e.effect === "flutter" ? pct(b[0]) + "% " + pct(cy) + "%" : pct(cx) + "% " + pct(b[3]) + "%";
    });

    // огни: тёплое мерцание или блики
    (scene.lights || []).forEach(function (p, i) {
      var d = el("div", p.kind === "glow" ? "rs-glow" : "rs-spark", cam);
      d.style.cssText = "left:" + p.x + "%;top:" + p.y + "%;width:" + p.size + "%;animation-delay:" + (-(i * 0.73) % 3) + "s;animation-duration:" + (2 + (i % 5) * 0.55) + "s";
    });

    // люди идут вглубь кадра, к точке схода
    var vp = scene.vanishing_point || [0.5, 0.45];
    (scene.persons || []).forEach(function (p, i) {
      var b = p.box, fx = b[0] + b[2] / 2, fy = b[1] + b[3];
      var w = el("div", "rs-walker", cam);
      w.style.cssText = "left:" + pct(b[0]) + "%;top:" + pct(b[1]) + "%;width:" + pct(b[2]) + "%;height:" + pct(b[3]) + "%;animation-delay:" + (-i * 5) + "s";
      w.style.setProperty("--dx", pct((vp[0] - fx) * 0.35 / b[2]) + "%");   // сдвиг в долях размера человека
      w.style.setProperty("--dy", pct((vp[1] - fy) * 0.35 / b[3]) + "%");
      var bob = el("div", "rs-bob", w), im = el("img", "", bob); im.src = base + p.sprite; im.alt = "";
    });
  };
})();
