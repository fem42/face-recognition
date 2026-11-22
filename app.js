// app.js - Interfaccia: carica foto, trova il viso con MediaPipe, analizza, disegna il report.
import { buildRegions, analyze, lightingWarnings } from "./analysis.js";

const MP_VER = "0.10.14";
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}`;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WORK_FACE_WIDTH = 600;
const MAX_INPUT = 1600;

const $ = id => document.getElementById(id);
const status = (t, err = false) => { $("status").textContent = t; $("status").className = err ? "err" : ""; };
let landmarker = null;
let lastResult = null;
let lastDraw = null;

async function getLandmarker() {
  if (landmarker) return landmarker;
  status("Carico il modello del viso (solo la prima volta)…");
  let mod = null, lastErr = null;
  for (const u of [`${MP_BASE}/vision_bundle.mjs`, `${MP_BASE}/vision_bundle.js`, `${MP_BASE}/+esm`]) {
    try { mod = await import(u); if (mod.FaceLandmarker || mod.default?.FaceLandmarker) break; } catch (e) { lastErr = e; }
  }
  if (!mod) throw new Error("impossibile caricare MediaPipe (" + (lastErr?.message || lastErr) + ")");
  const { FaceLandmarker, FilesetResolver } = mod.FaceLandmarker ? mod : mod.default;
  const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL },
    runningMode: "IMAGE",
    numFaces: 1,
  });
  return landmarker;
}

async function decode(file) {
  // Safari/iPhone: createImageBitmap con opzioni puo' non essere supportato -> uso <img>
  try { return await createImageBitmap(file, { imageOrientation: "from-image" }); } catch { }
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    return img;
  } finally { setTimeout(() => URL.revokeObjectURL(url), 1000); }
}

async function loadImage(file) {
  const bmp = await decode(file);
  if (!bmp.width) throw new Error("formato foto non leggibile (se è HEIC, prova a scattare direttamente dalla pagina)");
  const s = Math.min(1, MAX_INPUT / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * s); c.height = Math.round(bmp.height * s);
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  return c;
}

const nextFrame = () => new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));

async function run(file) {
  try {
    $("results").hidden = true;
    const lmk = await getLandmarker();
    status("Cerco il viso…");
    const src = await loadImage(file);
    const res = lmk.detect(src);
    if (!res.faceLandmarks || !res.faceLandmarks.length) {
      status("Nessun viso trovato. Prova con una foto frontale e ben illuminata.", true);
      return;
    }
    const pts = res.faceLandmarks[0].map(p => [p.x * src.width, p.y * src.height]);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
    const mw = (x1 - x0) * 0.12, mh = (y1 - y0) * 0.12;
    const cx0 = Math.max(0, Math.floor(x0 - mw)), cy0 = Math.max(0, Math.floor(y0 - mh));
    const cx1 = Math.min(src.width, Math.ceil(x1 + mw)), cy1 = Math.min(src.height, Math.ceil(y1 + mh));
    const s = WORK_FACE_WIDTH / (x1 - x0);
    const w = Math.round((cx1 - cx0) * s), h = Math.round((cy1 - cy0) * s);
    const work = document.createElement("canvas");
    work.width = w; work.height = h;
    const wctx = work.getContext("2d", { willReadFrequently: true });
    wctx.imageSmoothingQuality = "high";
    wctx.drawImage(src, cx0, cy0, cx1 - cx0, cy1 - cy0, 0, 0, w, h);
    const lm = pts.map(([x, y]) => [(x - cx0) * s, (y - cy0) * s]);

    status("Analizzo la pelle (qualche secondo)…");
    await nextFrame();
    const img = wctx.getImageData(0, 0, w, h);
    const R = buildRegions(w, h, lm);
    const { scores, raw, overlay, L } = analyze(img.data, w, h, R);
    const warnings = lightingWarnings(L, w, h, R.pelle.m);
    const overall = Math.round(Object.values(scores).reduce((a, b) => a + b, 0) / Object.keys(scores).length);

    lastDraw = [work, img, R, overlay];
    drawFace(work, img, R, overlay);
    drawRadar(scores);
    showScores(overall, scores, warnings);
    lastResult = {
      data: new Date().toISOString(), punteggio: overall,
      punteggi: Object.fromEntries(Object.entries(scores).map(([k, v]) => [k, Math.round(v)])),
      misure_grezze: Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, +v.toFixed(3)])),
      avvisi_luce: warnings,
    };
    saveHistory(lastResult);
    status("");
    $("results").hidden = false;
    $("results").scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    console.error(e);
    status("Errore: " + (e.message || e), true);
  }
}

function drawFace(work, img, R, overlay) {
  const c = $("face");
  c.width = work.width; c.height = work.height;
  const ctx = c.getContext("2d");
  const out = new ImageData(new Uint8ClampedArray(img.data), img.width, img.height);
  const d = out.data, skin = R.pelle.m;
  for (let i = 0; i < skin.length; i++) if (skin[i]) {
    d[4 * i] = d[4 * i] * 0.85 + 230 * 0.15; d[4 * i + 1] = d[4 * i + 1] * 0.85 + 200 * 0.15; d[4 * i + 2] = d[4 * i + 2] * 0.85 + 255 * 0.15;
  }
  const colors = { pori: [255, 140, 0], punti_neri: [40, 40, 40], rughe: [200, 80, 255], macchie: [160, 90, 0], rossori: [255, 60, 60], lucidita: [0, 230, 255] };
  if ($("showMarks").checked) {
    for (const [k, col] of Object.entries(colors)) {
      const m = overlay[k]; if (!m) continue;
      for (let i = 0; i < m.length; i++) if (m[i]) { d[4 * i] = col[0]; d[4 * i + 1] = col[1]; d[4 * i + 2] = col[2]; }
    }
  }
  ctx.putImageData(out, 0, 0);
  ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 2;
  for (const k of ["fronte", "guance", "naso", "mento", "sotto_occhi", "zampe_gallina"]) {
    for (const poly of R[k].polys) {
      ctx.beginPath(); poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.stroke();
    }
  }
}

function drawRadar(scores) {
  const c = $("radar"), dpr = window.devicePixelRatio || 1, size = 360;
  c.width = size * dpr; c.height = size * dpr; c.style.width = size + "px"; c.style.height = size + "px";
  const ctx = c.getContext("2d"); ctx.scale(dpr, dpr);
  const labels = Object.keys(scores), n = labels.length, cx = size / 2, cy = size / 2, r = size * 0.32;
  const ang = i => -Math.PI / 2 + (2 * Math.PI * i) / n;
  ctx.strokeStyle = "#d5d9e0"; ctx.lineWidth = 1;
  for (const f of [0.25, 0.5, 0.75, 1]) {
    ctx.beginPath();
    for (let i = 0; i <= n; i++) { const a = ang(i % n); const x = cx + Math.cos(a) * r * f, y = cy + Math.sin(a) * r * f; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
  }
  for (let i = 0; i < n; i++) { ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(ang(i)) * r, cy + Math.sin(ang(i)) * r); ctx.stroke(); }
  ctx.beginPath();
  labels.forEach((k, i) => { const v = scores[k] / 100; const x = cx + Math.cos(ang(i)) * r * v, y = cy + Math.sin(ang(i)) * r * v; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.closePath(); ctx.fillStyle = "rgba(74,111,165,0.25)"; ctx.fill(); ctx.strokeStyle = "#4a6fa5"; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle = "#222"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  labels.forEach((k, i) => {
    const x = cx + Math.cos(ang(i)) * (r + 34), y = cy + Math.sin(ang(i)) * (r + 26);
    ctx.font = "12px system-ui, sans-serif"; ctx.fillText(k, x, y - 8);
    ctx.font = "bold 14px system-ui, sans-serif"; ctx.fillText(Math.round(scores[k]), x, y + 8);
  });
}

function showScores(overall, scores, warnings) {
  $("overall").textContent = overall;
  $("bar").style.width = overall + "%";
  $("warnings").innerHTML = warnings.map(w => `<li>${w}</li>`).join("");
  $("warnBox").hidden = !warnings.length;
}

// Storico locale (solo numeri, niente foto), per vedere i progressi
function saveHistory(r) {
  try {
    const h = JSON.parse(localStorage.getItem("storico") || "[]");
    h.push({ data: r.data, punteggio: r.punteggio, punteggi: r.punteggi });
    localStorage.setItem("storico", JSON.stringify(h.slice(-50)));
    renderHistory();
  } catch { /* storage non disponibile */ }
}
function renderHistory() {
  let h = [];
  try { h = JSON.parse(localStorage.getItem("storico") || "[]"); } catch { }
  $("historyBox").hidden = h.length < 2;
  $("history").innerHTML = h.slice(-10).reverse()
    .map(e => `<li><span>${new Date(e.data).toLocaleString("it-IT", { dateStyle: "short", timeStyle: "short" })}</span><b>${e.punteggio}</b></li>`).join("");
}

$("file").addEventListener("change", e => { if (e.target.files[0]) run(e.target.files[0]); e.target.value = ""; });
$("showMarks").addEventListener("change", () => { if (lastDraw) drawFace(...lastDraw); });
$("download").addEventListener("click", () => {
  if (!lastResult) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" }));
  a.download = `analisi-pelle-${lastResult.data.slice(0, 10)}.json`; a.click();
});
$("clearHistory").addEventListener("click", () => { try { localStorage.removeItem("storico"); } catch { } renderHistory(); });
renderHistory();
status("Pronta. Scegli o scatta una foto.");
