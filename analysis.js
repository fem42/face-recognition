// analysis.js - Misure della pelle da un'immagine del viso.
// Nessuna dipendenza: funziona nel browser e in Node (per i test).
// Porta in JavaScript la logica di skin_analyzer.py.

// ---------------------------------------------------------------------------
// Indici landmark MediaPipe Face Mesh (478 punti)
// ---------------------------------------------------------------------------
export const IDX = {
  FACE_OVAL: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
  FOREHEAD: [54, 103, 67, 109, 10, 338, 297, 332, 284, 300, 293, 334, 296, 336, 107, 66, 105, 63, 70],
  CHEEK_R: [116, 117, 118, 119, 100, 142, 203, 206, 216, 192, 213, 147, 123],
  CHEEK_L: [345, 346, 347, 348, 329, 371, 423, 426, 436, 416, 433, 376, 352],
  NOSE: [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 64, 294, 48, 278, 115, 344, 129, 358],
  CHIN: [152, 148, 176, 149, 150, 377, 400, 378, 379, 199, 200, 201, 421, 208, 428],
  UNDER_EYE_R: [33, 7, 163, 144, 145, 153, 154, 155, 133, 111, 117, 118, 119, 120, 121, 128],
  UNDER_EYE_L: [263, 249, 390, 373, 374, 380, 381, 382, 362, 340, 346, 347, 348, 349, 350, 357],
  EYE_R: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  EYE_L: [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
  BROW_R: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
  BROW_L: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
  LIPS: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
  NOSTRILS: [98, 327, 2, 64, 294, 240, 460, 97, 326],
};

// ---------------------------------------------------------------------------
// Geometria e maschere
// ---------------------------------------------------------------------------
export function convexHull(points) {
  const p = points.map(q => [q[0], q[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  upper.pop(); lower.pop();
  return lower.concat(upper); // senso antiorario (in coordinate y-giu': orario a schermo)
}

/** Riempie un poligono convesso (in senso coerente) dentro una maschera Uint8. */
export function fillConvex(mask, w, h, hull) {
  if (hull.length < 3) return;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of hull) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
  x1 = Math.min(w - 1, Math.ceil(x1)); y1 = Math.min(h - 1, Math.ceil(y1));
  const n = hull.length;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      let inside = true;
      for (let i = 0; i < n; i++) {
        const a = hull[i], b = hull[(i + 1) % n];
        if ((b[0] - a[0]) * (y + 0.5 - a[1]) - (b[1] - a[1]) * (x + 0.5 - a[0]) < 0) { inside = false; break; }
      }
      if (inside) mask[y * w + x] = 1;
    }
  }
}

function hullMask(w, h, pts) {
  const m = new Uint8Array(w * h);
  const hull = convexHull(pts);
  fillConvex(m, w, h, hull);
  return { m, polys: [hull] };
}

function union(...regs) {
  const m = new Uint8Array(regs[0].m.length);
  const polys = [];
  for (const r of regs) { for (let i = 0; i < m.length; i++) m[i] |= r.m[i]; polys.push(...r.polys); }
  return { m, polys };
}

function crowFeet(w, h, outer, inner) {
  const ew = Math.hypot(outer[0] - inner[0], outer[1] - inner[1]);
  const d = [(outer[0] - inner[0]) / ew, (outer[1] - inner[1]) / ew];
  const n = [-d[1], d[0]];
  const c = [outer[0] + d[0] * ew * 0.4, outer[1] + d[1] * ew * 0.4];
  const a = ew * 0.35, b = ew * 0.5;
  const P = (sa, sb) => [c[0] + d[0] * a * sa + n[0] * b * sb, c[1] + d[1] * a * sa + n[1] * b * sb];
  return hullMask(w, h, [P(-1, -1), P(1, -1), P(1, 1), P(-1, 1)]);
}

/** Massimo (dilate) o minimo (erode) su finestra quadrata k x k, separabile. */
function morph(src, w, h, k, isMax) {
  const r = k >> 1, tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  const op = isMax ? Math.max : Math.min;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let v = src[row + x];
      for (let dx = -r; dx <= r; dx++) { const xx = x + dx; if (xx >= 0 && xx < w) v = op(v, src[row + xx]); }
      tmp[row + x] = v;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = tmp[y * w + x];
      for (let dy = -r; dy <= r; dy++) { const yy = y + dy; if (yy >= 0 && yy < h) v = op(v, tmp[yy * w + x]); }
      out[y * w + x] = v;
    }
  }
  return out;
}

/** Zone del viso a partire dai landmark (gia' in pixel dell'immagine di lavoro). */
export function buildRegions(w, h, lm) {
  const P = idx => idx.map(i => lm[i]);
  const R = {};
  R.fronte = hullMask(w, h, P(IDX.FOREHEAD));
  R.guance = union(hullMask(w, h, P(IDX.CHEEK_R)), hullMask(w, h, P(IDX.CHEEK_L)));
  R.naso = hullMask(w, h, P(IDX.NOSE));
  R.mento = hullMask(w, h, P(IDX.CHIN));
  R.sotto_occhi = union(hullMask(w, h, P(IDX.UNDER_EYE_R)), hullMask(w, h, P(IDX.UNDER_EYE_L)));
  R.zampe_gallina = union(crowFeet(w, h, lm[33], lm[133]), crowFeet(w, h, lm[263], lm[362]));

  const excl0 = union(hullMask(w, h, P(IDX.EYE_R)), hullMask(w, h, P(IDX.EYE_L)),
    hullMask(w, h, P(IDX.BROW_R)), hullMask(w, h, P(IDX.BROW_L)),
    hullMask(w, h, P(IDX.LIPS)), hullMask(w, h, P(IDX.NOSTRILS)));
  const exclF = morph(Float32Array.from(excl0.m), w, h, 15, true);
  const oval = hullMask(w, h, P(IDX.FACE_OVAL));
  const skin = new Uint8Array(w * h);
  for (let i = 0; i < skin.length; i++) {
    const ex = exclF[i] > 0;
    skin[i] = oval.m[i] && !ex ? 1 : 0;
    if (ex) { R.sotto_occhi.m[i] = 0; R.zampe_gallina.m[i] = 0; }
  }
  R.pelle = { m: skin, polys: oval.polys };
  R.zona_T = union(R.fronte, R.naso, R.mento);
  return R;
}

/** Zone approssimate da un riquadro del viso (x, y, larghezza, altezza). */
export function buildRegionsFromBox(w, h, box) {
  const [fx0, fy0, fw, fh] = box;
  const rect = (a, b, c, d) => hullMask(w, h, [
    [fx0 + a * fw, fy0 + b * fh], [fx0 + c * fw, fy0 + b * fh],
    [fx0 + c * fw, fy0 + d * fh], [fx0 + a * fw, fy0 + d * fh]]);
  const R = {};
  R.fronte = rect(.22, .05, .78, .22);
  R.guance = union(rect(.14, .52, .36, .72), rect(.64, .52, .86, .72));
  R.naso = rect(.42, .38, .58, .66);
  R.mento = rect(.38, .88, .62, .98);
  R.sotto_occhi = union(rect(.20, .45, .40, .52), rect(.60, .45, .80, .52));
  R.zampe_gallina = union(rect(.04, .34, .14, .48), rect(.86, .34, .96, .48));
  const skin = new Uint8Array(w * h);
  const cx = fx0 + fw / 2, cy = fy0 + fh * .55, ax = fw * .45, ay = fh * .5;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2 <= 1) skin[y * w + x] = 1;
  }
  const excl = union(rect(.18, .30, .44, .44), rect(.56, .30, .82, .44), rect(.32, .74, .68, .86));
  for (let i = 0; i < skin.length; i++) if (excl.m[i]) skin[i] = 0;
  R.pelle = { m: skin, polys: [] };
  R.zona_T = union(R.fronte, R.naso, R.mento);
  return R;
}

// ---------------------------------------------------------------------------
// Elaborazione immagine
// ---------------------------------------------------------------------------
function gaussianBlur(src, w, h, sigma) {
  const r = Math.ceil(sigma * 3), k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); s += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  const clamp = (v, hi) => (v < 0 ? -v : v > hi ? 2 * hi - v : v); // riflessione ai bordi
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0;
    for (let i = -r; i <= r; i++) v += k[i + r] * src[y * w + clamp(x + i, w - 1)];
    tmp[y * w + x] = v;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0;
    for (let i = -r; i <= r; i++) v += k[i + r] * tmp[clamp(y + i, h - 1) * w + x];
    out[y * w + x] = v;
  }
  return out;
}

function blackhat(gray, w, h, k) {
  const closed = morph(morph(gray, w, h, k, true), w, h, k, false);
  const out = new Float32Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = closed[i] - gray[i];
  return out;
}

function srgbToLin(c) { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; }
const LIN = Float32Array.from({ length: 256 }, (_, i) => srgbToLin(i));
function fLab(t) { return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116; }

/** Canali derivati da RGBA: grigio, L (0-100), a*, V e S (0-255). */
function channels(rgba, w, h) {
  const n = w * h;
  const gray = new Float32Array(n), L = new Float32Array(n), A = new Float32Array(n);
  const V = new Float32Array(n), S = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    const R = LIN[r], G = LIN[g], B = LIN[b];
    const X = (0.412453 * R + 0.357580 * G + 0.180423 * B) / 0.950456;
    const Y = 0.212671 * R + 0.715160 * G + 0.072169 * B;
    const Z = (0.019334 * R + 0.119193 * G + 0.950227 * B) / 1.088754;
    const fy = fLab(Y);
    L[i] = Y > 0.008856 ? 116 * fy - 16 : 903.3 * Y;
    A[i] = 500 * (fLab(X) - fy);
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    V[i] = mx; S[i] = mx === 0 ? 0 : (255 * (mx - mn)) / mx;
  }
  return { gray, L, A, V, S };
}

/** Componenti connesse (8-vicini) di bin & mask, filtrate per area ed allungamento. */
function components(bin, mask, w, h, amin, amax, { maxElong = null, minElong = null } = {}) {
  const n = w * h, label = new Int32Array(n), out = new Uint8Array(n), stack = new Int32Array(n);
  let next = 0, count = 0;
  for (let s = 0; s < n; s++) {
    if (!bin[s] || !mask[s] || label[s]) continue;
    next++;
    let sp = 0, area = 0, x0 = w, y0 = h, x1 = 0, y1 = 0;
    stack[sp++] = s; label[s] = next;
    const pix = [];
    while (sp) {
      const p = stack[--sp], x = p % w, y = (p / w) | 0;
      pix.push(p); area++;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const q = yy * w + xx;
        if (bin[q] && mask[q] && !label[q]) { label[q] = next; stack[sp++] = q; }
      }
    }
    if (area < amin || area > amax) continue;
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    const el = Math.max(bw, bh) / Math.max(1, Math.min(bw, bh));
    if (maxElong && el > maxElong) continue;
    if (minElong && el < minElong) continue;
    count++;
    for (const p of pix) out[p] = 1;
  }
  return { count, mask: out };
}

function sel(arr, mask) { const v = []; for (let i = 0; i < arr.length; i++) if (mask[i]) v.push(arr[i]); return v; }
function quantile(values, q) {
  if (!values.length) return 0;
  const s = Float64Array.from(values).sort();
  const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
const median = v => quantile(v, 0.5);
function std(v) { if (!v.length) return 0; const m = v.reduce((a, b) => a + b, 0) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); }
const count = m => { let c = 0; for (let i = 0; i < m.length; i++) c += m[i]; return Math.max(1, c); };
const or = (...ms) => { const o = new Uint8Array(ms[0].length); for (const m of ms) for (let i = 0; i < o.length; i++) o[i] |= m[i]; return o; };

export function score(value, best, worst) {
  const t = (value - best) / (worst - best);
  return Math.max(0, Math.min(100, 100 * (1 - t)));
}

// ---------------------------------------------------------------------------
// Analisi
// ---------------------------------------------------------------------------
export function analyze(rgba, w, h, R) {
  const { gray, L, A, V, S } = channels(rgba, w, h);
  const skin = R.pelle.m, raw = {}, overlay = {};

  // Texture
  const g3 = gaussianBlur(gray, w, h, 3);
  const zone = or(R.guance.m, R.fronte.m);
  const hp = []; for (let i = 0; i < zone.length; i++) if (zone[i]) hp.push(gray[i] - g3[i]);
  raw.texture = std(hp);

  // Pori
  const bhS = blackhat(gray, w, h, 7);
  const pz = or(R.guance.m, R.naso.m);
  const poreBin = bhS.map(v => (v > 10 ? 1 : 0));
  const pores = components(poreBin, pz, w, h, 3, 45, { maxElong: 2.5 });
  raw.pori_per_10k_px = (pores.count / count(pz)) * 1e4;
  overlay.pori = pores.mask;

  // Punti neri
  const lowL = quantile(sel(L, skin), 0.15);
  const darkBin = new Uint8Array(w * h);
  for (let i = 0; i < darkBin.length; i++) darkBin[i] = bhS[i] > 18 && L[i] < lowL ? 1 : 0;
  const bk = components(darkBin, R.naso.m, w, h, 3, 60, { maxElong: 2.5 });
  raw.punti_neri_per_10k_px = (bk.count / count(R.naso.m)) * 1e4;
  overlay.punti_neri = bk.mask;

  // Rughe
  const bhL = blackhat(gray, w, h, 15);
  const wz = or(R.fronte.m, R.zampe_gallina.m, R.sotto_occhi.m);
  const wr = components(bhL.map(v => (v > 12 ? 1 : 0)), wz, w, h, 25, 5000, { minElong: 3 });
  raw["rughe_densita_%"] = (100 * count(wr.mask)) / count(wz);
  overlay.rughe = wr.mask;

  // Macchie (sfondo locale = sfocatura ampia)
  const Lbg = gaussianBlur(L, w, h, 12);
  const spotBin = new Uint8Array(w * h);
  for (let i = 0; i < spotBin.length; i++) spotBin[i] = Lbg[i] - L[i] > 7 ? 1 : 0;
  const sp = components(spotBin, skin, w, h, 40, 3000, { maxElong: 3 });
  raw["macchie_area_%"] = (100 * count(sp.mask)) / count(skin);
  overlay.macchie = sp.mask;

  // Rossori e acne
  const Abg = gaussianBlur(A, w, h, 12);
  const redBin = new Uint8Array(w * h);
  for (let i = 0; i < redBin.length; i++) redBin[i] = A[i] - Abg[i] > 5 ? 1 : 0;
  const reds = components(redBin, skin, w, h, 20, 6000);
  raw["rossori_area_%"] = (100 * count(reds.mask)) / count(skin);
  const acne = components(redBin, skin, w, h, 20, 400, { maxElong: 2 });
  raw.acne_per_10k_px = (acne.count / count(skin)) * 1e4;
  overlay.rossori = reds.mask;

  // Occhiaie
  raw.occhiaie_delta_L = median(sel(L, R.guance.m)) - median(sel(L, R.sotto_occhi.m));

  // Oleosita' (riflessi nella zona T)
  const vthr = Math.max(200, quantile(sel(V, skin), 0.97));
  const shine = new Uint8Array(w * h);
  for (let i = 0; i < shine.length; i++) shine[i] = V[i] >= vthr && S[i] < 70 && R.zona_T.m[i] ? 1 : 0;
  raw["lucidita_zonaT_%"] = (100 * count(shine)) / count(R.zona_T.m);
  overlay.lucidita = shine;

  // Luminosita' e uniformita'
  raw.luminosita_L = median(sel(L, skin));
  raw.disuniformita_std = std(sel(gaussianBlur(L, w, h, 8), skin));

  const scores = {
    "Rughe": score(raw["rughe_densita_%"], 0.5, 8),
    "Texture": score(raw.texture, 3, 10),
    "Pori": score(raw.pori_per_10k_px, 5, 60),
    "Punti neri": score(raw.punti_neri_per_10k_px, 2, 40),
    "Macchie": score(raw["macchie_area_%"], 0.5, 10),
    "Rossori": score(raw["rossori_area_%"], 1, 18),
    "Acne": score(raw.acne_per_10k_px, 0.5, 12),
    "Occhiaie": score(raw.occhiaie_delta_L, 2, 16),
    "Oleosità": score(raw["lucidita_zonaT_%"], 0.5, 12),
    "Luminosità": score(-raw.luminosita_L, -75, -35),
    "Uniformità": score(raw.disuniformita_std, 3, 14),
  };
  return { scores, raw, overlay, L };
}

export function lightingWarnings(L, w, h, skin) {
  let sx = 0, sy = 0, n = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (skin[y * w + x]) { sx += x; sy += y; n++; }
  const cx = sx / n, cy = sy / n;
  const left = [], right = [], top = [], bot = [], all = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x; if (!skin[i]) continue;
    all.push(L[i]); (x < cx ? left : right).push(L[i]); (y < cy ? top : bot).push(L[i]);
  }
  const warn = [];
  if (Math.abs(median(left) - median(right)) > 8) warn.push("Luce laterale: un lato del viso è più scuro dell'altro.");
  if (median(top) - median(bot) > 8) warn.push("Luce dall'alto: ombre su occhiaie e solchi (peggiora i punteggi).");
  const med = median(all);
  if (med < 40) warn.push("Foto scura: i punteggi saranno poco affidabili.");
  if (med > 85) warn.push("Foto sovraesposta: dettagli persi.");
  return warn;
}
