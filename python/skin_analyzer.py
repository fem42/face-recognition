#!/usr/bin/env python3
"""
skin_analyzer.py - Analisi della pelle da una foto (approssimazione "fai da te"
degli specchi tipo Inovshop / Perfect Corp).

Uso:
    python skin_analyzer.py foto.jpg
    python skin_analyzer.py foto.jpg --out report.png

Dipendenze (obbligatorie):
    pip install opencv-python numpy matplotlib mediapipe
Facoltativa, per la stima dell'eta:
    pip install insightface onnxruntime

Come funziona:
  1. Trova il viso e 478 punti di riferimento (MediaPipe). Se MediaPipe non e'
     disponibile usa un rilevatore piu' semplice di OpenCV con zone approssimate.
  2. Divide il viso in zone: fronte, guance, naso, contorno occhi, zampe di
     gallina, mento.
  3. Misura ogni voce con visione artificiale classica (niente "magia").
  4. Converte ogni misura in un punteggio 0-100 (100 = migliore). Le soglie
     sono fissate a mano: i numeri hanno senso soprattutto per CONFRONTARE foto
     tue fatte con la stessa luce, non come valore assoluto.
  5. Se insightface e' installato stima l'eta' apparente del viso con una rete
     neurale e la corregge leggermente in base ai punteggi legati all'eta'.

Voci NON misurate (idratazione, elasticita', palpebre calanti): da una foto
non si misurano davvero; anche gli specchi commerciali le "stimano".
"""

import argparse
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Indici dei landmark MediaPipe Face Mesh (478 punti)
# --------------------------------------------------------------------------
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FOREHEAD = [54, 103, 67, 109, 10, 338, 297, 332, 284,
            300, 293, 334, 296, 336, 107, 66, 105, 63, 70]
CHEEK_R = [116, 117, 118, 119, 100, 142, 203, 206, 216, 192, 213, 147, 123]
CHEEK_L = [345, 346, 347, 348, 329, 371, 423, 426, 436, 416, 433, 376, 352]
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 64, 294, 48, 278,
        115, 344, 129, 358]
CHIN = [152, 148, 176, 149, 150, 377, 400, 378, 379, 199, 200, 201, 421,
        208, 428]
UNDER_EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133,
               111, 117, 118, 119, 120, 121, 128]
UNDER_EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362,
               340, 346, 347, 348, 349, 350, 357]
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160,
         161, 246]
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386,
         387, 388, 466]
BROW_R = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_L = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
        267, 0, 37, 39, 40, 185]
NOSTRILS = [98, 327, 2, 64, 294, 240, 460, 97, 326]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

WORK_FACE_WIDTH = 600  # il viso viene riscalato a questa larghezza (px)


# --------------------------------------------------------------------------
# 1. Rilevamento del viso
# --------------------------------------------------------------------------
def _landmarks_mediapipe(img_bgr):
    """Restituisce un array (478, 2) di coordinate in pixel oppure None."""
    try:
        import mediapipe as mp
    except ImportError:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Vecchia API (mediapipe <= 0.10.14 circa)
    if hasattr(mp, "solutions"):
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                             max_num_faces=1,
                                             refine_landmarks=True) as fm:
            res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        pts = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in pts], dtype=np.float32)

    # Nuova API "Tasks": serve il file del modello (scaricato una volta)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, "face_landmarker.task")
    if not os.path.exists(model_path):
        print("Scarico il modello MediaPipe (circa 4 MB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
        except Exception as e:  # rete assente ecc.
            print(f"  Download non riuscito ({e}). Uso il rilevatore semplice.")
            return None
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_faces=1)
    with vision.FaceLandmarker.create_from_options(opts) as det:
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]],
                    dtype=np.float32)


def _face_box_haar(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    path = os.path.join(getattr(getattr(cv2, "data", None), "haarcascades", ""),
                        "haarcascade_frontalface_default.xml")
    casc = cv2.CascadeClassifier(path)
    if casc.empty():
        return None
    faces = casc.detectMultiScale(gray, 1.1, 6, minSize=(120, 120))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])  # il viso piu' grande


def _hull_mask(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    hull = cv2.convexHull(np.asarray(pts, np.int32))
    cv2.fillConvexPoly(m, hull, 255)
    return m


def _crow_feet_mask(shape, outer, inner):
    """Zona laterale all'angolo esterno dell'occhio."""
    outer, inner = np.asarray(outer, float), np.asarray(inner, float)
    eye_w = np.linalg.norm(outer - inner)
    d = (outer - inner) / (eye_w + 1e-6)          # direzione verso l'esterno
    n = np.array([-d[1], d[0]])                  # perpendicolare
    c = outer + d * eye_w * 0.40
    a, b = eye_w * 0.35, eye_w * 0.50
    pts = [c - d * a - n * b, c + d * a - n * b, c + d * a + n * b,
           c - d * a + n * b]
    return _hull_mask(shape, pts)


def build_regions(img_bgr, manual_box=None):
    """Ritaglia il viso, lo riscala e costruisce le maschere delle zone.

    Restituisce (immagine_viso, dict_maschere, metodo)."""
    lm = None if manual_box else _landmarks_mediapipe(img_bgr)
    method = "mediapipe"
    if lm is not None:
        x0, y0 = lm.min(0)
        x1, y1 = lm.max(0)
    else:
        box = manual_box or _face_box_haar(img_bgr)
        if box is None:
            raise SystemExit("Nessun viso trovato nella foto. Prova con --box x,y,larghezza,altezza")
        method = "manuale" if manual_box else "haar"
        method += " (zone approssimate)"
        x, y, w, h = box
        # il riquadro Haar taglia la fronte: lo allunghiamo verso l'alto
        x0, y0, x1, y1 = x, y - 0.12 * h, x + w, y + h

    # ritaglio con margine e riscalatura a larghezza fissa
    mw, mh = (x1 - x0) * 0.12, (y1 - y0) * 0.12
    H, W = img_bgr.shape[:2]
    cx0, cy0 = int(max(0, x0 - mw)), int(max(0, y0 - mh))
    cx1, cy1 = int(min(W, x1 + mw)), int(min(H, y1 + mh))
    crop = img_bgr[cy0:cy1, cx0:cx1]
    s = WORK_FACE_WIDTH / (x1 - x0)
    face = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    shp = face.shape

    R = {}
    if lm is not None:
        p = (lm - [cx0, cy0]) * s
        P = lambda idx: p[idx]
        R["fronte"] = _hull_mask(shp, P(FOREHEAD))
        R["guance"] = _hull_mask(shp, P(CHEEK_R)) | _hull_mask(shp, P(CHEEK_L))
        R["naso"] = _hull_mask(shp, P(NOSE))
        R["mento"] = _hull_mask(shp, P(CHIN))
        R["sotto_occhi"] = (_hull_mask(shp, P(UNDER_EYE_R)) |
                            _hull_mask(shp, P(UNDER_EYE_L)))
        R["zampe_gallina"] = (_crow_feet_mask(shp, p[33], p[133]) |
                              _crow_feet_mask(shp, p[263], p[362]))
        excl = (_hull_mask(shp, P(EYE_R)) | _hull_mask(shp, P(EYE_L)) |
                _hull_mask(shp, P(BROW_R)) | _hull_mask(shp, P(BROW_L)) |
                _hull_mask(shp, P(LIPS)) | _hull_mask(shp, P(NOSTRILS)))
        excl = cv2.dilate(excl, np.ones((15, 15), np.uint8))
        skin = _hull_mask(shp, P(FACE_OVAL)) & ~excl
        # le zone sotto gli occhi non devono includere l'occhio
        R["sotto_occhi"] &= ~excl
        R["zampe_gallina"] &= ~excl
    else:
        fx0, fy0 = (x0 - cx0) * s, (y0 - cy0) * s
        fw, fh = (x1 - x0) * s, (y1 - y0) * s

        def rect(a, b, c, d):  # frazioni del riquadro viso
            m = np.zeros(shp[:2], np.uint8)
            cv2.rectangle(m, (int(fx0 + a * fw), int(fy0 + b * fh)),
                          (int(fx0 + c * fw), int(fy0 + d * fh)), 255, -1)
            return m
        R["fronte"] = rect(.22, .05, .78, .22)
        R["guance"] = rect(.14, .52, .36, .72) | rect(.64, .52, .86, .72)
        R["naso"] = rect(.42, .38, .58, .66)
        R["mento"] = rect(.38, .88, .62, .98)
        R["sotto_occhi"] = rect(.20, .45, .40, .52) | rect(.60, .45, .80, .52)
        R["zampe_gallina"] = rect(.04, .34, .14, .48) | rect(.86, .34, .96, .48)
        skin = np.zeros(shp[:2], np.uint8)
        cv2.ellipse(skin, (int(fx0 + fw / 2), int(fy0 + fh * .55)),
                    (int(fw * .45), int(fh * .5)), 0, 0, 360, 255, -1)
        excl = (rect(.18, .30, .44, .44) | rect(.56, .30, .82, .44) |
                rect(.32, .74, .68, .86))
        skin &= ~excl
    R["pelle"] = skin
    R["zona_T"] = R["fronte"] | R["naso"] | R["mento"]
    return face, R, method


# --------------------------------------------------------------------------
# 2. Misure
# --------------------------------------------------------------------------
def score(value, best, worst):
    """Mappa lineare: valore 'best' -> 100, 'worst' -> 0 (con clip)."""
    t = (value - best) / (worst - best)
    return float(np.clip(100 * (1 - t), 0, 100))


def _components(binary, mask, amin, amax, max_elong=None, min_elong=None):
    """Componenti connesse dentro la maschera, filtrate per area/forma."""
    b = (binary > 0).astype(np.uint8) & (mask > 0).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(b, 8)
    keep = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if not (amin <= a <= amax):
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        el = max(w, h) / max(1, min(w, h))
        if max_elong and el > max_elong:
            continue
        if min_elong and el < min_elong:
            continue
        keep.append((i, a, cents[i]))
    out = np.zeros_like(b)
    for i, _, _ in keep:
        out[lab == i] = 1
    return keep, out


def analyze(face, R):
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A = lab[..., 0] * 100 / 255, lab[..., 1] - 128   # L 0-100, a* +/-
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    skin = R["pelle"] > 0
    area = lambda m: max(1, int((m > 0).sum()))
    raw, overlay = {}, {}

    # --- Texture: irregolarita' fine della superficie -----------------------
    hp = gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 3).astype(np.float32)
    zone = (R["guance"] | R["fronte"]) > 0
    raw["texture"] = float(hp[zone].std())

    # --- Pori: piccoli punti scuri e rotondi su guance e naso ---------------
    bh_small = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    pz = R["guance"] | R["naso"]
    pores, pmask = _components(bh_small > 10, pz, 3, 45, max_elong=2.5)
    raw["pori_per_10k_px"] = len(pores) / area(pz) * 1e4
    overlay["pori"] = pmask

    # --- Punti neri: pori molto scuri sul naso -----------------------------
    dark = (bh_small > 18) & (L < np.percentile(L[skin], 15))
    bk, bkmask = _components(dark.astype(np.uint8), R["naso"], 3, 60,
                             max_elong=2.5)
    raw["punti_neri_per_10k_px"] = len(bk) / area(R["naso"]) * 1e4
    overlay["punti_neri"] = bkmask

    # --- Rughe: linee scure sottili e allungate ----------------------------
    bh_line = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    wz = R["fronte"] | R["zampe_gallina"] | R["sotto_occhi"]
    _, wmask = _components(bh_line > 12, wz, 25, 5000, min_elong=3.0)
    raw["rughe_densita_%"] = 100 * wmask.sum() / area(wz)
    overlay["rughe"] = wmask

    # --- Macchie: zone piu' scure dello sfondo locale ----------------------
    Lbg = cv2.medianBlur(L.astype(np.uint8), 41).astype(np.float32)
    spots, smask = _components(((Lbg - L) > 7).astype(np.uint8), R["pelle"],
                               40, 3000, max_elong=3.0)
    raw["macchie_area_%"] = 100 * smask.sum() / area(R["pelle"])
    overlay["macchie"] = smask

    # --- Rossori / acne: zone piu' rosse dello sfondo locale ---------------
    Ab = A + 128
    Abg = cv2.medianBlur(np.clip(Ab, 0, 255).astype(np.uint8), 41).astype(np.float32)
    redd = ((Ab - Abg) > 5).astype(np.uint8)
    reds, rmask = _components(redd, R["pelle"], 20, 6000)
    raw["rossori_area_%"] = 100 * rmask.sum() / area(R["pelle"])
    acne = [c for c in reds if c[1] <= 400]
    acne_mask = np.zeros_like(rmask)
    _, acne_mask = _components(redd, R["pelle"], 20, 400, max_elong=2.0)
    raw["acne_per_10k_px"] = len([1 for _ in acne]) / area(R["pelle"]) * 1e4
    overlay["rossori"] = rmask

    # --- Occhiaie: quanto il sottocchio e' piu' scuro della guancia --------
    raw["occhiaie_delta_L"] = float(np.median(L[R["guance"] > 0]) -
                                    np.median(L[R["sotto_occhi"] > 0]))

    # --- Oleosita': riflessi speculari nella zona T ------------------------
    V, S = hsv[..., 2].astype(np.float32), hsv[..., 1].astype(np.float32)
    vthr = max(200, np.percentile(V[skin], 97))
    shine = (V >= vthr) & (S < 70) & (R["zona_T"] > 0)
    raw["lucidita_zonaT_%"] = 100 * shine.sum() / area(R["zona_T"])
    overlay["lucidita"] = shine.astype(np.uint8)

    # --- Luminosita' e uniformita' del colorito ----------------------------
    raw["luminosita_L"] = float(np.median(L[skin]))
    Lsmooth = cv2.GaussianBlur(L, (0, 0), 8)
    raw["disuniformita_std"] = float(Lsmooth[skin].std())

    # Soglie empiriche (da tarare sulle tue foto!)
    scores = {
        "Rughe":        score(raw["rughe_densita_%"], 0.5, 8),
        "Texture":      score(raw["texture"], 3.0, 10),
        "Pori":         score(raw["pori_per_10k_px"], 5, 60),
        "Punti neri":   score(raw["punti_neri_per_10k_px"], 2, 40),
        "Macchie":      score(raw["macchie_area_%"], 0.5, 10),
        "Rossori":      score(raw["rossori_area_%"], 1, 18),
        "Acne":         score(raw["acne_per_10k_px"], 0.5, 12),
        "Occhiaie":     score(raw["occhiaie_delta_L"], 2, 16),
        "Oleosita'":    score(raw["lucidita_zonaT_%"], 0.5, 12),
        "Luminosita'":  score(-raw["luminosita_L"], -75, -35),
        "Uniformita'":  score(raw["disuniformita_std"], 3, 14),
    }
    return scores, raw, overlay


# --------------------------------------------------------------------------
# 3. Eta'
# --------------------------------------------------------------------------
AGING_KEYS = ["Rughe", "Texture", "Macchie", "Occhiaie", "Uniformita'"]


def estimate_age(img_bgr, scores):
    """Eta' apparente con insightface + piccola correzione dai punteggi."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        return None, None
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "genderage"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    faces = app.get(img_bgr)
    if not faces:
        return None, None
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    apparent = float(f.age)
    # correzione arbitraria: punteggio medio "di invecchiamento" 75 = neutro,
    # ogni 5 punti sopra/sotto sposta di 1 anno, max +/-5
    aging = np.mean([scores[k] for k in AGING_KEYS])
    skin_age = apparent + float(np.clip((75 - aging) / 5, -5, 5))
    return round(apparent), round(skin_age)


# --------------------------------------------------------------------------
# 4. Controllo della luce
# --------------------------------------------------------------------------
def lighting_warnings(face, R):
    L = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * 100 / 255
    skin = R["pelle"] > 0
    h, w = skin.shape
    xs = np.arange(w)[None, :].repeat(h, 0)
    cx = xs[skin].mean()
    left, right = np.median(L[skin & (xs < cx)]), np.median(L[skin & (xs >= cx)])
    ys = np.arange(h)[:, None].repeat(w, 1)
    cy = ys[skin].mean()
    top, bot = np.median(L[skin & (ys < cy)]), np.median(L[skin & (ys >= cy)])
    warn = []
    if abs(left - right) > 8:
        warn.append("Luce laterale: un lato del viso e' piu' scuro dell'altro.")
    if top - bot > 8:
        warn.append("Luce dall'alto: ombre su occhiaie e solchi (peggiora i punteggi).")
    med = np.median(L[skin])
    if med < 40:
        warn.append("Foto scura: i punteggi saranno poco affidabili.")
    if med > 85:
        warn.append("Foto sovraesposta: dettagli persi.")
    return warn


# --------------------------------------------------------------------------
# 5. Report grafico
# --------------------------------------------------------------------------
def make_report(face, R, overlay, scores, ages, warnings, method, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis = face.copy()
    tint = vis.copy()
    tint[R["pelle"] > 0] = (tint[R["pelle"] > 0] * 0.85 +
                            np.array([255, 200, 230]) * 0.15).astype(np.uint8)
    vis = tint
    colors = {"pori": (0, 140, 255), "punti_neri": (40, 40, 40),
              "rughe": (255, 80, 200), "macchie": (0, 90, 160),
              "rossori": (60, 60, 255), "lucidita": (255, 255, 0)}
    for k, c in colors.items():
        m = overlay.get(k)
        if m is not None:
            vis[m > 0] = c
    for k in ["fronte", "guance", "naso", "mento", "sotto_occhi", "zampe_gallina"]:
        cnts, _ = cv2.findContours(R[k], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 255), 2)

    labels = list(scores.keys())
    vals = [scores[k] for k in labels]
    overall = round(float(np.mean(vals)))

    fig = plt.figure(figsize=(13, 7.5), dpi=110)
    ax0 = fig.add_axes([0.02, 0.08, 0.40, 0.84])
    ax0.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax0.axis("off")
    legend = "  ".join(["arancio=pori", "rosa=rughe", "rosso=rossori",
                        "marrone=macchie", "azzurro=lucidita'"])
    ax0.set_title(legend, fontsize=8)

    fig.text(0.46, 0.94, "IL TUO REPORT PELLE", fontsize=15, weight="bold")
    sub = f"Punteggio pelle {overall}"
    if ages[0] is not None:
        sub += f"     Eta' pelle {ages[1]}  (eta' apparente {ages[0]})"
    else:
        sub += "     Eta': installa insightface per stimarla"
    fig.text(0.46, 0.895, sub, fontsize=12)

    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ax = fig.add_axes([0.50, 0.12, 0.42, 0.74], polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    v = vals + vals[:1]
    a = ang + ang[:1]
    ax.plot(a, v, color="#4a6fa5", lw=2)
    ax.fill(a, v, color="#4a6fa5", alpha=0.25)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color="grey")
    ax.set_xticks(ang)
    ax.set_xticklabels([f"{k}\n{round(s)}" for k, s in zip(labels, vals)],
                       fontsize=10)

    foot = f"Metodo viso: {method}. Punteggi indicativi: confronta solo foto con la stessa luce."
    if warnings:
        foot = "ATTENZIONE: " + " ".join(warnings) + "\n" + foot
    fig.text(0.46, 0.02, foot, fontsize=8.5, color="#a33" if warnings else "grey")
    fig.savefig(out_path)
    plt.close(fig)
    return overall


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Analisi della pelle da foto")
    ap.add_argument("foto")
    ap.add_argument("--out", default=None, help="file PNG del report")
    ap.add_argument("--box", default=None,
                    help="riquadro del viso x,y,larghezza,altezza (se il viso non viene trovato)")
    args = ap.parse_args()

    img = cv2.imread(args.foto)
    if img is None:
        sys.exit(f"Impossibile leggere {args.foto}")

    box = tuple(int(v) for v in args.box.split(",")) if args.box else None
    face, R, method = build_regions(img, box)
    scores, raw, overlay = analyze(face, R)
    warnings = lighting_warnings(face, R)
    ages = estimate_age(img, scores)

    base = os.path.splitext(args.out or (os.path.splitext(args.foto)[0] + "_report.png"))[0]
    out_png = base + ".png"
    overall = make_report(face, R, overlay, scores, ages, warnings, method, out_png)

    result = {"punteggio": overall, "eta_apparente": ages[0], "eta_pelle": ages[1],
              "punteggi": {k: round(v) for k, v in scores.items()},
              "misure_grezze": {k: round(v, 3) for k, v in raw.items()},
              "avvisi_luce": warnings, "metodo_viso": method}
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nPunteggio pelle: {overall}")
    if ages[0] is not None:
        print(f"Eta' pelle: {ages[1]}  (eta' apparente del viso: {ages[0]})")
    for k, v in scores.items():
        print(f"  {k:<13} {round(v):>3}")
    for w in warnings:
        print("  ! " + w)
    print(f"\nReport: {out_png}\nDati:   {base}.json")


if __name__ == "__main__":
    main()
