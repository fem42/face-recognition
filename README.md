# Analisi della pelle

Pagina web che analizza la pelle del viso da un selfie: rughe, texture, pori, punti neri,
macchie, rossori, acne, occhiaie, oleosità, luminosità e uniformità (punteggi 0–100, 100 = migliore).

- **Tutto gira nel browser**: la foto non viene caricata su nessun server.
- Il viso viene trovato con [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) (478 punti).
- Le misure usano visione artificiale classica (black-hat morfologico, componenti connesse, spazio colore Lab).
- Le soglie sono fissate a mano: i numeri servono a confrontare le **proprie** foto nel tempo, con la stessa luce.

Non è uno strumento diagnostico.

## File
- `index.html`, `app.js`: interfaccia
- `analysis.js`: misure (senza dipendenze)
- `python/skin_analyzer.py`: stessa analisi da riga di comando, con stima opzionale dell'età (insightface)

## Pubblicazione
Settings → Pages → Source: *Deploy from a branch* → `main` / `(root)`.
