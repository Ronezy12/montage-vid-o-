# Montage vidéo — DJ RON

Montage vertical 9:16 (1080×1920, H.264/AAC) pour Reels, TikTok et Shorts.

## Organisation

| Dossier | Contenu |
|---|---|
| `rushs/` | Médias sources originaux (jamais modifiés, non versionnés) |
| `projet/` | Fichiers de travail (copies, proxys) |
| `xml/` | Timelines exportables (FCPXML pour DaVinci Resolve) |
| `exports/` | Rendus finaux |
| `scripts/` | Outils d'automatisation |
| `rapports/` | Inventaire, planches-contact, contrôles qualité |

## Étape 1 — Audit des rushs

```bash
pip install -r requirements.txt
python3 scripts/audit.py            # lit rushs/, écrit rapports/
```

Produit `rapports/inventaire.md` / `.json` : durée, résolution, orientation,
FPS, HDR, doublons, netteté, exposition, loudness (LUFS), tempo (BPM), beats
et hausses d'énergie (montées/drops), plus une planche-contact horodatée par rush.
