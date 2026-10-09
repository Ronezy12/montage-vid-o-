#!/usr/bin/env python3
"""Audit des rushs : inventaire technique, doublons, netteté, exposition,
planches-contact et analyse musicale (tempo, beats, montées d'énergie).

Usage : python3 scripts/audit.py [dossier_rushs] [dossier_rapports]
Les fichiers sources ne sont jamais modifiés (lecture seule).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".m2ts", ".mxf", ".webm"}
AUDIO_EXT = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".aiff", ".aif", ".ogg"}
SAMPLES_PER_RUSH = 24  # images analysées par rush


def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def quick_hash(path, chunk=4 << 20):
    """Empreinte rapide : taille + début + fin du fichier (détection de doublons)."""
    h = hashlib.sha1(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        f.seek(max(0, path.stat().st_size - chunk))
        h.update(f.read(chunk))
    return h.hexdigest()


def parse_fps(rate):
    num, _, den = rate.partition("/")
    return float(num) / float(den or 1) if float(den or 1) else 0.0


def video_info(probe):
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w, h = int(v["width"]), int(v["height"])
    rotation = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    rotation = int(v.get("tags", {}).get("rotate", rotation))
    if abs(rotation) % 180 == 90:
        w, h = h, w
    return {
        "codec": v.get("codec_name"),
        "largeur": w,
        "hauteur": h,
        "rotation": rotation,
        "orientation": "vertical" if h > w else "horizontal" if w > h else "carré",
        "fps": round(parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate", "0/1")), 3),
        "pix_fmt": v.get("pix_fmt"),
        "transfert": v.get("color_transfer"),  # ex. arib-std-b67 = HLG (iPhone HDR)
    }


def analyse_frames(path, duration):
    """Netteté (variance du laplacien), luminance, saturation et mouvement sur N images."""
    cap = cv2.VideoCapture(str(path))
    stats, prev = [], None
    for i in range(SAMPLES_PER_RUSH):
        t = duration * (i + 0.5) / SAMPLES_PER_RUSH
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        motion = float(np.mean(cv2.absdiff(gray, prev))) if prev is not None and prev.shape == gray.shape else None
        prev = gray
        stats.append({
            "t": round(t, 2),
            "nettete": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
            "luminance": round(float(gray.mean()), 1),
            "noirs_ecrases_pct": round(float((gray < 8).mean() * 100), 1),
            "blancs_brules_pct": round(float((gray > 247).mean() * 100), 1),
            "saturation": round(float(hsv[..., 1].mean()), 1),
            "diff_image_suivante": round(motion, 1) if motion is not None else None,
        })
    cap.release()
    return stats


def contact_sheet(path, duration, out_png, cols=6, rows=4):
    """Planche-contact horodatée : cols x rows images réparties sur toute la durée."""
    n = cols * rows
    step = max(duration / n, 0.04)
    vf = (
        f"fps=1/{step:.3f},scale=320:-2,"
        "drawtext=text='%{pts\\:hms}':x=6:y=6:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6,"
        f"tile={cols}x{rows}:padding=4:color=black"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vf", vf, "-frames:v", "1", str(out_png)],
        check=True,
    )


def analyse_audio(path):
    """Tempo, beats, courbe d'énergie et candidats montées/drops (détection réelle, pas d'invention)."""
    import librosa

    sr = 22050
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    y = np.frombuffer(raw, dtype=np.float32)
    if y.size == 0 or float(np.abs(y).max()) < 1e-4:
        return {"present": False}
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=512)
    # Énergie lissée sur ~2 s, puis on repère les fortes hausses (montée -> drop)
    win = max(1, int(2 * sr / 512))
    smooth = np.convolve(rms, np.ones(win) / win, mode="same")
    db = 20 * np.log10(np.maximum(smooth, 1e-6))
    jump = db[win:] - db[:-win]
    jump[:win] = 0  # le lissage crée une fausse « montée » au début et une chute à la fin
    jump[-win:] = 0
    candidates = []
    for idx in np.argsort(jump)[::-1]:
        t = float(times[idx + win])
        if jump[idx] < 4:  # hausse < 4 dB : pas un événement marquant
            break
        if all(abs(t - c["t"]) > 8 for c in candidates):
            candidates.append({"t": round(t, 2), "hausse_db": round(float(jump[idx]), 1)})
        if len(candidates) >= 8:
            break
    peak = float(np.abs(y).max())
    return {
        "present": True,
        "tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 1),
        "nb_beats": len(beats),
        "beats": [round(float(b), 3) for b in beats],
        "crete_dbfs": round(20 * np.log10(peak), 1),
        "risque_saturation": peak > 0.99,
        "hausses_energie": sorted(candidates, key=lambda c: c["t"]),
        "energie_par_seconde_db": [
            round(float(db[np.searchsorted(times, s)]), 1) for s in range(int(times[-1]))
        ],
    }


def loudness(path):
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "ebur128", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    for line in r.splitlines()[::-1]:
        if line.strip().startswith("I:"):
            return float(line.split()[1])
    return None


def main():
    root = Path(__file__).resolve().parent.parent
    rush_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "rushs"
    rep_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "rapports"
    sheets = rep_dir / "planches"
    sheets.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in rush_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT | AUDIO_EXT)
    if not files:
        sys.exit(f"Aucun média trouvé dans {rush_dir}")

    inventory, seen = [], {}
    for p in files:
        print(f"→ {p.name}", flush=True)
        probe = ffprobe(p)
        dur = float(probe["format"].get("duration", 0))
        entry = {
            "fichier": str(p.relative_to(rush_dir)),
            "taille_mo": round(p.stat().st_size / 1e6, 1),
            "duree_s": round(dur, 2),
            "date_creation": probe["format"].get("tags", {}).get("creation_time"),
        }
        digest = quick_hash(p)
        entry["doublon_de"] = seen.get(digest)
        seen.setdefault(digest, entry["fichier"])

        has_video = any(s["codec_type"] == "video" and s.get("disposition", {}).get("attached_pic") != 1
                        for s in probe["streams"])
        has_audio = any(s["codec_type"] == "audio" for s in probe["streams"])
        entry["type"] = "video" if has_video else "audio"
        if has_video:
            entry.update(video_info(probe))
            frames = analyse_frames(p, dur)
            entry["images"] = frames
            if frames:
                sharp = [f["nettete"] for f in frames]
                entry["nettete_mediane"] = float(np.median(sharp))
                entry["luminance_moyenne"] = round(float(np.mean([f["luminance"] for f in frames])), 1)
                motions = [f["diff_image_suivante"] for f in frames if f["diff_image_suivante"] is not None]
                entry["variation_moyenne"] = round(float(np.mean(motions)), 1) if motions else None
            sheet = sheets / f"{p.stem}.png"
            contact_sheet(p, dur, sheet)
            entry["planche_contact"] = str(sheet.relative_to(rep_dir))
        if has_audio:
            entry["loudness_lufs"] = loudness(p)
            entry["audio"] = analyse_audio(p)
        inventory.append(entry)

    (rep_dir / "inventaire.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False))
    write_markdown(inventory, rep_dir / "inventaire.md")
    print(f"\nInventaire : {rep_dir / 'inventaire.md'}")


def write_markdown(inv, out):
    lines = ["# Inventaire des rushs", "",
             "| Fichier | Type | Durée | Résolution | Orientation | FPS | Netteté méd. | Lum. | LUFS | BPM | Doublon |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in inv:
        audio = e.get("audio") or {}
        lines.append("| {} | {} | {:.1f} s | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            e["fichier"], e["type"], e["duree_s"],
            f"{e['largeur']}×{e['hauteur']}" if "largeur" in e else "—",
            e.get("orientation", "—"), e.get("fps", "—"), e.get("nettete_mediane", "—"),
            e.get("luminance_moyenne", "—"), e.get("loudness_lufs", "—"),
            audio.get("tempo_bpm", "—"), e.get("doublon_de") or "",
        ))
    lines += ["", "## Événements musicaux détectés (hausses d'énergie ≥ 4 dB sur 2 s)", ""]
    for e in inv:
        audio = e.get("audio") or {}
        if audio.get("present"):
            ev = ", ".join(f"{c['t']} s (+{c['hausse_db']} dB)" for c in audio["hausses_energie"]) or "aucune"
            sat = " ⚠ crête proche de 0 dBFS" if audio["risque_saturation"] else ""
            lines.append(f"- **{e['fichier']}** : {audio['tempo_bpm']} BPM, crête {audio['crete_dbfs']} dBFS{sat} — {ev}")
    lines += ["", "_Netteté = variance du laplacien (plus haut = plus net ; à comparer entre rushs d'une même caméra)._", ""]
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
