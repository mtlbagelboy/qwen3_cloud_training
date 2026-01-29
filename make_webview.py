#!/usr/bin/env python3
"""
Generate a tiny local webview for browsing audio segments + transcripts.

Example:
  python3 make_webview.py --jsonl metadata.jsonl
  python3 -m http.server 8000
  open http://localhost:8000/webview/index.html
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _normalize_audio_path(p: str) -> str:
    # Store paths relative to repo root without leading "./" so the webview can
    # reference them consistently.
    p = p.strip()
    if p.startswith("./"):
        p = p[2:]
    return p

def _ensure_audio_symlink() -> None:
    """
    Make `webview/audio` a symlink to `../audio` so `webview/index*.html` can
    load audio via relative paths under `file://` without needing a local server.
    """
    webview_dir = Path("webview")
    webview_dir.mkdir(parents=True, exist_ok=True)
    link_path = webview_dir / "audio"
    target = Path("..") / "audio"

    try:
        if link_path.exists() or link_path.is_symlink():
            return
        os.symlink(str(target), str(link_path))
    except OSError as e:
        print(f"Warning: could not create symlink {link_path} -> {target}: {e}")

def _write_embedded_html(out_path: Path, samples: List[Dict[str, Any]]) -> None:
    template_path = Path("webview/index.html")
    html = template_path.read_text(encoding="utf-8")
    marker = "/*__SAMPLES_JSON__*/"
    if marker not in html:
        raise RuntimeError(f"Missing marker {marker} in {template_path}")
    payload = json.dumps({"samples": samples}, ensure_ascii=False)
    # Hardening: avoid accidental `</script>` termination or HTML parsing issues.
    # Embed-safe JSON escapes per common practice.
    payload = payload.replace("</", "<\\/")
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    html = html.replace(marker, payload)
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default="metadata.jsonl", help="Path to jsonl with {audio,text,...}")
    parser.add_argument("--out", type=str, default="webview/samples.json", help="Output json path")
    parser.add_argument("--embedded_out", type=str, default="webview/index_embedded.html", help="Standalone HTML output")
    parser.add_argument("--require_audio_exists", action="store_true", help="Drop entries whose audio file is missing")
    args = parser.parse_args()

    in_path = Path(args.jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    missing = 0
    for obj in _iter_jsonl(in_path):
        audio = obj.get("audio")
        text = obj.get("text")
        if not audio or not text:
            continue

        audio = _normalize_audio_path(str(audio))
        if args.require_audio_exists and not Path(audio).exists():
            missing += 1
            continue

        samples.append(
            {
                "audio": audio,
                "text": text,
                "language": obj.get("language"),
                "alignment_score": obj.get("alignment_score"),
            }
        )

    out_path.write_text(json.dumps({"samples": samples}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _ensure_audio_symlink()
    _write_embedded_html(Path(args.embedded_out), samples)

    print(f"Wrote {len(samples)} samples to {out_path}")
    print(f"Wrote standalone HTML to {args.embedded_out}")
    if missing:
        print(f"Dropped {missing} samples with missing audio (require_audio_exists=true)")
    print("Next:")
    print("  Option A (no server): open webview/index_embedded.html")
    print("  Option B (server): python3 -m http.server 8000  # then open http://localhost:8000/webview/index.html")


if __name__ == "__main__":
    main()
