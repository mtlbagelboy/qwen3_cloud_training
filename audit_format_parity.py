#!/usr/bin/env python3
"""
Audit helper: compare this repo's training-time input construction with the
installed qwen_tts inference code.

Run on your training box (e.g., RunPod) after installing deps via `setup.sh`.

Example:
  python audit_format_parity.py \
    --model_path ./Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl ./train_with_codes.jsonl \
    --codec_format inference
"""

import argparse
import inspect
import json
from typing import Iterable, List, Tuple

import torch

from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from transformers import AutoConfig


def _iter_snippets(source: str, patterns: Iterable[str], context: int = 6, max_snippets: int = 8) -> List[Tuple[str, str]]:
    lines = source.splitlines()
    out: List[Tuple[str, str]] = []
    for pat in patterns:
        for idx, line in enumerate(lines):
            if pat not in line:
                continue
            start = max(0, idx - context)
            end = min(len(lines), idx + context + 1)
            snippet = "\n".join(lines[start:end])
            out.append((pat, snippet))
            if len(out) >= max_snippets:
                return out
    return out


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--codec_format", type=str, default="inference", choices=["inference", "teacher_forcing"])
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--dump_inference_source", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    _print_header("Load config + processor")
    model = Qwen3TTSModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="cpu",
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(args.model_path)

    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if not (0 <= args.sample_idx < len(lines)):
        raise SystemExit(f"--sample_idx out of range: {args.sample_idx} (n={len(lines)})")
    sample = lines[args.sample_idx]

    ds = TTSDataset([sample], model.processor, config, language=args.language, codec_format=args.codec_format)
    assistant_text = ds._build_assistant_text(sample["text"])
    text_ids = ds._tokenize_texts(assistant_text)[:, :-5]
    audio_codes = torch.tensor(sample["audio_codes"], dtype=torch.long)
    fake_ref_mel = torch.zeros((1, 1, 128), dtype=torch.float32)

    batch = ds.collate_fn([{"text_ids": text_ids, "audio_codes": audio_codes, "ref_mel": fake_ref_mel}])
    input_ids = batch["input_ids"][0]
    attention_mask = batch["attention_mask"][0].bool()
    codec_embedding_mask = batch["codec_embedding_mask"][0].squeeze(-1)
    codec_mask = batch["codec_mask"][0]
    codec_0_labels = batch["codec_0_labels"][0]

    speaker_pos = ((~codec_embedding_mask) & attention_mask).nonzero().flatten().tolist()
    _print_header("Training-time format summary")
    print(f"codec_format: {args.codec_format}")
    print(f"seq_len (attention true): {int(attention_mask.sum().item())}")
    print(f"speaker_pos (inferred): {speaker_pos}")
    if len(speaker_pos) == 1:
        sp = speaker_pos[0]
        print(f"codec token at speaker_pos: {int(input_ids[sp, 1].item())} (expected placeholder 0)")

    first_audio_pos = codec_mask.nonzero().flatten().min().item() if codec_mask.any() else None
    last_audio_pos = codec_mask.nonzero().flatten().max().item() if codec_mask.any() else None
    print(f"first_audio_codec_pos (codec_mask True): {first_audio_pos}")
    print(f"last_audio_codec_pos  (codec_mask True): {last_audio_pos}")

    # Show the fixed codec prefix region (pos 0..15) to make mismatches obvious.
    _print_header("Token window (positions 0..24)")
    max_pos = min(25, input_ids.shape[0])
    for pos in range(max_pos):
        if not attention_mask[pos]:
            break
        txt = int(input_ids[pos, 0].item())
        c0 = int(input_ids[pos, 1].item())
        lab = int(codec_0_labels[pos].item())
        cm = bool(codec_mask[pos].item())
        cem = bool(codec_embedding_mask[pos].item())
        print(f"{pos:02d}  text={txt:6d}  codec0={c0:6d}  label={lab:6d}  codec_mask={int(cm)}  codec_emb_mask={int(cem)}")

    _print_header("Embedding path check (inference vs training)")
    talker = model.model.talker
    has_proj = hasattr(talker, "text_projection") and talker.text_projection is not None
    print(f"talker.text_projection present: {has_proj}")
    if has_proj:
        with torch.no_grad():
            ids = torch.randint(low=0, high=100, size=(1, 8), dtype=torch.long)
            emb = talker.model.text_embedding(ids)
            proj = talker.text_projection(emb)
            diff = (proj - emb).abs().mean().item()
            print(f"mean(|text_projection(text_emb)-text_emb|): {diff:.6f}")

    if args.dump_inference_source:
        _print_header("Inference source snippets (qwen_tts)")
        patterns = [
            "codec_think_id",
            "codec_think_bos_id",
            "codec_think_eos_id",
            "codec_language_id",
            "codec_bos_id",
            "spk_id",
            "text_projection",
            "codec_embedding",
        ]

        for fn in [
            getattr(Qwen3TTSModel, "generate_custom_voice", None),
            getattr(Qwen3TTSModel, "generate_voice_clone", None),
            getattr(Qwen3TTSModel, "create_voice_clone_prompt", None),
        ]:
            if fn is None:
                continue
            try:
                src = inspect.getsource(fn)
            except OSError:
                continue
            print(f"\n--- {fn.__qualname__} ---")
            for pat, snippet in _iter_snippets(src, patterns):
                print(f"\n[match: {pat}]")
                print(snippet)


if __name__ == "__main__":
    main()

