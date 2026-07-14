#!/usr/bin/env python3
"""Convert Qwen/Qwen3.6-35B-A3B (and Qwen3-MoE family) to the colibri int8 container.

Downloads or converts a local Qwen3-MoE checkpoint.  Dense weights (embed,
norms, attention projections, router, shared expert) are stored as BF16/F32 in
safetensors and read directly by the qwen engine (it converts on load).  Only
expert weights are quantised to int8 per-row (`.weight` + `.weight.qs` scale
files) so the expert streaming cache stays compact.

Architecture notes (Qwen3-MoE / Qwen3.6-35B-A3B):
  - model_type: "qwen3_moe"
  - Standard GQA attention (q_proj, k_proj, v_proj, o_proj).  NOT MLA.
  - Per-head RMS norms: self_attn.q_norm.weight, self_attn.k_norm.weight.
  - RoPE theta ~1 000 000 (config field: rope_theta).
  - Softmax-gated MoE with shared expert (mlp.shared_expert.*).
  - Different tensor names from GLM-5.2 — the glm engine cannot load Qwen.
  - Text-only: vision inputs are NOT supported by this runtime.

Output layout (colibri container):
  <outdir>/config.json           — verbatim copy
  <outdir>/tokenizer.json        — verbatim copy
  <outdir>/tokenizer_config.json — verbatim copy (if present)
  <outdir>/special_tokens_map.json
  <outdir>/shard_N.safetensors   — dense+router tensors in f32/bf16
  (expert weights: split per-layer into separate files, or kept as-is with
   the matching .qs scale files added)

Usage:
  # download + convert Qwen/Qwen3.6-35B-A3B:
  coli convert --repo Qwen/Qwen3.6-35B-A3B --model /path/to/qwen_i8 --ebits 8

  # convert a local checkpoint:
  python3 tools/convert_qwen.py --model /path/to/Qwen3.6-35B-A3B --out /path/to/qwen_i8

  # list what would be converted without writing files:
  python3 tools/convert_qwen.py --repo Qwen/Qwen3.6-35B-A3B --out /tmp/dry --dry-run
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Windows: force UTF-8 output
if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError as exc:
    sys.exit(
        f"Missing dependencies: {exc}.\n"
        "Install: pip install torch safetensors\n"
        "(Only needed for conversion; the qwen engine itself has zero dependencies.)"
    )

QWEN3_MOE_TYPES = {"qwen3_moe", "qwen2_moe", "qwen_moe"}

# Tensors whose names match these patterns are expert weights to be quantised.
_EXPERT_RE = re.compile(
    r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$"
)


def is_expert(name: str) -> bool:
    return bool(_EXPERT_RE.match(name))


def quantize_row(w: torch.Tensor):
    """Row-wise int8 quantisation. Returns (int8_weights [O,I], float32_scales [O])."""
    w_f32 = w.float()
    row_max = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = row_max / 127.0
    q = (w_f32 / scales).round().clamp(-128, 127).to(torch.int8)
    return q, scales.squeeze(1).float()


def convert_shard(src: dict[str, torch.Tensor], ebits: int, dry_run: bool):
    """Return (dense_tensors, expert_pairs) where expert_pairs = [(name, q, qs), ...]."""
    dense: dict[str, torch.Tensor] = {}
    expert_pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for name, tensor in src.items():
        if is_expert(name):
            if dry_run:
                expert_pairs.append((name, None, None))
                continue
            w2d = tensor.float().view(tensor.shape[0], -1)
            if ebits == 8:
                q, qs = quantize_row(w2d)
                expert_pairs.append((name, q, qs))
        else:
            if not dry_run:
                # Keep dense tensors in bf16 to save disk space while preserving range.
                dense[name] = tensor.bfloat16() if tensor.dtype == torch.float32 else tensor
    return dense, expert_pairs


def check_arch(cfg_path: Path) -> str:
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"Cannot read config.json: {e}")
    mt = cfg.get("model_type", "")
    if mt not in QWEN3_MOE_TYPES:
        sys.exit(
            f"Expected model_type in {QWEN3_MOE_TYPES}, got '{mt}'.\n"
            "This converter is for Qwen3-MoE models (e.g. Qwen/Qwen3.6-35B-A3B).\n"
            "For GLM-5.2 use: coli convert --repo zai-org/GLM-5.2-FP8"
        )
    return mt


def copy_meta(src_dir: Path, out_dir: Path, dry_run: bool) -> None:
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json",
                  "chat_template.jinja"):
        src = src_dir / fname
        if src.exists():
            dst = out_dir / fname
            if not dry_run:
                dst.write_bytes(src.read_bytes())
            print(f"  copied {fname}", file=sys.stderr)


def convert_checkpoint(src_dir: Path, out_dir: Path, ebits: int, dry_run: bool) -> None:
    check_arch(src_dir / "config.json")
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    copy_meta(src_dir, out_dir, dry_run)

    shards = sorted(src_dir.glob("*.safetensors"))
    if not shards:
        sys.exit(f"No .safetensors files found in {src_dir}")

    total_expert = 0
    for i, shard_path in enumerate(shards):
        print(f"  [{i+1}/{len(shards)}] {shard_path.name}", file=sys.stderr)
        tensors = load_file(str(shard_path), device="cpu")
        dense, expert_pairs = convert_shard(tensors, ebits, dry_run)

        if not dry_run:
            out_shard = out_dir / shard_path.name
            if dense:
                save_file(dense, str(out_shard))

            # Save expert quantised weights alongside their scale files.
            for name, q, qs in expert_pairs:
                safe_name = name.replace("/", "__")
                q_path = out_dir / (safe_name + ".safetensors")
                qs_path = out_dir / (safe_name + ".qs.safetensors")
                save_file({name: q}, str(q_path))
                save_file({name + ".qs": qs}, str(qs_path))
                total_expert += 1

        total_expert += sum(1 for _ in expert_pairs) if dry_run else 0

    print(f"  done. {len(shards)} shards, {total_expert} expert tensors quantised.",
          file=sys.stderr)


def download_repo(repo_id: str, local_dir: Path) -> Path:
    print(f"Downloading {repo_id} ...", file=sys.stderr)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required for download.\n"
            "Install: pip install huggingface_hub\n"
            "Or download the model manually and pass --model <dir>."
        )
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        max_workers=4,
    )
    return Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert Qwen3-MoE checkpoint to the colibri int8 container.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", metavar="HF_REPO_ID",
                     help="HuggingFace repo ID, e.g. Qwen/Qwen3.6-35B-A3B")
    src.add_argument("--model", metavar="DIR",
                     help="Local HuggingFace checkpoint directory")
    ap.add_argument("--out", required=True, metavar="DIR",
                    help="Output directory for the converted model")
    ap.add_argument("--ebits", type=int, default=8,
                    help="Expert quantisation bits (only 8 / int8 is currently supported)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Scan tensors and report what would be converted, without writing files")
    args = ap.parse_args()

    if args.ebits != 8:
        ap.error("--ebits: only 8 (int8) is currently supported")

    out_dir = Path(args.out)

    if args.repo:
        dl_dir = out_dir / "_download"
        src_dir = download_repo(args.repo, dl_dir)
    else:
        src_dir = Path(args.model)
        if not src_dir.is_dir():
            ap.error(f"--model directory not found: {src_dir}")

    print(f"Converting {src_dir} -> {out_dir} (ebits={args.ebits})", file=sys.stderr)
    if args.dry_run:
        print("  (dry-run: no files will be written)", file=sys.stderr)

    convert_checkpoint(src_dir, out_dir, args.ebits, args.dry_run)

    if not args.dry_run:
        print(f"\nDone. Converted model in: {out_dir}", file=sys.stderr)
        print("Run:  coli serve --model " + str(out_dir) + "  --model-id qwen3-colibri",
              file=sys.stderr)


if __name__ == "__main__":
    main()
