#!/usr/bin/env python3
"""Export EVA-CLIP visual tower to ONNX and generate PyTorch reference outputs.

Run on Ubuntu GPU server.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uninavid.model.multimodal_encoder.eva_vit import EVAVisionTowerLavis  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export EVA-CLIP visual tower ONNX and dump reference tensors."
    )
    parser.add_argument("--eva-ckpt", type=str, required=True, help="Path to eva_vit_g.pth")
    parser.add_argument(
        "--processor-dir",
        type=str,
        default="uninavid/processor/clip-patch14-224",
        help="Path to CLIPImageProcessor config directory.",
    )
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--outdir", type=str, default="artifacts/eva_clip_step1", help="Output directory.")
    parser.add_argument(
        "--onnx-name",
        type=str,
        default="eva_vit_g_bs1_224.onnx",
        help="Output ONNX filename under outdir.",
    )
    parser.add_argument(
        "--ref-name",
        type=str,
        default="eva_ref_bs1_224.npz",
        help="Output reference tensor filename under outdir.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for PyTorch forward and export.",
    )
    parser.add_argument(
        "--run-ort-check",
        action="store_true",
        help="If onnxruntime is installed, run ONNX vs PyTorch numeric check.",
    )
    return parser.parse_args()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.reshape(-1).astype(np.float64)
    b64 = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-12
    return float(np.dot(a64, b64) / denom)


def summarize_diff(ref: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(ref - pred)
    return {
        "max_abs": float(abs_err.max()),
        "mean_abs": float(abs_err.mean()),
        "cosine": cosine_similarity(ref, pred),
    }


def load_and_preprocess(image_path: Path, processor_dir: Path) -> np.ndarray:
    processor = CLIPImageProcessor.from_pretrained(str(processor_dir))
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor.preprocess(image, return_tensors="pt")["pixel_values"]
    # Fixed pipeline target for this step: bs=1, 3x224x224.
    if tuple(pixel_values.shape) != (1, 3, 224, 224):
        raise ValueError(f"Unexpected preprocessed shape: {tuple(pixel_values.shape)}")
    return pixel_values.numpy().astype(np.float32, copy=False)


def export_onnx(model: torch.nn.Module, pixel_values_np: np.ndarray, onnx_path: Path, opset: int, device: torch.device) -> None:
    model.eval()
    dummy = torch.from_numpy(pixel_values_np).to(device=device, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["image_features"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )


def maybe_run_ort_check(onnx_path: Path, pixel_values_np: np.ndarray, ref_np: np.ndarray) -> Dict[str, float] | None:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return None

    session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out = session.run(None, {"pixel_values": pixel_values_np})[0].astype(np.float32, copy=False)
    return summarize_diff(ref_np, out)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    eva_ckpt = Path(args.eva_ckpt).resolve()
    processor_dir = Path(args.processor_dir).resolve()
    image_path = Path(args.image).resolve()
    onnx_path = outdir / args.onnx_name
    ref_path = outdir / args.ref_name
    meta_path = outdir / "meta.json"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but --device=cuda was set.")
    device = torch.device(args.device)

    pixel_values_np = load_and_preprocess(image_path, processor_dir)

    # Keep this model instance for ONNX export and reference inference to avoid duplicate load.
    tower = EVAVisionTowerLavis(
        vision_tower=str(eva_ckpt),
        image_processor=str(processor_dir),
        args=None,
        use_checkpoint=False,
        drop_path_rate=0.0,
    )
    model = tower.vision_tower.eval().to(device)

    pixel_values = torch.from_numpy(pixel_values_np).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ref = model(pixel_values)
        if device.type == "cuda":
            torch.cuda.synchronize()
        torch_latency_ms = (time.perf_counter() - t0) * 1000.0
    ref_np = ref.detach().cpu().numpy().astype(np.float32, copy=False)

    export_onnx(model, pixel_values_np, onnx_path, args.opset, device=device)

    np.savez(ref_path, pixel_values=pixel_values_np, image_features=ref_np)

    ort_metrics = maybe_run_ort_check(onnx_path, pixel_values_np, ref_np) if args.run_ort_check else None

    meta = {
        "eva_ckpt": str(eva_ckpt),
        "processor_dir": str(processor_dir),
        "image": str(image_path),
        "input_shape": list(pixel_values_np.shape),
        "output_shape": list(ref_np.shape),
        "torch_device": str(device),
        "torch_latency_ms": float(torch_latency_ms),
        "onnx_path": str(onnx_path),
        "ref_path": str(ref_path),
        "ort_metrics": ort_metrics,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] ONNX exported: {onnx_path}")
    print(f"[OK] Reference saved: {ref_path}")
    print(f"[INFO] Input shape: {tuple(pixel_values_np.shape)}")
    print(f"[INFO] Output shape: {tuple(ref_np.shape)}")
    print(f"[INFO] PyTorch latency (single run): {torch_latency_ms:.3f} ms")
    if ort_metrics is not None:
        print(
            "[INFO] ORT check - "
            f"max_abs={ort_metrics['max_abs']:.6f}, "
            f"mean_abs={ort_metrics['mean_abs']:.6f}, "
            f"cosine={ort_metrics['cosine']:.8f}"
        )


if __name__ == "__main__":
    main()
