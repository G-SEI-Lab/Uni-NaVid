#!/usr/bin/env python3
"""Build and validate EVA-CLIP TensorRT engine against Ubuntu PyTorch reference.

Run on Jetson AGX Orin.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from transformers import CLIPImageProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorRT validation for EVA-CLIP visual tower.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model.")
    parser.add_argument("--engine", type=str, required=True, help="Path to TensorRT engine.")
    parser.add_argument("--ref-npz", type=str, required=True, help="Reference npz from Ubuntu script.")
    parser.add_argument("--outdir", type=str, default="artifacts/eva_clip_step1", help="Output directory.")

    parser.add_argument("--build-engine", action="store_true", help="Build engine with trtexec.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 when building engine.")
    parser.add_argument("--workspace-mib", type=int, default=4096, help="Builder workspace size in MiB.")
    parser.add_argument("--trtexec", type=str, default="trtexec", help="Path to trtexec binary.")
    parser.add_argument("--input-name", type=str, default="pixel_values", help="ONNX input tensor name.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip trtexec latency benchmark.")

    parser.add_argument("--image", type=str, default=None, help="Optional image path to verify preprocessing.")
    parser.add_argument(
        "--processor-dir",
        type=str,
        default="uninavid/processor/clip-patch14-224",
        help="Optional processor path for preprocessing check.",
    )
    parser.add_argument("--max-abs-thr", type=float, default=0.10, help="Max absolute error threshold.")
    parser.add_argument("--mean-abs-thr", type=float, default=0.01, help="Mean absolute error threshold.")
    parser.add_argument("--cos-thr", type=float, default=0.995, help="Cosine similarity threshold.")
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


def trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int8:
        return torch.int8
    if hasattr(trt, "bool") and dtype == trt.bool:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


def run_subprocess(cmd: List[str]) -> None:
    print("[CMD] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_engine(
    trtexec_bin: str,
    onnx_path: Path,
    engine_path: Path,
    input_name: str,
    fp16: bool,
    workspace_mib: int,
) -> None:
    # Static ONNX: input shape is already fixed in the model.
    # TensorRT 10.x: use --memPoolSize instead of deprecated --workspace.
    common = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mib}",
    ]
    if fp16:
        common.append("--fp16")

    candidate_cmds = [
        common + ["--builderOptimizationLevel=5"],
        common,
    ]

    last_err: Exception | None = None
    for idx, cmd in enumerate(candidate_cmds, start=1):
        try:
            run_subprocess(cmd)
            return
        except subprocess.CalledProcessError as err:
            last_err = err
            if idx < len(candidate_cmds):
                print("[WARN] Build command failed, retrying with a simpler trtexec flag set.")

    if last_err is not None:
        raise last_err


def benchmark_engine(
    trtexec_bin: str,
    engine_path: Path,
    input_name: str,
) -> None:
    # Static engine already carries the input shape.
    cmd = [
        trtexec_bin,
        f"--loadEngine={engine_path}",
        "--warmUp=200",
        "--iterations=100",
        "--duration=0",
        "--useSpinWait",
    ]
    run_subprocess(cmd)


def infer_trt_legacy(engine: trt.ICudaEngine, x: np.ndarray) -> Dict[str, np.ndarray]:
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context.")

    num_bindings = engine.num_bindings
    bindings: List[int] = [0] * num_bindings
    tensors: Dict[str, torch.Tensor] = {}
    output_names: List[str] = []

    input_indices = [i for i in range(num_bindings) if engine.binding_is_input(i)]
    if len(input_indices) != 1:
        raise RuntimeError(f"Expected 1 input binding, got {len(input_indices)}")
    input_idx = input_indices[0]

    if -1 in tuple(engine.get_binding_shape(input_idx)):
        context.set_binding_shape(input_idx, tuple(x.shape))

    for i in range(num_bindings):
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        t_dtype = trt_dtype_to_torch(dtype)
        shape = tuple(context.get_binding_shape(i))

        if engine.binding_is_input(i):
            inp = torch.from_numpy(x).to(device="cuda", dtype=t_dtype).contiguous()
            tensors[name] = inp
            bindings[i] = int(inp.data_ptr())
        else:
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Unresolved output shape for binding {name}: {shape}")
            out = torch.empty(shape, device="cuda", dtype=t_dtype)
            tensors[name] = out
            output_names.append(name)
            bindings[i] = int(out.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream
    ok = context.execute_async_v2(bindings, stream)
    if not ok:
        raise RuntimeError("TensorRT execute_async_v2 failed.")
    torch.cuda.synchronize()

    return {name: tensors[name].detach().cpu().numpy() for name in output_names}


def infer_trt_modern(engine: trt.ICudaEngine, x: np.ndarray) -> Dict[str, np.ndarray]:
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context.")

    input_names: List[str] = []
    output_names: List[str] = []
    tensors: Dict[str, torch.Tensor] = {}

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    if len(input_names) != 1:
        raise RuntimeError(f"Expected 1 input tensor, got {len(input_names)}")

    input_name = input_names[0]
    context.set_input_shape(input_name, tuple(x.shape))

    for name in input_names:
        dtype = engine.get_tensor_dtype(name)
        t_dtype = trt_dtype_to_torch(dtype)
        inp = torch.from_numpy(x).to(device="cuda", dtype=t_dtype).contiguous()
        tensors[name] = inp
        context.set_tensor_address(name, int(inp.data_ptr()))

    for name in output_names:
        dtype = engine.get_tensor_dtype(name)
        t_dtype = trt_dtype_to_torch(dtype)
        shape = tuple(context.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            raise RuntimeError(f"Unresolved output shape for tensor {name}: {shape}")
        out = torch.empty(shape, device="cuda", dtype=t_dtype)
        tensors[name] = out
        context.set_tensor_address(name, int(out.data_ptr()))

    stream = torch.cuda.current_stream().cuda_stream
    ok = context.execute_async_v3(stream)
    if not ok:
        raise RuntimeError("TensorRT execute_async_v3 failed.")
    torch.cuda.synchronize()

    return {name: tensors[name].detach().cpu().numpy() for name in output_names}


def infer_trt(engine_path: Path, x: np.ndarray) -> Tuple[str, np.ndarray]:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    serialized = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

    if hasattr(engine, "num_bindings") and hasattr(engine, "get_binding_name"):
        outputs = infer_trt_legacy(engine, x)
    else:
        outputs = infer_trt_modern(engine, x)

    if len(outputs) != 1:
        raise RuntimeError(f"Expected single output, got {list(outputs.keys())}")
    out_name = list(outputs.keys())[0]
    return out_name, outputs[out_name]


def maybe_check_preprocess(image_path: Path, processor_dir: Path, ref_input: np.ndarray) -> Dict[str, float]:
    processor = CLIPImageProcessor.from_pretrained(str(processor_dir))
    image = Image.open(image_path).convert("RGB")
    cur = processor.preprocess(image, return_tensors="pt")["pixel_values"].numpy().astype(np.float32, copy=False)
    if tuple(cur.shape) != tuple(ref_input.shape):
        raise ValueError(f"Preprocess shape mismatch: {tuple(cur.shape)} vs {tuple(ref_input.shape)}")
    return summarize_diff(ref_input, cur)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on Jetson for TensorRT validation.")

    onnx_path = Path(args.onnx).resolve()
    engine_path = Path(args.engine).resolve()
    ref_path = Path(args.ref_npz).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ref_data = np.load(ref_path)
    ref_input = ref_data["pixel_values"].astype(np.float32, copy=False)
    ref_output = ref_data["image_features"].astype(np.float32, copy=False)

    preprocess_metrics = None
    if args.image is not None:
        preprocess_metrics = maybe_check_preprocess(
            image_path=Path(args.image).resolve(),
            processor_dir=Path(args.processor_dir).resolve(),
            ref_input=ref_input,
        )

    if args.build_engine or not engine_path.exists():
        build_engine(
            trtexec_bin=args.trtexec,
            onnx_path=onnx_path,
            engine_path=engine_path,
            input_name=args.input_name,
            fp16=args.fp16,
            workspace_mib=args.workspace_mib,
        )

    if not args.skip_benchmark:
        benchmark_engine(
            trtexec_bin=args.trtexec,
            engine_path=engine_path,
            input_name=args.input_name,
        )

    out_name, trt_output = infer_trt(engine_path, ref_input)
    trt_output_fp32 = trt_output.astype(np.float32, copy=False)
    metrics = summarize_diff(ref_output, trt_output_fp32)

    passed = (
        metrics["max_abs"] <= args.max_abs_thr
        and metrics["mean_abs"] <= args.mean_abs_thr
        and metrics["cosine"] >= args.cos_thr
    )

    np.save(outdir / "trt_output.npy", trt_output_fp32)
    report = {
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "ref_npz": str(ref_path),
        "trt_output_name": out_name,
        "input_shape": list(ref_input.shape),
        "ref_output_shape": list(ref_output.shape),
        "trt_output_shape": list(trt_output_fp32.shape),
        "preprocess_metrics": preprocess_metrics,
        "metrics": metrics,
        "thresholds": {
            "max_abs_thr": args.max_abs_thr,
            "mean_abs_thr": args.mean_abs_thr,
            "cos_thr": args.cos_thr,
        },
        "passed": passed,
    }
    (outdir / "trt_compare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if preprocess_metrics is not None:
        print(
            "[INFO] Preprocess check - "
            f"max_abs={preprocess_metrics['max_abs']:.8f}, "
            f"mean_abs={preprocess_metrics['mean_abs']:.8f}, "
            f"cosine={preprocess_metrics['cosine']:.8f}"
        )
    print(
        "[INFO] TRT vs Torch - "
        f"max_abs={metrics['max_abs']:.6f}, "
        f"mean_abs={metrics['mean_abs']:.6f}, "
        f"cosine={metrics['cosine']:.8f}"
    )
    print(f"[INFO] Report: {outdir / 'trt_compare_report.json'}")
    if not passed:
        print("[FAIL] Validation failed.")
        sys.exit(2)
    print("[PASS] Validation passed.")


if __name__ == "__main__":
    main()
