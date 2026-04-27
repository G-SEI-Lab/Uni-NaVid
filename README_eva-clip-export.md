# EVA-CLIP ONNX 导出与 Jetson TensorRT 验证流程

本文档记录 Uni-NaVid 中 EVA-CLIP visual tower 的部署验证流程：先在 Ubuntu GPU 服务器导出静态 ONNX 与 PyTorch 参考输出，再将产物拷贝到 Jetson AGX Orin 上构建 TensorRT engine，并完成 benchmark 与数值对比。

当前流程目标固定为：

```text
batch = 1
input = 1 x 3 x 224 x 224
output = 1 x 257 x 1408
```

其中：

- `pixel_values`：EVA-CLIP 图像输入 tensor。
- `image_features`：EVA-CLIP visual tower 输出视觉 token 特征。
- ONNX 为静态 shape 模型，不使用动态 batch profile。

---

## 目录结构建议

在 Uni-NaVid 项目根目录下执行：

```text
Uni-NaVid/
├── scripts/
│   ├── eva_clip_export_onnx_ref.py
│   └── eva_clip_trt_validate.py
├── model_zoo/
│   └── eva_vit_g.pth
├── uninavid/
│   └── processor/
│       └── clip-patch14-224/
├── test_224.png
└── eva_step1_artifacts/
    ├── eva_vit_g_bs1_224.onnx
    ├── eva_ref_bs1_224.npz
    ├── meta.json
    ├── eva_vit_g_bs1_224_fp16.plan
    └── eva_vit_g_bs1_224_fp32.plan
```

---

## 1. Ubuntu GPU 服务器：导出 ONNX + 参考输出

### 1.1 执行命令

在 Ubuntu GPU 服务器上运行：

```bash
python scripts/eva_clip_export_onnx_ref.py \
  --eva-ckpt ./model_zoo/eva_vit_g.pth \
  --processor-dir ./uninavid/processor/clip-patch14-224 \
  --image test_224.png \
  --outdir ./eva_step1_artifacts \
  --device cuda \
  --run-ort-check
```

### 1.2 生成文件

执行完成后，`./eva_step1_artifacts` 下应生成：

```text
eva_vit_g_bs1_224.onnx
eva_ref_bs1_224.npz
meta.json
```

说明：

- `eva_vit_g_bs1_224.onnx`：静态 shape ONNX 模型，输入为 `1x3x224x224`。
- `eva_ref_bs1_224.npz`：PyTorch reference 数据，包含：
  - `pixel_values`：预处理后的输入。
  - `image_features`：PyTorch EVA-CLIP 输出。
- `meta.json`：导出元信息，包括输入输出 shape、PyTorch 单次前向延时、ONNX Runtime 对比结果等。

### 1.3 预期输出示例

```text
[OK] ONNX exported: .../eva_step1_artifacts/eva_vit_g_bs1_224.onnx
[OK] Reference saved: .../eva_step1_artifacts/eva_ref_bs1_224.npz
[INFO] Input shape: (1, 3, 224, 224)
[INFO] Output shape: (1, 257, 1408)
[INFO] PyTorch latency (single run): ... ms
[INFO] ORT check - max_abs=..., mean_abs=..., cosine=...
```

### 1.4 注意事项

该导出脚本使用 `CLIPImageProcessor` 对输入图片进行预处理，并强制检查预处理输出 shape 为 `(1, 3, 224, 224)`。ONNX 导出时使用：

```text
input_names = ["pixel_values"]
output_names = ["image_features"]
dynamic_axes = None
```

因此导出的 ONNX 是静态输入模型。

如果 CUDA constant folding 报错，脚本会自动 fallback 到 `do_constant_folding=False` 重新导出。

---

## 2. 拷贝产物到 Jetson AGX Orin

将 Ubuntu 服务器上的整个 `eva_step1_artifacts` 目录拷贝到 Jetson 的 Uni-NaVid 项目根目录。

示例：

```bash
scp -r ./eva_step1_artifacts ubuntu@<JETSON_IP>:~/workspace/huangtao/Uni-NaVid/
```

或者使用 U 盘 / rsync：

```bash
rsync -av ./eva_step1_artifacts/ ubuntu@<JETSON_IP>:~/workspace/huangtao/Uni-NaVid/eva_step1_artifacts/
```

Jetson 端确认文件存在：

```bash
cd ~/workspace/huangtao/Uni-NaVid
ls -lh ./eva_step1_artifacts
```

应至少包含：

```text
eva_vit_g_bs1_224.onnx
eva_ref_bs1_224.npz
meta.json
```

---

## 3. Jetson AGX Orin：构建 TensorRT Engine

建议先切换到最大性能模式，以获得更稳定的 benchmark 数字：

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

确认 TensorRT 版本：

```bash
/usr/src/tensorrt/bin/trtexec --version
```

本文流程在 TensorRT 10.3 上验证。TensorRT 10.x 推荐使用：

```text
--memPoolSize=workspace:4096
```

不要使用旧版本常见的：

```text
--workspace=4096
```

---

## 4. 构建 FP16 Engine

由于 ONNX 是静态 shape 模型，构建时不要传：

```text
--minShapes
--optShapes
--maxShapes
```

直接使用：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=./eva_step1_artifacts/eva_vit_g_bs1_224.onnx \
  --saveEngine=./eva_step1_artifacts/eva_vit_g_bs1_224_fp16.plan \
  --memPoolSize=workspace:4096 \
  --fp16 \
  --builderOptimizationLevel=5
```

构建成功时应看到：

```text
&&&& PASSED TensorRT.trtexec
```

FP16 engine 的 TensorRT 日志中 precision 通常显示为：

```text
Precision: FP32+FP16
```

这是正常的。`--fp16` 表示允许 TensorRT 使用 FP16 tactic/kernel，并不表示强制所有层都以 FP16 执行。TensorRT 会根据算子支持、性能和精度约束生成 mixed precision engine。

---

## 5. 构建 FP32 Engine，可选

用于对照验证：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=./eva_step1_artifacts/eva_vit_g_bs1_224.onnx \
  --saveEngine=./eva_step1_artifacts/eva_vit_g_bs1_224_fp32.plan \
  --memPoolSize=workspace:4096 \
  --builderOptimizationLevel=5
```

构建成功时应看到：

```text
&&&& PASSED TensorRT.trtexec
```

---

## 6. TensorRT Benchmark

### 6.1 FP16 benchmark

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=./eva_step1_artifacts/eva_vit_g_bs1_224_fp16.plan \
  --warmUp=200 \
  --iterations=100 \
  --duration=0 \
  --useSpinWait
```

当前实测结果示例：

```text
Throughput: 32.5165 qps
Latency mean: 30.5251 ms
GPU Compute Time mean: 30.4482 ms
Engine size: 1887 MiB
```

### 6.2 FP32 benchmark

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=./eva_step1_artifacts/eva_vit_g_bs1_224_fp32.plan \
  --warmUp=200 \
  --iterations=100 \
  --duration=0 \
  --useSpinWait
```

当前实测结果示例：

```text
Throughput: 13.9426 qps
Latency mean: 71.0930 ms
GPU Compute Time mean: 71.0096 ms
Engine size: 3764 MiB
```

### 6.3 FP16 vs FP32 延时对比

| Engine | Precision | Latency mean | GPU Compute Time mean | Throughput | Engine Size |
|---|---:|---:|---:|---:|---:|
| `eva_vit_g_bs1_224_fp16.plan` | FP32+FP16 | ~30.5 ms | ~30.4 ms | ~32.5 qps | ~1887 MiB |
| `eva_vit_g_bs1_224_fp32.plan` | FP32 | ~71.1 ms | ~71.0 ms | ~13.9 qps | ~3764 MiB |

FP16 相比 FP32 约有：

```text
71.1 / 30.5 ≈ 2.33x
```

即单帧 EVA-CLIP visual encoder TensorRT 推理约加速 2.3 倍。

注意：这里的 latency 是 EVA-CLIP TensorRT engine 的单次推理延时，不包含 Uni-NaVid 端到端流程中的相机取图、图像预处理、visual projector、cross-attention、Vicuna decoding、ROS 通信等开销。

---

## 7. Jetson TensorRT Python Runtime 数值验证

使用 `scripts/eva_clip_trt_validate.py` 对比 Jetson TensorRT 输出与 Ubuntu PyTorch reference 输出。

### 7.1 FP16 验证

推荐先使用适合 FP16 full-token 特征的阈值：

```bash
python scripts/eva_clip_trt_validate.py \
  --onnx ./eva_step1_artifacts/eva_vit_g_bs1_224.onnx \
  --engine ./eva_step1_artifacts/eva_vit_g_bs1_224_fp16.plan \
  --ref-npz ./eva_step1_artifacts/eva_ref_bs1_224.npz \
  --fp16 \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --image ./test_224.png \
  --processor-dir ./uninavid/processor/clip-patch14-224 \
  --outdir ./eva_step1_jetson \
  --max-abs-thr 20 \
  --mean-abs-thr 0.05 \
  --cos-thr 0.999
```

输出文件：

```text
./eva_step1_jetson/trt_output.npy
./eva_step1_jetson/trt_compare_report.json
```

当前 FP16 实测指标示例：

```text
Preprocess check:
  max_abs  = 0.00000024
  mean_abs = 0.00000009
  cosine   = 1.00000000

TRT vs Torch:
  max_abs  = 13.923141
  mean_abs = 0.024471
  cosine   = 0.99924600
```

补充分布分析示例：

```text
full cosine = 0.9992460016
CLS cosine  = 0.9999888582
CLS max_abs = 0.06511688
CLS mean_abs = 0.010247403
```

解释：FP16 full-token 输出存在少量 patch-token/channel outlier，因此逐元素 `max_abs` 会比较大；但整体 cosine 和 CLS token 一致性较高。对 Uni-NaVid 中“视觉 token 与自然语言交叉注意力，再传入 Vicuna”的使用场景，建议以 full-token cosine、CLS/patch token 分布、以及下游端到端输出一致性作为最终判断标准，而不是仅用 `max_abs <= 0.1`。

### 7.2 FP32 验证

```bash
python scripts/eva_clip_trt_validate.py \
  --onnx ./eva_step1_artifacts/eva_vit_g_bs1_224.onnx \
  --engine ./eva_step1_artifacts/eva_vit_g_bs1_224_fp32.plan \
  --ref-npz ./eva_step1_artifacts/eva_ref_bs1_224.npz \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --image ./test_224.png \
  --processor-dir ./uninavid/processor/clip-patch14-224 \
  --outdir ./eva_step1_jetson_fp32 \
  --max-abs-thr 3.0 \
  --mean-abs-thr 0.01 \
  --cos-thr 0.9999
```

输出文件：

```text
./eva_step1_jetson_fp32/trt_output.npy
./eva_step1_jetson_fp32/trt_compare_report.json
```

当前 FP32 实测指标示例：

```text
Preprocess check:
  max_abs  = 0.00000024
  mean_abs = 0.00000009
  cosine   = 1.00000000

TRT vs Torch:
  max_abs  = 2.731407
  mean_abs = 0.004840
  cosine   = 0.99996930
```

解释：FP32 与 PyTorch reference 的整体一致性很好，但仍可能因为单点 outlier 无法满足非常严格的 `max_abs <= 0.1`。因此建议 FP32 使用更合理的阈值组合：

```text
mean_abs <= 0.01
cosine >= 0.9999
max_abs <= 3.0
```

---

## 8. 推荐验收标准

### 8.1 EVA-CLIP visual encoder 层面

FP16 初步验收可采用：

```text
full cosine >= 0.999
mean_abs <= 0.05
p99_abs <= 0.30
p99.9_abs <= 1.00
CLS cosine >= 0.9999
```

FP32 对照验收可采用：

```text
full cosine >= 0.9999
mean_abs <= 0.01
max_abs <= 3.0
```

### 8.2 Uni-NaVid 端到端层面

由于 EVA-CLIP 的输出后续会进入：

```text
EVA-CLIP visual tokens
→ visual projector / adapter
→ cross-attention with natural language
→ Vicuna
→ final text/action output
```

正式部署建议增加端到端验证：

```text
1. 选取 20~100 张真实导航场景图。
2. 固定 prompt。
3. 固定 decoding 参数：temperature=0, do_sample=False。
4. 对比 PyTorch EVA-CLIP 与 TensorRT FP16 EVA-CLIP 的最终输出。
5. 记录 final answer/action 一致率、logits top-k overlap、关键导航动作一致率。
```

如果最终 action / response 基本一致，则 FP16 EVA-CLIP TensorRT engine 可以进入 Uni-NaVid 实机部署。

---

## 9. 常见问题

### 9.1 TensorRT 10 报 `Unknown option: --workspace`

TensorRT 10.x 不再推荐：

```text
--workspace=4096
```

应使用：

```text
--memPoolSize=workspace:4096
```

### 9.2 静态 ONNX 报 `Static model does not take explicit shapes`

如果看到：

```text
Static model does not take explicit shapes since the shape of inference tensors will be determined by the model itself
```

说明当前 ONNX 是静态 shape 模型，不要传：

```text
--minShapes=pixel_values:1x3x224x224
--optShapes=pixel_values:1x3x224x224
--maxShapes=pixel_values:1x3x224x224
```

正确构建命令应直接让 TensorRT 从 ONNX 读取 shape：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=./eva_step1_artifacts/eva_vit_g_bs1_224.onnx \
  --saveEngine=./eva_step1_artifacts/eva_vit_g_bs1_224_fp16.plan \
  --memPoolSize=workspace:4096 \
  --fp16 \
  --builderOptimizationLevel=5
```

### 9.3 `Precision: FP32+FP16` 是否正常？

正常。`--fp16` 表示允许 TensorRT 使用 FP16 kernel，但 TensorRT 可能保留部分层或 I/O 为 FP32。因此日志显示 `FP32+FP16` 是正常的 mixed precision engine。

### 9.4 `Latency` 与 `GPU Compute Time` 的区别

在 `trtexec` 中可近似理解为：

```text
Latency ≈ H2D Latency + GPU Compute Time + D2H Latency
```

- `GPU Compute Time`：GPU kernel 真正执行模型计算的时间。
- `Latency`：一次 TensorRT inference 从输入传输、GPU 计算到输出传回的总耗时。

当前测试中 H2D/D2H 很小，所以 `Latency` 与 `GPU Compute Time` 非常接近，瓶颈主要是模型计算本身。

### 9.5 Pillow / PIL 报 `Image.Resampling` 不存在

如果 Jetson 环境中 `transformers` 导入 `CLIPImageProcessor` 时报：

```text
AttributeError: module 'PIL.Image' has no attribute 'Resampling'
```

通常是 venv 加载了系统旧版 Pillow。可检查：

```bash
python - <<'PY'
import PIL
from PIL import Image
print(PIL.__file__)
print(PIL.__version__)
print(hasattr(Image, "Resampling"))
PY
```

建议在 venv 中安装新版 Pillow：

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade --force-reinstall pip setuptools wheel
python -m pip install --no-cache-dir --force-reinstall "Pillow==9.5.0"
```

或在脚本中临时兼容旧 Pillow：

```python
from PIL import Image

if not hasattr(Image, "Resampling"):
    Image.Resampling = Image

from transformers import CLIPImageProcessor
```

---

## 10. 一键命令汇总

### Ubuntu GPU 服务器

```bash
python scripts/eva_clip_export_onnx_ref.py \
  --eva-ckpt ./model_zoo/eva_vit_g.pth \
  --processor-dir ./uninavid/processor/clip-patch14-224 \
  --image test_224.png \
  --outdir ./eva_step1_artifacts \
  --device cuda \
  --run-ort-check
```

### Jetson 构建 FP16

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=./eva_step1_artifacts/eva_vit_g_bs1_224.onnx \
  --saveEngine=./eva_step1_artifacts/eva_vit_g_bs1_224_fp16.plan \
  --memPoolSize=workspace:4096 \
  --fp16 \
  --builderOptimizationLevel=5
```

### Jetson benchmark FP16

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=./eva_step1_artifacts/eva_vit_g_bs1_224_fp16.plan \
  --warmUp=200 \
  --iterations=100 \
  --duration=0 \
  --useSpinWait
```

### Jetson 验证 FP16

```bash
python scripts/eva_clip_trt_validate.py \
  --onnx ./eva_step1_artifacts/eva_vit_g_bs1_224.onnx \
  --engine ./eva_step1_artifacts/eva_vit_g_bs1_224_fp16.plan \
  --ref-npz ./eva_step1_artifacts/eva_ref_bs1_224.npz \
  --fp16 \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --image ./test_224.png \
  --processor-dir ./uninavid/processor/clip-patch14-224 \
  --outdir ./eva_step1_jetson \
  --max-abs-thr 20 \
  --mean-abs-thr 0.05 \
  --cos-thr 0.999
```

### Jetson 构建 FP32 对照

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=./eva_step1_artifacts/eva_vit_g_bs1_224.onnx \
  --saveEngine=./eva_step1_artifacts/eva_vit_g_bs1_224_fp32.plan \
  --memPoolSize=workspace:4096 \
  --builderOptimizationLevel=5
```

### Jetson benchmark FP32

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=./eva_step1_artifacts/eva_vit_g_bs1_224_fp32.plan \
  --warmUp=200 \
  --iterations=100 \
  --duration=0 \
  --useSpinWait
```

### Jetson 验证 FP32

```bash
python scripts/eva_clip_trt_validate.py \
  --onnx ./eva_step1_artifacts/eva_vit_g_bs1_224.onnx \
  --engine ./eva_step1_artifacts/eva_vit_g_bs1_224_fp32.plan \
  --ref-npz ./eva_step1_artifacts/eva_ref_bs1_224.npz \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --image ./test_224.png \
  --processor-dir ./uninavid/processor/clip-patch14-224 \
  --outdir ./eva_step1_jetson_fp32 \
  --max-abs-thr 3.0 \
  --mean-abs-thr 0.01 \
  --cos-thr 0.9999
```

---

## 11. 当前阶段结论

1. Ubuntu GPU 服务器负责导出静态 ONNX 与 PyTorch reference。
2. Jetson AGX Orin 负责构建 TensorRT engine、benchmark、以及与 Ubuntu reference 对比。
3. ONNX 是静态 `1x3x224x224` 模型，因此 TensorRT 构建时不传 `--minShapes/--optShapes/--maxShapes`。
4. TensorRT 10.x 使用 `--memPoolSize=workspace:4096`。
5. 当前单帧 224 输入下，FP16 TensorRT EVA-CLIP visual encoder 延时约 `30.5 ms`，FP32 约 `71.1 ms`，FP16 约快 `2.33x`。
6. FP16 数值验证不建议只看 `max_abs`，更应结合 `cosine`、分位数误差、CLS/patch token 指标，以及 Uni-NaVid 端到端输出一致性。
