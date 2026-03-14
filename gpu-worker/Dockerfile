# Rust GPU worker — parakeet-rs (Parakeet TDT + Sortformer diarization)
# No Python, no PyTorch, no NeMo. Just Rust + ONNX Runtime + CUDA.

# ── Stage 1: Build with CUDA-enabled ORT ──
FROM rust:1.89-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
      pkg-config libssl-dev cmake \
    && rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime GPU (ort crate v2.0.0-rc.12 → ORT 1.24.3)
RUN curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-x64-gpu-1.24.3.tgz" \
      -o /tmp/ort.tgz \
    && mkdir -p /opt/ort \
    && tar xzf /tmp/ort.tgz -C /opt/ort --strip-components=1 \
    && rm /tmp/ort.tgz

WORKDIR /build
COPY Cargo.toml ./
COPY src ./src

# Tell ort-sys to dynamically link against our pre-downloaded CUDA-enabled ORT
ENV ORT_LIB_LOCATION="/opt/ort/lib" \
    ORT_PREFER_DYNAMIC_LINK=1
RUN cargo build --release

# ── Stage 2: Download + convert models ──
FROM python:3.12-slim AS models

RUN pip install --no-cache-dir huggingface-hub onnxruntime

# TDT model — download FP32, convert encoder to FP16 for 2x Tensor Core throughput
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('istupakov/parakeet-tdt-0.6b-v3-onnx', local_dir='/models/tdt')"

# Convert encoder to FP16 (the big model — 600M params).
# Uses onnxruntime's float16 converter which handles >2GB models on-disk.
COPY <<'CONVERT_SCRIPT' /tmp/convert_fp16.py
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

model_path = "/models/tdt/encoder-model.onnx"
model = onnx.load(model_path, load_external_data=True)

# Convert float32 initializers to float16 in-place
converted = 0
for init in model.graph.initializer:
    if init.data_type == TensorProto.FLOAT:
        arr = numpy_helper.to_array(init).astype(np.float16)
        new_tensor = numpy_helper.from_array(arr, init.name)
        init.CopyFrom(new_tensor)
        converted += 1

print(f"Converted {converted} tensors to FP16")

# Save with external data (model > 2GB protobuf limit)
onnx.save(
    model,
    model_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="encoder-model.onnx_data",
    size_threshold=0,
)
print("Saved FP16 encoder")
CONVERT_SCRIPT
RUN python /tmp/convert_fp16.py

# Sortformer v2.1 diarization model
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('altunenes/parakeet-rs', 'diar_streaming_sortformer_4spk-v2.1.onnx', local_dir='/models/sortformer')"

# ── Stage 3: Runtime ──
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1001 app && useradd -u 1001 -g app -s /bin/bash -m app

# Copy binary
COPY --from=builder /build/target/release/yt-mcp-gpu-worker /usr/local/bin/yt-mcp-gpu-worker

# Copy ONNX Runtime GPU shared libs (main lib + CUDA/TensorRT providers)
# Use cp instead of COPY to preserve symlinks
COPY --from=builder /opt/ort/lib/ /tmp/ort-lib/
RUN cp -a /tmp/ort-lib/*.so* /usr/local/lib/ && rm -rf /tmp/ort-lib && ldconfig

# Copy models
COPY --from=models /models /models

RUN mkdir -p /tmp/worker && chown -R app:app /tmp/worker

USER app

ENV WORKER_PORT=3001 \
    MODEL_DIR=/models \
    RUST_LOG=info

EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${WORKER_PORT}/worker/healthz" >/dev/null 2>&1 || exit 1

CMD ["yt-mcp-gpu-worker"]
