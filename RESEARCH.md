# WhisperKit Transcription Speed Research

## Hardware

Apple M1 Max (32 GPU cores, 10 CPU cores, ~400 GB/s memory bandwidth)
Audio: ted_60.m4a (60s TED talk clip)
macOS 15.7.4

## Model Benchmark Results

All benchmarks use auto compute selection (cpuAndGPU on ≥14 GPU cores), 3 runs each.

| Model | Size | Tok/s (avg) | Speed Factor | vs large-v2 | Quality |
|-------|------|-------------|-------------|-------------|---------|
| **large-v3-v20240930_turbo** | ~600MB | **78.57** | **17.3x RT** | **5.5x faster** | Excellent |
| **distil-large-v3** | 1.4GB | **65.79** | **15.5x RT** | **4.6x faster** | Excellent |
| large-v3 | 2.9GB | 14.39 | 3.2x RT | 1.0x | Reference |
| **large-v2** (baseline) | 2.9GB | **14.36** | **3.5x RT** | **1.0x** | Reference |
| large-v2_949MB (quantized) | 949MB | 11.00 | 2.8x RT | 0.77x slower | Good |

### Key Finding: Model Architecture > Model Quantization

The turbo model (4 decoder layers) is **5.5x faster** than large-v2 (32 decoder layers) with near-identical quality. The quantized large-v2 (949MB) is actually **23% slower** than full-precision — CoreML's GPU handles native float16 better than dequantizing int8.

## Compute Unit Experiments

### Text Decoder Compute Units (large-v2)

| Compute Units | Tok/s | Notes |
|--------------|-------|-------|
| cpuAndGPU | 14.37 | Best — GPU memory bandwidth wins |
| cpuAndNeuralEngine | 10.06 | ~30% slower |
| all | 5.68 | Worst — scheduling overhead |

### GPU Benefits All Model Sizes on High-Core Macs

| Model | NE (tok/s) | GPU (tok/s) | GPU Speedup |
|-------|-----------|------------|-------------|
| large-v2 (32 layers) | 10.06 | 14.37 | +43% |
| turbo (4 layers) | 55.97 | 78.57 | +40% |
| distil-large-v3 (2 layers) | 53.87 | 65.79 | +22% |

**Finding:** The previous auto-selection only chose cpuAndGPU for "large" models. GPU is faster for ALL model sizes on ≥14 GPU core Macs. Fixed in `TranscribeCLIUtils.swift`.

### Other CLI Settings

| Setting | Tok/s | Notes |
|---------|-------|-------|
| + prefill cache + language=en | 14.27 | Marginal improvement |
| - VAD chunking (none) | 11.34 | Slower — processes silence |

## Time Budget Analysis (large-v2)

Per-token breakdown (~72ms total):
- CoreML model inference: ~68ms (95%)
- KV cache update (memcpy): ~0.5ms
- Token sampling/logits: ~0.5ms
- Swift overhead (allocations, logging, progress): ~3ms

The decoder reads ~1.5GB of weights per forward pass. At ~400 GB/s M1 Max bandwidth, the theoretical floor is ~3.75ms/token, but real-world GPU utilization is much lower.

## Code Optimizations Implemented

### 1. Text Decoder Hot Loop (TextDecoder.swift)

- **Hoisted invariants** (`isDebugLogging`, `hasCallback`, model guard) outside token loop
- **Guarded debug logging** — avoids eager `tokenizer.decode()` string interpolation
- **Guarded progress computation** — CLI (no callback) skips per-token tokenizer.decode + zlib compressionRatio, eliminating O(n²) per-window overhead
- **Direct MLFeatureProvider access** — extracts logits/cache directly instead of through TextDecoderOutput wrapper

Net impact on large-v2: ~1-2% (within noise, CoreML dominates). Significant for callback-heavy streaming paths.

### 2. Auto Compute Unit Selection (TranscribeCLIUtils.swift)

- **Broadened GPU selection** to all models on ≥14 GPU core Macs (was large-only)
- **Added modelPath fallback** for model name detection when --model-path is used without --model
- Impact: **+22-43% faster** for turbo/distil models that were incorrectly using Neural Engine

## CoreML Configuration Experiments (All Negative)

| Setting | Result | Notes |
|---------|--------|-------|
| allowLowPrecisionAccumulationOnGPU | 28% regression | Forces different GPU path |
| specializationStrategy = .fastPrediction | 28% regression | Hurts autoregressive decoding |
| Both combined | 28% regression | No compounding |

**Conclusion:** Default MLModelConfiguration is already optimal.

## Pipeline Overlap Attempt (Reverted)

Attempted to overlap audio encoder (Neural Engine) with text decoder (GPU) across windows:
- **Problem 1:** Seek prediction mismatch — predicted next seek differs from actual by ~640 samples
- **Problem 2:** VAD chunking produces single-window chunks, eliminating pipelining opportunity
- **Reverted** — no benefit for typical audio

## Remaining Opportunities

### Speculative Decoding
- Use tiny/base model to draft N tokens, verify with large model in batched forward pass
- Potential: 2-3x speedup depending on acceptance rate
- Requires: parallel model loading, batch verification logic

### MLState-Based Models (macOS 15+/iOS 18+)
- CoreML state API keeps KV cache GPU-resident between predictions
- Eliminates CPU↔GPU cache transfer overhead
- Requires: model re-export with state annotations
- WhisperKit has MLState/MLTensor infrastructure but not used in main inference path

### Recommendations
1. **For speed:** Use `large-v3-v20240930_turbo` — 5.5x faster than large-v2 with near-identical English quality
2. **For balanced speed/quality:** Use `distil-large-v3` — 4.6x faster, excellent quality
3. **For max quality:** Use `large-v2` or `large-v3` — 14 tok/s, best multilingual support
4. **Avoid quantized models** on GPU-rich Macs — CoreML's native float16 is faster than int8 dequantization
