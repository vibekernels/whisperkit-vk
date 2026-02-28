# WhisperKit Transcription Speed Research

## Cross-Engine Comparison (10 GPU core Mac)

Benchmarks on Apple Silicon (10 GPU cores), ted_60.m4a (60s), macOS, release builds.

| Engine | Model | Backend | Total Time | Speed Factor | RTF |
|---|---|---|---|---|---|
| FluidAudio 0.10.1 | parakeet-tdt-0.6b-v3 (CoreML, 6-bit decoder) | ANE | **0.31s** | **194x** | 0.005 |
| whisper.cpp 1.8.3 | large-v2 (GGML) | Metal GPU | 12.4s | 4.8x | 0.21 |
| WhisperKit (repo) | large-v2 (CoreML) | ANE | 14.7s | 4.1x | 0.24 |
| WhisperKit (repo) | large-v2 (CoreML) | GPU | 27.2s | 2.2x | 0.45 |

**Key takeaways:**
- **FluidAudio's Parakeet TDT is ~40x faster than whisper.cpp** and ~47x faster than WhisperKit on the same hardware. Parakeet is non-autoregressive (0.6B params) vs Whisper's autoregressive decoding (1.5B params) — a fundamentally different architecture.
- **whisper.cpp is ~18% faster than WhisperKit** on large-v2. whisper.cpp uses Metal GPU with **batched decoding** (multiple tokens per forward pass, 5.7ms/tok for 1358 tokens), while WhisperKit uses CoreML with **autoregressive decoding** (one token per forward pass, ~62ms/tok). Batched decoding reads the ~1.5GB weight matrix once per batch rather than once per token, amortizing the memory bandwidth cost. whisper.cpp has no ANE access (custom GGML Metal shaders, not CoreML). The ~10.8x per-token advantage is partially offset by a slower encoder (2.85s vs WhisperKit's faster CoreML encoder), netting ~18% end-to-end.
- **WhisperKit ANE (4.1x RT) is competitive with whisper.cpp Metal GPU (4.8x RT)** despite autoregressive decoding, because the ANE has a different memory subsystem optimized for sequential inference workloads.
- **ANE is ~85% faster than GPU** for WhisperKit's CoreML text decoder on 10-core Macs (opposite of high-core Macs — see below).
- Transcription quality is comparable across all engines. FluidAudio captures slightly more filler words ("uh") and has better punctuation.

## M1 Max Hardware (32 GPU cores)

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

### ANE vs GPU is Hardware-Dependent (10 GPU core Mac)

| Compute Units | Tok/s | Notes |
|--------------|-------|-------|
| cpuAndNeuralEngine | 16.14 | Best — ANE wins on low-core Macs |
| cpuAndGPU | 8.44 | ~48% slower |

**Finding:** The GPU vs ANE preference reverses on lower-core Macs. On ≥14 GPU cores, GPU wins (+43%). On 10 GPU cores, ANE wins (+85%). The auto-selection threshold of 14 cores is well-calibrated — but the current code also forces `cpuAndGPU` for all large models regardless of core count (`else if large`), which hurts 10-core Macs.

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

### 3. Micro-Optimizations Attempted (Reverted — No Measurable Impact)

Tested eliminating per-token array copies in `GreedyTokenSampler.update()` (returning single-element arrays instead of rebuilding full accumulated arrays) and gating progress callback computation behind nil check. A/B benchmark (3 runs each, ANE, large-v2) showed identical performance: origin avg 16.09 tok/s vs optimized avg 16.09 tok/s. Changes reverted as the per-token overhead is dominated by CoreML inference (~62ms/token), not array operations or string formatting (~microseconds).

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

## Batched Decoding Feasibility (Investigated — Not Viable)

whisper.cpp achieves 5.7ms/tok via batched decoding (multiple tokens per forward pass), while WhisperKit gets ~62ms/tok with autoregressive single-token decoding — a 10.8x per-token gap. Investigated whether the same approach could be applied to WhisperKit.

**Finding: CoreML models are incompatible with batched decoding.**

The WhisperKit CoreML text decoder models are compiled with fixed single-token input shapes and zero shape flexibility:

| Input/Output | Shape | Constraint |
|---|---|---|
| `input_ids` | `[1]` (scalar Int32) | Not `[1, N]` — single token only |
| `logits` output | `[1, 1, 51865]` | Single position output |
| `key_cache_updates` | `[1, 40960, 1, 1]` | Single token KV slice |
| `kv_cache_update_mask` | One `1` value per pass | Single token insertion |
| Model metadata | `hasShapeFlexibility: "0"` | All inputs fixed |

The model literally cannot accept more than one token per forward pass. Enabling batched decoding would require:

1. **Re-export PyTorch → CoreML** with flexible batch dimension on `input_ids` (`[1, N]`), logits (`[1, N, vocab_size]`), and KV cache updates (`[1, embed_dim, 1, N]`)
2. **Modify attention mechanism** in the model to compute N parallel queries
3. **Update Swift decode loop** — batch token sampling, N-token KV cache inserts
4. **Rebuild and redistribute all model artifacts** on HuggingFace (`argmaxinc/whisperkit-coreml`)

This is a model architecture + export change, not a code optimization. WhisperKit consumes pre-built CoreML models — the re-export would need to happen upstream in the model conversion pipeline.

**Conclusion:** The batched decoding speed gap is a model format limitation. No code changes in this repo can address it.

## Unified Engine Integration (Parakeet + Whisper)

Integrated FluidAudio's Parakeet TDT as a first-class ASR backend alongside Whisper via a new `TranscriptionEngine` protocol abstraction. Parakeet is the default engine.

### Architecture

```
TranscriptionEngine (protocol)  ← Sources/WhisperKit/Core/TranscriptionEngine.swift
├── WhisperEngine                ← Sources/WhisperKit/Core/WhisperEngine.swift
└── ParakeetEngine               ← Sources/WhisperKitParakeet/ParakeetEngine.swift
```

- `WhisperEngine`: thin adapter forwarding to the existing `WhisperKit` class
- `ParakeetEngine`: wraps FluidAudio `AsrManager`, maps `ASRResult` → `TranscriptionResult`
- `WhisperKitParakeet` is a separate SPM target (depends on WhisperKit + FluidAudio) to keep FluidAudio out of the core library

### CLI Usage

```bash
# Parakeet (default)
whisperkit-cli transcribe --audio-path audio.m4a

# Whisper
whisperkit-cli transcribe --audio-path audio.m4a --engine whisper --model large-v2

# Parakeet English-only model
whisperkit-cli transcribe --audio-path audio.m4a --parakeet-model-version v2
```

### Integrated Benchmark (10 GPU core Mac, ted_60.m4a 60s)

| Metric | Homebrew Whisper (large-v3) | Local Parakeet (v3) |
|--------|---------------------------|---------------------|
| Inference time | 4.2s | 0.36s |
| Speed factor | 14.6x RT | 175x RT |
| Wall time (models cached) | 63.2s | 0.7s |
| Wall time (cold start) | 63.2s | 12.3s |
| CPU usage | 72% | 16% |

Parakeet is **~12x faster inference** and **~90x faster wall-clock** (cached) than Whisper large-v3 on the same hardware.

### Parakeet Hot Path Profiling

Instrumented `ParakeetEngine.transcribe(audioPath:)` to identify optimization targets (ted_60.m4a, 60s, 5 runs):

| Phase | Time | % |
|-------|------|---|
| Audio load + resample (16kHz mono) | 9ms | 2.5% |
| **ASR inference (preprocessor + encoder + TDT decoder)** | **356ms** | **96.8%** |
| Result mapping (ASRResult → TranscriptionResult) | 0.1ms | 0.03% |
| **Total** | **365ms** | |

The inference phase is entirely inside FluidAudio's CoreML pipeline:
- **Preprocessor** (mel spectrogram): ~5-8ms — CPU-only, cached MLArray allocation
- **Encoder** (FastConformer): ~15-20ms — ANE-optimized
- **TDT Decoder** (joint + LSTM loop): ~35-45ms per chunk — ANE, with blank-token reuse optimization (2-3x speedup on silence frames)
- **Chunk overhead**: 60s audio = 4 chunks (14.96s each, 2s overlap), sequential processing, token merging at boundaries

### Parakeet Adapter Optimizations Applied

1. **Model pre-warming** (+8% speed factor on first inference): Run 1s silent dummy transcription during `loadModels()` to JIT-compile CoreML Metal/ANE shaders. First-run speed factor improved from 152x → 165x.
2. **Eliminated AVURLAsset duration probe** (−2.5ms): Compute audio duration from sample count (`samples.count / 16000.0`) instead of async `AVURLAsset.load(.duration)` call.
3. **Pre-converted audio path**: Use FluidAudio's `AudioConverter` to resample to 16kHz mono before calling `transcribe([Float])`, so FluidAudio's samples path skips its internal converter.
4. **Disabled streaming I/O**: Set `ASRConfig(streamingEnabled: false)` so the URL path uses in-memory `ChunkProcessor` instead of disk-backed streaming for all file sizes.

**Net improvement: 151.9x → 175x RT** (~15% faster, 5-run median on ted_60.m4a).

### Optimization Paths Investigated But Not Viable

- **URL vs samples path**: Tested passing the file URL directly to FluidAudio vs pre-converting to samples. No meaningful difference (~370ms both ways) — FluidAudio's internal AudioConverter is already efficient.
- **Streaming vs non-streaming**: Disabling streaming mode (in-memory ChunkProcessor vs disk-backed) showed no improvement — disk I/O is not the bottleneck.
- **Parallel chunk processing**: Not possible from outside FluidAudio. Chunks share the same `AsrManager` and decoder state, processing is sequential. Even if parallelized, ANE executes a single pipeline, so concurrent CoreML predictions would queue.
- **Reducing chunk overlap**: The 2.0s overlap is hardcoded in FluidAudio's `ChunkProcessor`. Reducing it would require forking FluidAudio and risks token misalignment at boundaries.
- **TDT decoder optimizations**: The decoder already implements blank-token reuse (skipping LSTM updates during silence), BLAS-accelerated frame copies, ANE-aligned memory, and zero-copy encoder frame views. No opportunities from the adapter layer.

### Result Mapping Notes

- Parakeet is non-autoregressive — produces a single segment for the full audio
- `ASRResult.tokenTimings` → `[WordTiming]` with per-token start/end/confidence
- Whisper-specific fields (tokens, tokenLogProbs, compressionRatio, noSpeechProb, temperature) use defaults
- `decodeOptions` is ignored by Parakeet (no temperature, language selection, etc.)
- FluidAudio's streaming chunk path returns `duration=0` — worked around by computing duration from sample count

### Not Yet Implemented

- Streaming support for Parakeet (`StreamingAsrManager`)
- Server CLI (`OpenAIHandler`) engine support
- Language selection for Parakeet (model version determines it)
- Translate task for Parakeet (not supported by the model)

## FluidAudio Internal Optimization Experiment (v0.12.1)

Forked FluidAudio v0.12.1 and applied 5 internal optimizations (single-source decoder reset, cached projection layout, reusable chunk buffers, reusable DP table, fused audio padding). Result: **+0.7%** improvement (426.1 → 422.7ms median). Reverted to upstream v0.10.1 because:

1. v0.12.1 regressed from 175x → 141x RT vs v0.10.1 (new vocabulary boosting, CTC keyword spotting code paths)
2. The 0.7% fork optimization doesn't offset the 24% version regression
3. 96.8% of time is in CoreML kernels — untouchable from Swift code

**Decision:** Pinned `Package.swift` to `FluidAudio exact: "0.10.1"` for maximum speed (174x RT).

## Parakeet Compute Unit Comparison (10 GPU core Mac, FluidAudio v0.10.1)

Added `--parakeet-compute-units` flag to test ANE vs GPU for Parakeet TDT.

| Compute Units | Inference (median) | Speed Factor | Notes |
|---|---|---|---|
| `cpuAndNeuralEngine` (default) | **345.9 ms** | **174.2x RT** | ANE handles encoder + decoder + joint |
| `cpuAndGPU` | 620.2 ms | 96.9x RT | 79% slower than ANE |

Raw runs (5 each, ms):
```
ANE: 367.5  345.9  345.5  341.9  346.7
GPU: 620.2  619.7  620.3  619.9  621.8
```

**Finding:** ANE is **79% faster** than GPU for Parakeet on 10-core Macs. This matches the Whisper text decoder result (ANE +85% on 10-core). Transcription output is near-identical across compute units. GPU is not a path to faster Parakeet inference on this hardware.

## CoreML Model Palettization (Decoder + JointDecision)

Downloaded `.mlpackage` source files from HuggingFace (`FluidInference/parakeet-tdt-0.6b-v3-coreml`) and applied weight palettization via `coremltools.optimize.coreml.palettize_weights()`.

### Model weight analysis

| Component | Params | Original | Already Palettized? |
|---|---|---|---|
| Encoder | 444M (97%) | 433MB | **Yes** — 8-bit LUT by FluidInference |
| Decoder | 12M (2.5%) | 23MB | No — float16 |
| JointDecision | 6M (1.3%) | 12MB | No — float16 |

The Encoder is already optimally compressed at 8-bit. Attempted 8→6-bit re-palettization (decompress to fp16 via `decompress_weights()`, then re-palettize with k-means) but file size was unchanged (445.8→445.9MB) due to byte-alignment padding of 6-bit indices, and added run-to-run variance. **Not worth it for the Encoder.**

Decoder and JointDecision are float16 — viable targets for palettization.

### Palettization results

| Config | Decoder Size | Joint Size | Inference (median) | Speed Factor | Quality |
|---|---|---|---|---|---|
| Baseline (fp16) | 23MB | 12MB | 345.9 ms | 174.2x RT | Reference |
| **6-bit Decoder+Joint** | **8.9MB** | **4.8MB** | **310.5 ms** | **193.8x RT** | Identical (punctuation-only diffs) |
| 4-bit Decoder+Joint | 5.9MB | 3.2MB | 313.8 ms | 191.8x RT | Minor errors ("I I would", "90 page") |
| 6-bit Encoder+Decoder+Joint | 445.9MB | 4.8MB | 310.4 ms | 194.0x RT | Same as above, more variance |

Raw runs (6-bit Decoder+Joint, 5 each, ms):
```
Baseline: 367.5  345.9  345.5  341.9  346.7
6-bit:    336.6  310.0  310.6  310.5  311.7
```

**Finding:** 6-bit palettization of Decoder + JointDecision yields **+11% speedup** (174x → 194x RT) with no meaningful quality loss. The smaller LUT weights reduce memory bandwidth in the hot TDT decode loop (~600+ LSTM forward passes per 60s audio). 4-bit pushes too far — introduces token duplication artifacts and is actually slightly slower (ANE may need extra cycles for 4-bit depalettization). Re-palettizing the Encoder from 8→6-bit has no effect because byte-aligned storage negates the compression.

### How to apply

```bash
pip3 install coremltools huggingface_hub
python3 model-optimization/palettize.py 6  # creates palettized .mlpackage files
xcrun coremlcompiler compile palettized/Decoder.mlpackage output/
xcrun coremlcompiler compile palettized/JointDecision.mlpackage output/
# Copy .mlmodelc files to ~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml/
```

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
1. **For maximum speed:** Use `--engine parakeet` (default) — 194x RT with 6-bit palettized decoder (174x stock), ~40x faster than Whisper, comparable quality
2. **For Whisper speed:** Use `--engine whisper --model large-v3-v20240930_turbo` — 5.5x faster than large-v2
3. **For balanced speed/quality:** Use `--engine whisper --model distil-large-v3` — 4.6x faster, excellent quality
4. **For max quality:** Use `--engine whisper --model large-v2` or `large-v3` — 14 tok/s, best multilingual support
5. **Avoid quantized models** on GPU-rich Macs — CoreML's native float16 is faster than int8 dequantization
6. **Fix auto-selection for <14 GPU core Macs** — the `else if large` branch forces cpuAndGPU which is ~48% slower than ANE on these machines
