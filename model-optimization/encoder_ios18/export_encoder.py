#!/usr/bin/env python3
"""
Re-export Parakeet TDT 0.6B Encoder from PyTorch with iOS 18 target.

The existing Encoder (433MB, spec v8/iOS 17) has 294 constexpr_lut_to_dense ops
whose 'shape' parameter doesn't exist in the iOS 18 op definition. This prevents
post-hoc palettization of the Encoder with iOS 18 features (per-channel-scale, etc).

Re-exporting from PyTorch with minimum_deployment_target=ct.target.iOS18 produces
a clean iOS 18 spec model that can be freely palettized.

Requirements (two venvs needed due to NeMo/coremltools torch version conflicts):
  Venv 1 (NeMo): torch 2.10+, nemo_toolkit[asr] — for loading model & tracing
  Venv 2 (Convert): torch 2.7, coremltools 9.0, scikit-learn — for CoreML conversion

Workflow:
  1. (Venv 1) Load NeMo model, trace, save TorchScript .pt file
  2. (Venv 2) Load .pt, convert to CoreML with monkey-patched _cast bug fix

Critical findings:
  - MUST use compute_precision=FLOAT16 for ANE performance (FP32 is 4x slower)
  - coremltools 9.0 has a bug in _cast() for multi-dim scalar arrays — needs monkey-patch
  - 6-bit palettized encoder (446MB) is same size as original 8-bit (446MB) after compilation
  - Performance is within noise of original (~180x RT vs ~177x RT)
  - No speed or size benefit from re-export; keep original unless iOS 18 features needed
"""

import os
import sys
import time
import shutil
import numpy as np

import torch
import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)

# ── Monkey-patch coremltools _cast bug ──
# The _cast function in coremltools 9.0 fails when x.val is a multi-dimensional
# numpy array with all dims == 1 (e.g., shape [1,1,1]). Fix: extract scalar first.
import coremltools.converters.mil.frontend.torch.ops as torch_ops
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb


def _cast_fixed(context, node, dtype, dtype_name):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
        raise ValueError("input to cast must be either a scalar or a length 1 tensor")
    if x.can_be_folded_to_const():
        val = x.val
        if isinstance(val, np.ndarray):
            val = val.item()
        if not isinstance(val, dtype):
            res = mb.const(val=dtype(val), name=node.name)
        else:
            res = x
    elif len(x.shape) > 0:
        x = mb.squeeze(x=x, name=node.name + "_item")
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    else:
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    context.add(res, node.name)


torch_ops._cast = _cast_fixed

# ── Configuration ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR
ORIGINAL_ENCODER = os.path.join(SCRIPT_DIR, "..", "original", "mlpackages", "Encoder.mlpackage")
TRACED_MODEL = os.path.join(SCRIPT_DIR, "encoder_traced.pt")

MEL_SHAPE = (1, 128, 1501)
MEL_LEN_SHAPE = (1,)
ENCODER_OUT_SHAPE = (1, 1024, 188)
ENCODER_LEN_SHAPE = (1,)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dir_size(path):
    return sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(path) for f in fns
    )


# ── Step 1: Load TorchScript model ──
# (Pre-saved from NeMo venv — see step1_trace_nemo.py or run manually)
if not os.path.exists(TRACED_MODEL):
    log("ERROR: TorchScript model not found. Run Step 1 in the NeMo venv first:")
    log(f"  Expected: {TRACED_MODEL}")
    log("  Steps:")
    log("    source ../encoder_export_venv/bin/activate")
    log("    python -c 'import nemo.collections.asr as nemo_asr; ...'")
    sys.exit(1)

log("Step 1: Loading TorchScript model...")
traced = torch.jit.load(TRACED_MODEL)
traced.eval()
mel = torch.randn(*MEL_SHAPE)
mel_len = torch.tensor([MEL_SHAPE[2]], dtype=torch.long)
with torch.no_grad():
    out, l = traced(mel, mel_len)
log(f"  shapes: encoder={list(out.shape)}, length={list(l.shape)}")


# ── Step 2: Convert to CoreML with iOS 18 + FLOAT16 ──
log("Step 2: Converting to CoreML (iOS 18, FLOAT16)...")
t0 = time.time()
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="mel", shape=MEL_SHAPE, dtype=np.float32),
        ct.TensorType(name="mel_length", shape=MEL_LEN_SHAPE, dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name="encoder", dtype=np.float32),
        ct.TensorType(name="encoder_length", dtype=np.int32),
    ],
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT16,  # CRITICAL for ANE performance
)
elapsed = time.time() - t0
log(f"  Done in {elapsed:.1f}s, spec={mlmodel.get_spec().specificationVersion}")

fp_path = os.path.join(OUTPUT_DIR, "Encoder_fp16.mlpackage")
if os.path.exists(fp_path):
    shutil.rmtree(fp_path)
mlmodel.save(fp_path)
fp_size = dir_size(fp_path)
log(f"  FP16 model: {fp_size/1e6:.1f}MB")


# ── Step 3: Validate vs original ──
log("Step 3: Numerical validation vs original...")
if os.path.exists(ORIGINAL_ENCODER):
    orig_model = ct.models.MLModel(ORIGINAL_ENCODER, compute_units=ct.ComputeUnit.CPU_ONLY)
    mel_val = np.random.randn(*MEL_SHAPE).astype(np.float32)
    mel_len_val = np.array([MEL_SHAPE[2]], dtype=np.int32)
    orig_out = orig_model.predict({"mel": mel_val, "mel_length": mel_len_val})
    new_out = mlmodel.predict({"mel": mel_val, "mel_length": mel_len_val})
    for key in ["encoder", "encoder_length"]:
        if key in orig_out and key in new_out:
            diff = np.abs(orig_out[key] - new_out[key]).max()
            log(f"  {key} max_abs_diff: {diff:.6f}")


# ── Step 4: 6-bit palettize ──
log("Step 4: 6-bit palettization (kmeans)...")
t0 = time.time()
config = OptimizationConfig(global_config=OpPalettizerConfig(nbits=6, mode="kmeans"))
pal = palettize_weights(mlmodel, config)
elapsed = time.time() - t0
pal_path = os.path.join(OUTPUT_DIR, "Encoder_6bit_fp16.mlpackage")
if os.path.exists(pal_path):
    shutil.rmtree(pal_path)
pal.save(pal_path)
pal_size = dir_size(pal_path)
log(f"  6-bit: {fp_size/1e6:.1f}MB -> {pal_size/1e6:.1f}MB ({pal_size/fp_size:.1%}) in {elapsed:.1f}s")

if os.path.exists(ORIGINAL_ENCODER):
    pal_out = pal.predict({"mel": mel_val, "mel_length": mel_len_val})
    for key in ["encoder", "encoder_length"]:
        if key in orig_out and key in pal_out:
            diff = np.abs(orig_out[key] - pal_out[key]).max()
            log(f"  6-bit {key} max_abs_diff vs orig: {diff:.6f}")


# ── Summary ──
orig_size = dir_size(ORIGINAL_ENCODER) if os.path.exists(ORIGINAL_ENCODER) else 0
log(f"\n{'='*60}")
log("SUMMARY")
log(f"{'='*60}")
log(f"  Original 8-bit (iOS 17):    {orig_size/1e6:.1f}MB")
log(f"  New FP16 (iOS 18):          {fp_size/1e6:.1f}MB")
log(f"  New 6-bit FP16 (iOS 18):    {pal_size/1e6:.1f}MB")
log("\nNext steps:")
log("  xcrun coremlcompiler compile Encoder_6bit_fp16.mlpackage compiled/")
log("  cp -r compiled/Encoder_6bit_fp16.mlmodelc ~/Library/Application\\ Support/FluidAudio/Models/.../Encoder.mlmodelc")
log("  swift run whisperkit-cli transcribe --engine parakeet --verbose --audio-path ...")
