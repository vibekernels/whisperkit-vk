#!/usr/bin/env python3
"""
iOS 18 Palettization Experiment
================================
Test if we can unlock iOS 18 palettization features (per-channel scale,
vector palettization) by bumping the CoreML model's spec version from 8
(iOS 17) to 9 (iOS 18), WITHOUT re-exporting from PyTorch.
"""

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)
import os
import sys
import shutil
import time
import traceback

SRC = "/Users/m1/whisperkit-vk/model-optimization/original/mlpackages"
DST = "/Users/m1/whisperkit-vk/model-optimization/ios18_experiment"
LOG = os.path.join(DST, "experiment_log.txt")

os.makedirs(DST, exist_ok=True)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

def save_log():
    with open(LOG, "w") as f:
        f.write("\n".join(log_lines))

log("=" * 70)
log("iOS 18 PALETTIZATION EXPERIMENT")
log(f"coremltools version: {ct.__version__}")
log("=" * 70)

# ── Step 1: Check current spec versions ──
log("\n── STEP 1: Current Model Spec Versions ──")
for name in ["Encoder", "Decoder", "JointDecision", "Preprocessor", "MelEncoder"]:
    path = os.path.join(SRC, f"{name}.mlpackage")
    if os.path.exists(path):
        model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
        spec = model.get_spec()
        log(f"  {name}: specificationVersion={spec.specificationVersion}")
    else:
        log(f"  {name}: NOT FOUND")

# ── Step 2: Attempt palettize_weights with enable_per_channel_scale (NO spec bump) ──
log("\n── STEP 2: Try per-channel-scale palettization WITHOUT spec bump ──")
try:
    model = ct.models.MLModel(os.path.join(SRC, "Decoder.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    config = OptimizationConfig(
        global_config=OpPalettizerConfig(
            nbits=4,
            mode="kmeans",
            enable_per_channel_scale=True,
        )
    )
    optimized = palettize_weights(model, config)
    log("  RESULT: SUCCESS (unexpected!)")
except Exception as e:
    log(f"  RESULT: FAILED as expected")
    log(f"  Error: {e}")

# ── Step 3: Bump spec version to 9 (iOS 18), then try ──
log("\n── STEP 3: Bump spec version 8→9 (iOS 18), then palettize ──")

targets = ["Encoder", "Decoder", "JointDecision"]
results = {}

for name in targets:
    src_path = os.path.join(SRC, f"{name}.mlpackage")
    dst_path = os.path.join(DST, f"{name}.mlpackage")
    
    if not os.path.exists(src_path):
        log(f"  {name}: SKIPPED (not found)")
        continue
    
    log(f"\n  Processing {name}...")
    
    try:
        # Load original model
        model = ct.models.MLModel(src_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        spec = model.get_spec()
        old_ver = spec.specificationVersion
        
        # Bump spec version to iOS 18 (9)
        spec.specificationVersion = 9
        log(f"    Bumped specificationVersion: {old_ver} → {spec.specificationVersion}")
        
        # Re-create MLModel from bumped spec (must pass weights_dir for mlProgram)
        model_v9 = ct.models.MLModel(spec, compute_units=ct.ComputeUnit.CPU_ONLY, weights_dir=model.weights_dir)

        # Configure 4-bit palettization with per-channel-scale
        config = OptimizationConfig(
            global_config=OpPalettizerConfig(
                nbits=4,
                mode="kmeans",
                enable_per_channel_scale=True,
            )
        )
        
        t0 = time.time()
        optimized = palettize_weights(model_v9, config)
        elapsed = time.time() - t0
        
        log(f"    Palettization: SUCCESS in {elapsed:.1f}s")
        
        # Check output spec version
        opt_spec = optimized.get_spec()
        log(f"    Output specificationVersion: {opt_spec.specificationVersion}")
        
        # Save
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        optimized.save(dst_path)
        
        # Size comparison
        orig_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(src_path) for f in fns
        )
        new_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(dst_path) for f in fns
        )
        log(f"    Size: {orig_size/1e6:.1f}MB → {new_size/1e6:.1f}MB ({new_size/orig_size:.1%})")
        
        results[name] = {"status": "SUCCESS", "orig_size": orig_size, "new_size": new_size, "time": elapsed}
        
    except Exception as e:
        log(f"    FAILED: {e}")
        log(f"    Traceback:\n{traceback.format_exc()}")
        results[name] = {"status": "FAILED", "error": str(e)}

# ── Step 4: Also try vector palettization (cluster_dim > 1) ──
log("\n── STEP 4: Try vector palettization (cluster_dim=2) with iOS 18 spec bump ──")
try:
    model = ct.models.MLModel(os.path.join(SRC, "Decoder.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    spec.specificationVersion = 9
    model_v9 = ct.models.MLModel(spec, compute_units=ct.ComputeUnit.CPU_ONLY, weights_dir=model.weights_dir)

    config = OptimizationConfig(
        global_config=OpPalettizerConfig(
            nbits=4,
            mode="kmeans",
            cluster_dim=2,  # vector palettization
            enable_per_channel_scale=True,
        )
    )
    
    t0 = time.time()
    optimized = palettize_weights(model_v9, config)
    elapsed = time.time() - t0
    log(f"  Vector palettization (cluster_dim=2): SUCCESS in {elapsed:.1f}s")
    
    vec_path = os.path.join(DST, "Decoder_vector.mlpackage")
    if os.path.exists(vec_path):
        shutil.rmtree(vec_path)
    optimized.save(vec_path)
    vec_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(vec_path) for f in fns
    )
    log(f"  Vector palettized Decoder size: {vec_size/1e6:.1f}MB")
    
except Exception as e:
    log(f"  Vector palettization FAILED: {e}")
    log(f"  Traceback:\n{traceback.format_exc()}")

# Copy non-palettized models
log("\n── Copying non-palettized models ──")
for name in ["Preprocessor", "MelEncoder"]:
    src_path = os.path.join(SRC, f"{name}.mlpackage")
    dst_path = os.path.join(DST, f"{name}.mlpackage")
    if os.path.exists(src_path) and not os.path.exists(dst_path):
        shutil.copytree(src_path, dst_path)
        log(f"  Copied {name}")

# Summary
log("\n── SUMMARY ──")
for name, r in results.items():
    if r["status"] == "SUCCESS":
        log(f"  {name}: {r['orig_size']/1e6:.1f}MB → {r['new_size']/1e6:.1f}MB in {r['time']:.1f}s")
    else:
        log(f"  {name}: FAILED - {r['error']}")

save_log()
log(f"\nLog saved to {LOG}")
