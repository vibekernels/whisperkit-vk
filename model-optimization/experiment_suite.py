#!/usr/bin/env python3
"""
Overnight optimization experiment suite for Parakeet TDT models.
Tests pruning, advanced palettization, and combined techniques.
"""

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OpMagnitudePrunerConfig,
    OptimizationConfig,
    palettize_weights,
    prune_weights,
    decompress_weights,
    get_weights_metadata,
)
import os
import shutil
import sys
import json
import time

SRC = "/Users/m1/whisperkit-vk/model-optimization/original/mlpackages"
RESULTS_DIR = "/Users/m1/whisperkit-vk/model-optimization/experiments"
COMPILED_DIR = "/Users/m1/whisperkit-vk/model-optimization/experiments/compiled"
MODEL_CACHE = "/Users/m1/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(COMPILED_DIR, exist_ok=True)

results = []


def save_model(model, name, experiment_name):
    """Save an mlpackage and return its size."""
    out_dir = os.path.join(RESULTS_DIR, experiment_name)
    out_path = os.path.join(out_dir, f"{name}.mlpackage")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_dir, exist_ok=True)
    model.save(out_path)
    size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(out_path)
        for f in fns
    )
    return out_path, size


def compile_model(mlpackage_path, name):
    """Compile mlpackage to mlmodelc."""
    out_path = os.path.join(COMPILED_DIR, f"{name}.mlmodelc")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.system(f'xcrun coremlcompiler compile "{mlpackage_path}" "{COMPILED_DIR}/" 2>/dev/null')
    return out_path


def install_models(models_dict):
    """Install compiled models into cache. models_dict = {name: compiled_path}"""
    for name, compiled_path in models_dict.items():
        dest = os.path.join(MODEL_CACHE, f"{name}.mlmodelc")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(compiled_path, dest)


def backup_originals():
    """Backup original models."""
    for name in ["Encoder", "Decoder", "JointDecision"]:
        src = os.path.join(MODEL_CACHE, f"{name}.mlmodelc")
        bak = os.path.join(MODEL_CACHE, f"{name}.mlmodelc.original")
        if not os.path.exists(bak) and os.path.exists(src):
            shutil.copytree(src, bak)


def restore_originals():
    """Restore original models from backup."""
    for name in ["Encoder", "Decoder", "JointDecision"]:
        bak = os.path.join(MODEL_CACHE, f"{name}.mlmodelc.original")
        dest = os.path.join(MODEL_CACHE, f"{name}.mlmodelc")
        if os.path.exists(bak):
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(bak, dest)


def load_model(name):
    """Load an mlpackage model."""
    path = os.path.join(SRC, f"{name}.mlpackage")
    return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)


# ============================================================
# EXPERIMENT 1: Pruning + 6-bit Palettization (Decoder + Joint)
# ============================================================
def exp_prune_then_palettize(target_sparsity=0.5, nbits=6):
    exp_name = f"prune{int(target_sparsity*100)}_pal{nbits}"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  Prune to {target_sparsity*100}% sparsity, then {nbits}-bit palettize")
    print(f"{'='*60}")

    models_to_install = {}

    for name in ["Decoder", "JointDecision"]:
        print(f"\n  Loading {name}...")
        model = load_model(name)

        print(f"  Pruning {name} to {target_sparsity*100}% sparsity...")
        prune_config = OptimizationConfig(
            global_config=OpMagnitudePrunerConfig(
                target_sparsity=target_sparsity,
                weight_threshold=2048,
            )
        )
        pruned = prune_weights(model, prune_config)

        print(f"  Palettizing {name} to {nbits}-bit...")
        pal_config = OptimizationConfig(
            global_config=OpPalettizerConfig(nbits=nbits, mode="kmeans")
        )
        optimized = palettize_weights(pruned, pal_config)

        pkg_path, size = save_model(optimized, name, exp_name)
        print(f"  {name}: {size/1e6:.1f}MB")

        compiled = compile_model(pkg_path, name)
        models_to_install[name] = compiled

    return exp_name, models_to_install


# ============================================================
# EXPERIMENT 2: Per-channel scale palettization (better accuracy)
# ============================================================
def exp_perchannel_palettize(nbits=4):
    exp_name = f"perchannel_pal{nbits}"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  {nbits}-bit palettization with per-channel scale")
    print(f"{'='*60}")

    models_to_install = {}

    for name in ["Decoder", "JointDecision"]:
        print(f"\n  Loading {name}...")
        model = load_model(name)

        print(f"  Palettizing {name} to {nbits}-bit with per-channel scale...")
        config = OptimizationConfig(
            global_config=OpPalettizerConfig(
                nbits=nbits,
                mode="kmeans",
                enable_per_channel_scale=True,
            )
        )
        optimized = palettize_weights(model, config)

        pkg_path, size = save_model(optimized, name, exp_name)
        print(f"  {name}: {size/1e6:.1f}MB")

        compiled = compile_model(pkg_path, name)
        models_to_install[name] = compiled

    return exp_name, models_to_install


# ============================================================
# EXPERIMENT 3: Vector palettization (cluster_dim > 1)
# ============================================================
def exp_vector_palettize(nbits=4, cluster_dim=2):
    exp_name = f"vector_pal{nbits}_dim{cluster_dim}"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  {nbits}-bit vector palettization, cluster_dim={cluster_dim}")
    print(f"{'='*60}")

    models_to_install = {}

    for name in ["Decoder", "JointDecision"]:
        print(f"\n  Loading {name}...")
        model = load_model(name)

        print(f"  Vector palettizing {name}...")
        config = OptimizationConfig(
            global_config=OpPalettizerConfig(
                nbits=nbits,
                mode="kmeans",
                cluster_dim=cluster_dim,
            )
        )
        optimized = palettize_weights(model, config)

        pkg_path, size = save_model(optimized, name, exp_name)
        print(f"  {name}: {size/1e6:.1f}MB")

        compiled = compile_model(pkg_path, name)
        models_to_install[name] = compiled

    return exp_name, models_to_install


# ============================================================
# EXPERIMENT 4: 4-bit encoder with per-channel scale
# ============================================================
def exp_encoder_4bit_perchannel():
    exp_name = "encoder_4bit_perchannel"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  Decompress encoder 8-bit → fp16, then 4-bit with per-channel scale")
    print(f"{'='*60}")

    models_to_install = {}

    print("\n  Loading Encoder...")
    model = load_model("Encoder")

    print("  Decompressing 8-bit LUT → fp16...")
    decompressed = decompress_weights(model)

    print("  Re-palettizing to 4-bit with per-channel scale...")
    config = OptimizationConfig(
        global_config=OpPalettizerConfig(
            nbits=4,
            mode="kmeans",
            enable_per_channel_scale=True,
        )
    )
    optimized = palettize_weights(decompressed, config)

    pkg_path, size = save_model(optimized, "Encoder", exp_name)
    print(f"  Encoder: {size/1e6:.1f}MB")

    compiled = compile_model(pkg_path, "Encoder")
    models_to_install["Encoder"] = compiled

    # Also include 6-bit decoder+joint from our best config
    pal_dir = "/Users/m1/whisperkit-vk/model-optimization/compiled"
    for name in ["Decoder", "JointDecision"]:
        src = os.path.join(pal_dir, f"{name}.mlmodelc")
        if os.path.exists(src):
            models_to_install[name] = src

    return exp_name, models_to_install


# ============================================================
# EXPERIMENT 5: Aggressive pruning on encoder (decompress → prune → 8-bit)
# ============================================================
def exp_encoder_prune():
    exp_name = "encoder_prune30_pal8"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  Decompress encoder, prune 30%, re-palettize 8-bit")
    print(f"{'='*60}")

    models_to_install = {}

    print("\n  Loading Encoder...")
    model = load_model("Encoder")

    print("  Decompressing 8-bit LUT → fp16...")
    decompressed = decompress_weights(model)

    print("  Pruning 30% of weights...")
    prune_config = OptimizationConfig(
        global_config=OpMagnitudePrunerConfig(
            target_sparsity=0.3,
            weight_threshold=2048,
        )
    )
    pruned = prune_weights(decompressed, prune_config)

    print("  Re-palettizing to 8-bit...")
    pal_config = OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=8, mode="kmeans")
    )
    optimized = palettize_weights(pruned, pal_config)

    pkg_path, size = save_model(optimized, "Encoder", exp_name)
    print(f"  Encoder: {size/1e6:.1f}MB")

    compiled = compile_model(pkg_path, "Encoder")
    models_to_install["Encoder"] = compiled

    # Include 6-bit decoder+joint
    pal_dir = "/Users/m1/whisperkit-vk/model-optimization/compiled"
    for name in ["Decoder", "JointDecision"]:
        src = os.path.join(pal_dir, f"{name}.mlmodelc")
        if os.path.exists(src):
            models_to_install[name] = src

    return exp_name, models_to_install


# ============================================================
# EXPERIMENT 6: Combined pruning + per-channel 4-bit (Decoder+Joint)
# ============================================================
def exp_prune_perchannel_4bit():
    exp_name = "prune30_perchannel_pal4"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  Prune 30% + 4-bit per-channel palettization (Decoder+Joint)")
    print(f"{'='*60}")

    models_to_install = {}

    for name in ["Decoder", "JointDecision"]:
        print(f"\n  Loading {name}...")
        model = load_model(name)

        print(f"  Pruning {name} to 30% sparsity...")
        prune_config = OptimizationConfig(
            global_config=OpMagnitudePrunerConfig(
                target_sparsity=0.3,
                weight_threshold=2048,
            )
        )
        pruned = prune_weights(model, prune_config)

        print(f"  Palettizing {name} to 4-bit with per-channel scale...")
        pal_config = OptimizationConfig(
            global_config=OpPalettizerConfig(
                nbits=4,
                mode="kmeans",
                enable_per_channel_scale=True,
            )
        )
        optimized = palettize_weights(pruned, pal_config)

        pkg_path, size = save_model(optimized, name, exp_name)
        print(f"  {name}: {size/1e6:.1f}MB")

        compiled = compile_model(pkg_path, name)
        models_to_install[name] = compiled

    return exp_name, models_to_install


# ============================================================
# EXPERIMENT 7: 2-bit palettization with per-channel scale (extreme)
# ============================================================
def exp_2bit_perchannel():
    exp_name = "perchannel_pal2"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  2-bit palettization with per-channel scale (Decoder+Joint)")
    print(f"{'='*60}")

    models_to_install = {}

    for name in ["Decoder", "JointDecision"]:
        print(f"\n  Loading {name}...")
        model = load_model(name)

        print(f"  Palettizing {name} to 2-bit with per-channel scale...")
        config = OptimizationConfig(
            global_config=OpPalettizerConfig(
                nbits=2,
                mode="kmeans",
                enable_per_channel_scale=True,
            )
        )
        optimized = palettize_weights(model, config)

        pkg_path, size = save_model(optimized, name, exp_name)
        print(f"  {name}: {size/1e6:.1f}MB")

        compiled = compile_model(pkg_path, name)
        models_to_install[name] = compiled

    return exp_name, models_to_install


if __name__ == "__main__":
    # Run specific experiment if given as argument, otherwise run all
    experiment_arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    experiments = {
        "prune50_pal6": lambda: exp_prune_then_palettize(0.5, 6),
        "prune30_pal6": lambda: exp_prune_then_palettize(0.3, 6),
        "perchannel_pal4": lambda: exp_perchannel_palettize(4),
        "perchannel_pal6": lambda: exp_perchannel_palettize(6),
        "vector_pal4_dim2": lambda: exp_vector_palettize(4, 2),
        "encoder_4bit_perchannel": exp_encoder_4bit_perchannel,
        "encoder_prune30_pal8": exp_encoder_prune,
        "prune30_perchannel_pal4": exp_prune_perchannel_4bit,
        "perchannel_pal2": exp_2bit_perchannel,
    }

    backup_originals()

    if experiment_arg == "all":
        to_run = list(experiments.keys())
    else:
        to_run = [experiment_arg]

    for exp_key in to_run:
        if exp_key not in experiments:
            print(f"Unknown experiment: {exp_key}")
            continue

        try:
            exp_name, models = experiments[exp_key]()
            install_models(models)
            print(f"\n  Installed {exp_name}. Ready for benchmarking.")

            result = {"experiment": exp_name, "status": "ready"}
            results.append(result)

        except Exception as e:
            print(f"\n  ERROR in {exp_key}: {e}")
            results.append({"experiment": exp_key, "status": "error", "error": str(e)})

        # Save intermediate results
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Restore originals at end
    restore_originals()
    print("\n\nAll experiments complete. Originals restored.")
    print(json.dumps(results, indent=2))
