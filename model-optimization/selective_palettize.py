#!/usr/bin/env python3
"""
Selective layer palettization: different bit widths for embedding vs LSTM weights.

Decoder structure (7 weight tensors, 11.8M params):
  - module_prediction_embed_weight_to_fp16: embedding (8193x640) = 44.4% of params
  - concat_1_to_fp16: LSTM layer 0 weight_ih (2560x640) = 13.9%
  - concat_2_to_fp16: LSTM layer 0 weight_hh (2560x640) = 13.9%
  - concat_4_to_fp16: LSTM layer 1 weight_ih (2560x640) = 13.9%
  - concat_5_to_fp16: LSTM layer 1 weight_hh (2560x640) = 13.9%
  - concat_0_to_fp16: LSTM layer 0 bias (2560) = ~0%
  - concat_3_to_fp16: LSTM layer 1 bias (2560) = ~0%
"""

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)
import os
import shutil
import sys

SRC = "/Users/m1/whisperkit-vk/model-optimization/original/mlpackages"
OUT = "/Users/m1/whisperkit-vk/model-optimization/selective"
COMPILED = "/Users/m1/whisperkit-vk/model-optimization/selective/compiled"

os.makedirs(OUT, exist_ok=True)
os.makedirs(COMPILED, exist_ok=True)

STRATEGIES = {
    # Strategy A: Protect embedding (6-bit), compress LSTM (4-bit)
    "embed6_lstm4": OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=4, mode="kmeans"),
        op_type_configs={
            "gather": OpPalettizerConfig(nbits=6, mode="kmeans"),
        },
    ),
    # Strategy B: Protect embedding + recurrent (6-bit), compress input proj (4-bit)
    "embed6_hh6_ih4": OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=4, mode="kmeans"),
        op_name_configs={
            "module_prediction_embed_weight_to_fp16": OpPalettizerConfig(nbits=6, mode="kmeans"),
            "concat_2_to_fp16": OpPalettizerConfig(nbits=6, mode="kmeans"),  # LSTM 0 weight_hh
            "concat_5_to_fp16": OpPalettizerConfig(nbits=6, mode="kmeans"),  # LSTM 1 weight_hh
        },
    ),
    # Strategy C: All 4-bit (no per-channel scale, just plain 4-bit for comparison)
    "all_4bit": OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=4, mode="kmeans"),
    ),
    # Strategy D: Embedding 8-bit (preserve quality), LSTM 4-bit (aggressive)
    "embed8_lstm4": OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=4, mode="kmeans"),
        op_type_configs={
            "gather": OpPalettizerConfig(nbits=8, mode="kmeans"),
        },
    ),
}


def process_strategy(name, config):
    print(f"\n{'='*60}")
    print(f"Strategy: {name}")
    print(f"{'='*60}")

    for model_name in ["Decoder", "JointDecision"]:
        print(f"\n  Loading {model_name}...")
        model = ct.models.MLModel(
            os.path.join(SRC, f"{model_name}.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        print(f"  Palettizing {model_name} with {name} strategy...")
        optimized = palettize_weights(model, config)

        # Save
        out_dir = os.path.join(OUT, name)
        out_path = os.path.join(out_dir, f"{model_name}.mlpackage")
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_dir, exist_ok=True)
        optimized.save(out_path)

        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(out_path)
            for f in fns
        )
        print(f"  {model_name}: {size/1e6:.1f}MB")

        # Compile
        compiled_path = os.path.join(COMPILED, name, f"{model_name}.mlmodelc")
        compiled_dir = os.path.join(COMPILED, name)
        os.makedirs(compiled_dir, exist_ok=True)
        if os.path.exists(compiled_path):
            shutil.rmtree(compiled_path)
        os.system(f'xcrun coremlcompiler compile "{out_path}" "{compiled_dir}/" 2>/dev/null')
        print(f"  Compiled: {compiled_path}")


if __name__ == "__main__":
    strategy_arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if strategy_arg == "all":
        for name, config in STRATEGIES.items():
            process_strategy(name, config)
    elif strategy_arg in STRATEGIES:
        process_strategy(strategy_arg, STRATEGIES[strategy_arg])
    else:
        print(f"Unknown strategy: {strategy_arg}")
        print(f"Available: {', '.join(STRATEGIES.keys())}")
        sys.exit(1)

    print("\n\nAll strategies complete.")
