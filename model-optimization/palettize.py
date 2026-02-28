import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)
import os, sys, shutil

SRC = "/Users/m1/whisperkit-vk/model-optimization/original/mlpackages"
DST = "/Users/m1/whisperkit-vk/model-optimization/palettized"
os.makedirs(DST, exist_ok=True)

nbits = int(sys.argv[1]) if len(sys.argv) > 1 else 6

# Models to palettize (ANE models with large weights)
targets = ["Encoder", "Decoder", "JointDecision"]

for name in targets:
    src_path = os.path.join(SRC, f"{name}.mlpackage")
    dst_path = os.path.join(DST, f"{name}.mlpackage")
    
    if not os.path.exists(src_path):
        print(f"  Skipping {name} (not found)")
        continue
    
    print(f"Loading {name}...")
    model = ct.models.MLModel(src_path)
    
    print(f"  Palettizing {name} to {nbits}-bit...")
    config = OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=nbits, mode="kmeans")
    )
    palettized = palettize_weights(model, config)
    
    print(f"  Saving to {dst_path}...")
    palettized.save(dst_path)
    
    orig_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(src_path) for f in fns
    )
    new_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(dst_path) for f in fns
    )
    print(f"  {name}: {orig_size/1e6:.1f}MB -> {new_size/1e6:.1f}MB ({new_size/orig_size:.1%})")

# Copy non-palettized models as-is
for name in ["Preprocessor", "MelEncoder"]:
    src_path = os.path.join(SRC, f"{name}.mlpackage")
    dst_path = os.path.join(DST, f"{name}.mlpackage")
    if os.path.exists(src_path) and not os.path.exists(dst_path):
        print(f"Copying {name} (no palettization)...")
        shutil.copytree(src_path, dst_path)

print("\nDone!")
