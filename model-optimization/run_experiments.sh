#!/bin/bash
# Overnight experiment runner: optimize models → benchmark → record results
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/Users/m1/whisperkit-vk"
MODEL_CACHE="/Users/m1/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml"
AUDIO="$PROJECT_DIR/Tests/WhisperKitTests/Resources/ted_60.m4a"
CLI="$PROJECT_DIR/.build/release/whisperkit-cli"
RESULTS_FILE="$SCRIPT_DIR/experiments/benchmark_results.jsonl"

mkdir -p "$SCRIPT_DIR/experiments"

# Backup original models
backup_originals() {
    for model in Encoder Decoder JointDecision; do
        bak="$MODEL_CACHE/$model.mlmodelc.original"
        src="$MODEL_CACHE/$model.mlmodelc"
        if [ ! -d "$bak" ] && [ -d "$src" ]; then
            cp -r "$src" "$bak"
        fi
    done
}

# Restore original models
restore_originals() {
    for model in Encoder Decoder JointDecision; do
        bak="$MODEL_CACHE/$model.mlmodelc.original"
        dest="$MODEL_CACHE/$model.mlmodelc"
        if [ -d "$bak" ]; then
            rm -rf "$dest"
            cp -r "$bak" "$dest"
        fi
    done
}

# Benchmark: 5 runs, extract median inference time and speed factor
benchmark() {
    local exp_name="$1"
    local times=()
    local speeds=()
    local text=""

    for i in 1 2 3 4 5; do
        output=$("$CLI" transcribe --audio-path "$AUDIO" --verbose 2>&1)
        t=$(echo "$output" | grep "ASR inference:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        s=$(echo "$output" | grep "Speed factor:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        times+=("$t")
        speeds+=("$s")
    done

    # Get transcription text (non-verbose)
    text=$("$CLI" transcribe --audio-path "$AUDIO" 2>&1)

    # Sort times for median
    sorted_times=($(printf '%s\n' "${times[@]}" | sort -n))
    sorted_speeds=($(printf '%s\n' "${speeds[@]}" | sort -n))
    median_time="${sorted_times[2]}"
    median_speed="${sorted_speeds[2]}"

    # Save result
    echo "{\"experiment\":\"$exp_name\",\"median_ms\":$median_time,\"median_speed\":$median_speed,\"runs\":[${times[0]},${times[1]},${times[2]},${times[3]},${times[4]}]}" >> "$RESULTS_FILE"

    # Save transcription for diff
    echo "$text" > "$SCRIPT_DIR/experiments/${exp_name}_transcript.txt"

    echo "  Result: ${median_time}ms, ${median_speed}x RT"
    echo "  Runs: ${times[*]}"
}

# ============================================================
# Main
# ============================================================
echo "=== Overnight Optimization Experiments ==="
echo "Started: $(date)"
echo ""

backup_originals

# First, establish current best baseline (6-bit decoder+joint)
echo "--- Baseline (current best: 6-bit Decoder+Joint) ---"
restore_originals
# Install our current 6-bit models
COMPILED="$SCRIPT_DIR/compiled"
for model in Decoder JointDecision; do
    if [ -d "$COMPILED/$model.mlmodelc" ]; then
        rm -rf "$MODEL_CACHE/$model.mlmodelc"
        cp -r "$COMPILED/$model.mlmodelc" "$MODEL_CACHE/$model.mlmodelc"
    fi
done
benchmark "baseline_6bit"

# Experiment list
EXPERIMENTS=(
    "prune30_pal6"
    "prune50_pal6"
    "perchannel_pal4"
    "perchannel_pal6"
    "vector_pal4_dim2"
    "prune30_perchannel_pal4"
    "perchannel_pal2"
    "encoder_4bit_perchannel"
    "encoder_prune30_pal8"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "--- Experiment: $exp ---"

    # Restore originals before each experiment
    restore_originals

    # Run the optimization
    python3 "$SCRIPT_DIR/experiment_suite.py" "$exp" 2>&1 | grep -v "Running\|ops/s\|passes/s\|RuntimeWarning"

    # Benchmark
    benchmark "$exp"
done

# Final restore
restore_originals
# Reinstall 6-bit decoder+joint (our current best)
for model in Decoder JointDecision; do
    if [ -d "$COMPILED/$model.mlmodelc" ]; then
        rm -rf "$MODEL_CACHE/$model.mlmodelc"
        cp -r "$COMPILED/$model.mlmodelc" "$MODEL_CACHE/$model.mlmodelc"
    fi
done

echo ""
echo "=== All experiments complete ==="
echo "Finished: $(date)"
echo ""
echo "Results:"
cat "$RESULTS_FILE"
