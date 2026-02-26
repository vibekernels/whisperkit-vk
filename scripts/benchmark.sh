#!/bin/bash
set -euo pipefail

# WhisperKit transcription benchmark
# Usage:
#   ./Scripts/benchmark.sh                          # Run all models on short audio
#   ./Scripts/benchmark.sh --audio path/to/file.wav # Custom audio file
#   ./Scripts/benchmark.sh --models "tiny large-v2" # Specific models
#   ./Scripts/benchmark.sh --long                   # Include long audio (60s)
#   ./Scripts/benchmark.sh --all                    # All models, short + long audio

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="$REPO_DIR/.build/release/whisperkit-cli"

SHORT_AUDIO="$REPO_DIR/Tests/WhisperKitTests/Resources/jfk.wav"
LONG_AUDIO="$REPO_DIR/Tests/WhisperKitTests/Resources/ted_60.m4a"

DEFAULT_MODELS="tiny base small large-v2 large-v3"
MODELS=""
AUDIO_FILES=""
INCLUDE_LONG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --audio)
            AUDIO_FILES="$AUDIO_FILES $2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --long)
            INCLUDE_LONG=true
            shift
            ;;
        --all)
            MODELS="$DEFAULT_MODELS"
            INCLUDE_LONG=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --audio FILE    Audio file to benchmark (can be repeated)"
            echo "  --models LIST   Space-separated list of models (default: $DEFAULT_MODELS)"
            echo "  --long          Include long audio benchmark (ted_60.m4a, 60s)"
            echo "  --all           All models with short + long audio"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

MODELS="${MODELS:-$DEFAULT_MODELS}"

if [ -z "$AUDIO_FILES" ]; then
    AUDIO_FILES="$SHORT_AUDIO"
    if $INCLUDE_LONG; then
        AUDIO_FILES="$AUDIO_FILES $LONG_AUDIO"
    fi
fi

# Build CLI if needed
if [ ! -f "$CLI" ]; then
    echo "Building whisperkit-cli (release)..."
    cd "$REPO_DIR"
    swift build -c release --product whisperkit-cli
    echo ""
fi

echo "============================================"
echo "WhisperKit Benchmark"
echo "============================================"
echo "Models: $MODELS"
echo "Audio:  $AUDIO_FILES"
echo "============================================"
echo ""

for audio in $AUDIO_FILES; do
    filename=$(basename "$audio")
    echo "--------------------------------------------"
    echo "Audio: $filename"
    echo "--------------------------------------------"
    echo ""

    for model in $MODELS; do
        echo "=== $model ==="
        "$CLI" transcribe \
            --audio-path "$audio" \
            --model "$model" \
            --verbose
        echo ""
    done
done
