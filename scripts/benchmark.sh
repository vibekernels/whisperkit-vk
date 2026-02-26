#!/bin/bash
set -euo pipefail

# WhisperKit transcription benchmark
# Usage:
#   ./scripts/benchmark.sh                          # Run large-v2 on short + long audio
#   ./scripts/benchmark.sh --audio path/to/file.wav # Custom audio file
#   ./scripts/benchmark.sh --models "tiny large-v2" # Specific models
#   ./scripts/benchmark.sh --long                   # Include long audio (60s)
#   ./scripts/benchmark.sh --all                    # All models, short + long audio

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="$REPO_DIR/.build/release/whisperkit-cli"

SHORT_AUDIO="$REPO_DIR/Tests/WhisperKitTests/Resources/jfk.wav"
LONG_AUDIO="$REPO_DIR/Tests/WhisperKitTests/Resources/ted_60.m4a"

DEFAULT_MODELS="large-v2"
MODELS=""
AUDIO_FILES=""
INCLUDE_LONG=true
FAILURES=0

# Expected transcription outputs (trimmed, lowercased for comparison)
EXPECTED_jfk_wav="and so my fellow americans, ask not what your country can do for you, ask what you can do for your country."
EXPECTED_ted_60_m4a="so in college, i was a government major, which means i had to write a lot of papers. now, when a normal student writes a paper, they might spread the work out a little like this. so, you know, you get started maybe a little slowly, but you get enough done in the first week that, with some heavier days later on, everything gets done and things stay civil. and i would want to do that like that. that would be the plan. i would have it all ready to go but then actually the paper would come along and then i would kind of do this and that would happen every single paper but then came my 90 page senior thesis paper. you're supposed to spend a year on i knew for a paper like that my normal workflow was not an option. it was way too big a project so i planned things out and i decided i kind of had to go something like this this is how the year would go. so i'd start off light and i'd bump it up"

# Normalize text: trim whitespace, lowercase, collapse whitespace, strip trailing period
normalize() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr -s '[:space:]' ' ' | sed 's/^ //;s/ $//' | sed 's/\.$//'
}

check_output() {
    local filename="$1"
    local actual="$2"

    # Map filename to expected variable
    local varname="EXPECTED_${filename//[.-]/_}"
    local expected="${!varname:-}"

    if [ -z "$expected" ]; then
        return
    fi

    local norm_actual
    local norm_expected
    norm_actual=$(normalize "$actual")
    norm_expected=$(normalize "$expected")

    if [ "$norm_actual" = "$norm_expected" ]; then
        echo "  PASS: output matches expected text"
    else
        echo "  FAIL: output does not match expected text"
        echo "  Expected: $norm_expected"
        echo "  Actual:   $norm_actual"
        FAILURES=$((FAILURES + 1))
    fi
}

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
        output=$("$CLI" transcribe \
            --audio-path "$audio" \
            --model "$model" \
            --verbose 2>&1)
        echo "$output"

        # Extract transcription text (lines after "Transcription of <filename>:")
        transcription=$(echo "$output" | sed -n "/^Transcription of ${filename}:/,\$p" | tail -n +2 | sed '/^$/d')
        check_output "$filename" "$transcription"
        echo ""
    done
done

if [ "$FAILURES" -gt 0 ]; then
    echo "============================================"
    echo "FAILED: $FAILURES output check(s) did not match"
    echo "============================================"
    exit 1
fi
