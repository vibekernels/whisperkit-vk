This is an experimental fork of [WhisperKit](https://github.com/argmaxinc/WhisperKit) with AI-driven optimizations. It is not intended for production use.

## Build

```bash
swift build
```

## Benchmarking

Build the CLI in release mode:

```bash
swift build -c release --product whisperkit-cli
```

### Short audio (~11s)

```bash
.build/release/whisperkit-cli transcribe \
  --audio-path Tests/WhisperKitTests/Resources/jfk.wav \
  --model large-v2 \
  --verbose
```

### Long audio (~60s)

```bash
.build/release/whisperkit-cli transcribe \
  --audio-path Tests/WhisperKitTests/Resources/ted_60.m4a \
  --model large-v2 \
  --verbose
```

### Comparing models

Run the same file across different models to compare speed and accuracy:

```bash
for model in tiny base small large-v2 large-v3; do
  echo "=== $model ==="
  .build/release/whisperkit-cli transcribe \
    --audio-path Tests/WhisperKitTests/Resources/jfk.wav \
    --model $model \
    --verbose
done
```

The `--verbose` flag prints tokens per second, real-time factor, and speed factor after each transcription.

## License

WhisperKit is released under the [MIT License](LICENSE).

## Upstream

Based on [argmaxinc/WhisperKit](https://github.com/argmaxinc/WhisperKit).
