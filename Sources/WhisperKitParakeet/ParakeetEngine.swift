//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import Foundation
import WhisperKit

@_exported import FluidAudio

/// ``TranscriptionEngine`` adapter that wraps FluidAudio's Parakeet TDT model.
@available(macOS 14.0, iOS 17.0, *)
public final class ParakeetEngine: TranscriptionEngine, @unchecked Sendable {
    public let engineType: EngineType = .parakeet

    public private(set) var modelState: ModelState = .unloaded

    /// The Parakeet model version to use (e.g. `.v3` for multilingual).
    public let modelVersion: AsrModelVersion

    /// Enable verbose timing breakdown.
    public var verbose: Bool = false

    private var asrManager: AsrManager?

    public init(modelVersion: AsrModelVersion = .v3) {
        self.modelVersion = modelVersion
    }

    /// Downloads (if needed) and loads the Parakeet TDT models.
    public func loadModels() async throws {
        modelState = .downloading
        let models = try await AsrModels.downloadAndLoad(version: modelVersion)

        modelState = .loading
        // Disable streaming: use in-memory ChunkProcessor for all file sizes.
        // The streaming path reads chunks from disk which adds I/O per chunk;
        // in-memory processing is faster when the full audio fits in RAM.
        let config = ASRConfig(streamingEnabled: false)
        let manager = AsrManager(config: config)
        try await manager.initialize(models: models)
        asrManager = manager

        // Pre-warm CoreML models by running a 1-second silent inference.
        // This JIT-compiles Metal/ANE shaders and warms caches so the
        // first real transcription doesn't pay the cold-start penalty.
        let warmupSamples = [Float](repeating: 0, count: 16_000)
        _ = try? await manager.transcribe(warmupSamples, source: .system)

        modelState = .loaded
    }

    // MARK: - TranscriptionEngine

    public func transcribe(
        audioPath: String,
        decodeOptions: DecodingOptions?
    ) async throws -> [TranscriptionResult] {
        let manager = try requireManager()
        let url = URL(fileURLWithPath: audioPath)
        let pipelineStart = CFAbsoluteTimeGetCurrent()

        let t0 = CFAbsoluteTimeGetCurrent()

        // Convert to 16 kHz mono Float32. Duration is computed from the
        // sample count, avoiding a separate AVAudioFile probe.
        let converter = AudioConverter()
        let samples = try converter.resampleAudioFile(url)
        let audioDuration = Double(samples.count) / 16000.0
        let t1 = CFAbsoluteTimeGetCurrent()

        let asrResult = try await manager.transcribe(samples, source: .system)
        let t2 = CFAbsoluteTimeGetCurrent()

        let result = mapResult(asrResult, audioDuration: audioDuration, pipelineStart: pipelineStart)
        let t3 = CFAbsoluteTimeGetCurrent()

        if verbose {
            print(String(format: "  [parakeet] Audio load+resample: %.1f ms", (t1 - t0) * 1000))
            print(String(format: "  [parakeet] ASR inference:       %.1f ms", (t2 - t1) * 1000))
            print(String(format: "  [parakeet] Result mapping:      %.1f ms", (t3 - t2) * 1000))
            print(String(format: "  [parakeet] Total:               %.1f ms", (t3 - t0) * 1000))
            print(String(format: "  [parakeet] Samples: %d (%.1fs @ 16kHz)", samples.count, audioDuration))
        }

        return [result]
    }

    public func transcribe(
        audioArray: [Float],
        decodeOptions: DecodingOptions?
    ) async throws -> [TranscriptionResult] {
        let manager = try requireManager()
        let pipelineStart = CFAbsoluteTimeGetCurrent()
        let asrResult = try await manager.transcribe(audioArray, source: .system)
        let duration = asrResult.duration > 0 ? asrResult.duration : TimeInterval(audioArray.count) / 16000.0
        return [mapResult(asrResult, audioDuration: duration, pipelineStart: pipelineStart)]
    }

    // MARK: - Private

    private func requireManager() throws -> AsrManager {
        guard let manager = asrManager else {
            throw WhisperError.modelsUnavailable("Parakeet models not loaded. Call loadModels() first.")
        }
        return manager
    }

    /// Maps a FluidAudio ``ASRResult`` to a WhisperKit ``TranscriptionResult``.
    private func mapResult(_ asr: ASRResult, audioDuration: TimeInterval, pipelineStart: CFAbsoluteTime) -> TranscriptionResult {
        let timingsList = asr.tokenTimings ?? []

        let words: [WordTiming]? = timingsList.isEmpty ? nil : timingsList.map { timing in
            WordTiming(
                word: timing.token,
                tokens: [timing.tokenId],
                start: Float(timing.startTime),
                end: Float(timing.endTime),
                probability: Float(timing.confidence)
            )
        }

        let segment = TranscriptionSegment(
            id: 0,
            seek: 0,
            start: 0,
            end: Float(audioDuration),
            text: asr.text,
            tokens: timingsList.map { $0.tokenId },
            tokenLogProbs: [[:]], // not available from Parakeet
            temperature: 0,
            avgLogprob: 0,
            compressionRatio: 1.0,
            noSpeechProb: 0,
            words: words
        )

        var timings = TranscriptionTimings()
        timings.pipelineStart = pipelineStart
        timings.firstTokenTime = pipelineStart // non-autoregressive; no meaningful first-token time
        timings.inputAudioSeconds = audioDuration
        timings.fullPipeline = asr.processingTime

        return TranscriptionResult(
            text: asr.text,
            segments: [segment],
            language: "en", // Parakeet v2 is English; v3 auto-detects but doesn't expose language
            timings: timings
        )
    }
}
