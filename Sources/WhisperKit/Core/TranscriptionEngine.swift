//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// Identifies which ASR backend is in use.
public enum EngineType: String, Sendable, CaseIterable {
    case whisper
    case parakeet
}

/// Unified transcription interface shared by Whisper and Parakeet backends.
public protocol TranscriptionEngine: AnyObject, Sendable {
    /// Which backend this engine uses.
    var engineType: EngineType { get }

    /// Current model loading state.
    var modelState: ModelState { get }

    /// Transcribe an audio file at the given path.
    func transcribe(
        audioPath: String,
        decodeOptions: DecodingOptions?
    ) async throws -> [TranscriptionResult]

    /// Transcribe raw audio samples.
    func transcribe(
        audioArray: [Float],
        decodeOptions: DecodingOptions?
    ) async throws -> [TranscriptionResult]
}
