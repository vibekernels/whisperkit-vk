//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// ``TranscriptionEngine`` adapter that forwards calls to a ``WhisperKit`` instance.
public final class WhisperEngine: TranscriptionEngine, @unchecked Sendable {
    public let engineType: EngineType = .whisper
    public let whisperKit: WhisperKit

    public var modelState: ModelState {
        whisperKit.modelState
    }

    public init(_ whisperKit: WhisperKit) {
        self.whisperKit = whisperKit
    }

    public func transcribe(
        audioPath: String,
        decodeOptions: DecodingOptions?
    ) async throws -> [TranscriptionResult] {
        try await whisperKit.transcribe(audioPath: audioPath, decodeOptions: decodeOptions)
    }

    public func transcribe(
        audioArray: [Float],
        decodeOptions: DecodingOptions?
    ) async throws -> [TranscriptionResult] {
        try await whisperKit.transcribe(audioArray: audioArray, decodeOptions: decodeOptions)
    }
}
