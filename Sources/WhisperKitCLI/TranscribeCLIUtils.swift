//  Copyright © 2025 Argmax, Inc. All rights reserved.
//  For licensing see accompanying LICENSE.md file.

import Foundation
import CoreML
import IOKit
@preconcurrency import WhisperKit

internal class TranscribeCLIUtils {

    /// Returns the number of GPU cores on the current device using IOKit, or 0 on failure.
    private static func gpuCoreCount() -> Int {
        let matching = IOServiceMatching("AGXAccelerator")
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator) == KERN_SUCCESS else {
            return 0
        }
        defer { IOObjectRelease(iterator) }

        let entry = IOIteratorNext(iterator)
        guard entry != 0 else { return 0 }
        defer { IOObjectRelease(entry) }

        guard let prop = IORegistryEntryCreateCFProperty(entry, "gpu-core-count" as CFString, kCFAllocatorDefault, 0) else {
            return 0
        }
        return (prop.takeRetainedValue() as? Int) ?? 0
    }

    /// Returns true if the model name indicates a "large" variant.
    private static func isLargeModel(_ modelName: String?) -> Bool {
        guard let name = modelName?.lowercased() else { return false }
        return name.contains("large")
    }

    /// Creates WhisperKit configuration from CLI arguments
    static func createWhisperKitConfig(from arguments: TranscribeCLIArguments) -> WhisperKitConfig {
        var audioEncoderComputeUnits = arguments.audioEncoderComputeUnits.asMLComputeUnits
        var textDecoderComputeUnits = arguments.textDecoderComputeUnits.asMLComputeUnits

        // Resolve auto compute units for text decoder based on hardware + model size
        if arguments.textDecoderComputeUnits == .auto {
            let cores = gpuCoreCount()
            let large = isLargeModel(arguments.model)
            if large && cores >= 14 {
                textDecoderComputeUnits = .cpuAndGPU
            } else {
                textDecoderComputeUnits = .cpuAndNeuralEngine
            }
            if arguments.verbose {
                print("[auto] Text decoder compute: \(textDecoderComputeUnits == .cpuAndGPU ? "cpuAndGPU" : "cpuAndNeuralEngine") (model large: \(large), GPU cores: \(cores))")
            }
        }

        // Use gpu for audio encoder on macOS below 14
        if audioEncoderComputeUnits == .cpuAndNeuralEngine {
            if #unavailable(macOS 14.0) {
                audioEncoderComputeUnits = .cpuAndGPU
            }
        }

        let computeOptions = ModelComputeOptions(
            audioEncoderCompute: audioEncoderComputeUnits,
            textDecoderCompute: textDecoderComputeUnits
        )

        let downloadTokenizerFolder: URL? = arguments.downloadTokenizerPath.map { URL(filePath: $0) }
        let downloadModelFolder: URL? = arguments.downloadModelPath.map { URL(filePath: $0) }
        let modelName: String? = arguments.model.map { arguments.modelPrefix + "*" + $0 }

        return WhisperKitConfig(
            model: modelName,
            downloadBase: downloadModelFolder,
            modelFolder: arguments.modelPath,
            tokenizerFolder: downloadTokenizerFolder,
            computeOptions: computeOptions,
            verbose: arguments.verbose,
            logLevel: arguments.verbose ? .debug : .info,
            prewarm: false,
            load: true,
            useBackgroundDownloadSession: false
        )
    }
    
    /// Creates DecodingOptions from CLI arguments and task
    static func createDecodingOptions(from arguments: TranscribeCLIArguments, task: DecodingTask) -> DecodingOptions {
        let options = DecodingOptions(
            verbose: arguments.verbose,
            task: task,
            language: arguments.language,
            temperature: arguments.temperature,
            temperatureIncrementOnFallback: arguments.temperatureIncrementOnFallback,
            temperatureFallbackCount: arguments.temperatureFallbackCount,
            topK: arguments.bestOf,
            usePrefillPrompt: arguments.usePrefillPrompt || arguments.language != nil || task == .translate,
            usePrefillCache: arguments.usePrefillCache,
            skipSpecialTokens: arguments.skipSpecialTokens,
            withoutTimestamps: arguments.withoutTimestamps,
            wordTimestamps: arguments.wordTimestamps,
            clipTimestamps: arguments.clipTimestamps,
            supressTokens: arguments.supressTokens,
            compressionRatioThreshold: arguments.compressionRatioThreshold ?? 2.4,
            logProbThreshold: arguments.logprobThreshold ?? -1.0,
            firstTokenLogProbThreshold: arguments.firstTokenLogProbThreshold,
            noSpeechThreshold: arguments.noSpeechThreshold ?? 0.6,
            concurrentWorkerCount: arguments.concurrentWorkerCount,
            chunkingStrategy: ChunkingStrategy(rawValue: arguments.chunkingStrategy)
        )

        return options
    }
}
