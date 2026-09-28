//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 TonesFly. Fork addition (DET-2).

import CoreML
@testable import WhisperKit
import XCTest

/// DET-2: temperature-fallback draws are reproducible from a seed and keep
/// the top-k multinomial distribution; T=0 is untouched.
final class SeededSamplerTests: XCTestCase {

    private let logitValues: [Float] = [2, 1.5, 1, 0.5, 0, -1, -2, -3]

    private func logits() throws -> MLMultiArray {
        let type: MLMultiArrayDataType = MemoryLayout<FloatType>.size == 2 ? .float16 : .float32
        let array = try MLMultiArray(shape: [1, 1, NSNumber(value: logitValues.count)], dataType: type)
        for (i, v) in logitValues.enumerated() { array[i] = NSNumber(value: v) }
        return array
    }

    private func sampler(temperature: FloatType, seed: UInt64?) -> GreedyTokenSampler {
        GreedyTokenSampler(temperature: temperature, eotToken: 999, decodingOptions: DecodingOptions(topK: 5), seed: seed)
    }

    private func draws(_ s: GreedyTokenSampler, _ n: Int, _ logits: MLMultiArray) -> [Int] {
        var tokens: [Int] = []
        for _ in 0..<n { tokens = s.update(tokens: tokens, logits: logits, logProbs: []).tokens }
        return tokens
    }

    func testSameSeedReproducesTheDrawSequence() throws {
        let l = try logits()
        let a = draws(sampler(temperature: 1, seed: 42), 64, l)
        XCTAssertEqual(a, draws(sampler(temperature: 1, seed: 42), 64, l))
        XCTAssertNotEqual(a, draws(sampler(temperature: 1, seed: 43), 64, l))
        XCTAssertEqual(Set(a).count > 1, true, "T>0 must still sample")
    }

    func testSeededDrawsKeepTheTopKDistribution() throws {
        let l = try logits()
        let n = 20_000
        var counts = [Int](repeating: 0, count: logitValues.count)
        for seed in 0..<UInt64(n) {
            counts[draws(sampler(temperature: 1, seed: SamplingSeed.attempt(window: seed, temperature: 1)), 1, l)[0]] += 1
        }
        XCTAssertEqual(counts[5...].reduce(0, +), 0, "outside top-k")
        let weights = logitValues[0..<5].map { exp(Double($0)) }
        let total = weights.reduce(0, +)
        let chi2 = zip(counts[0..<5], weights).reduce(0.0) { acc, pair in
            let expected = Double(n) * pair.1 / total
            return acc + pow(Double(pair.0) - expected, 2) / expected
        }
        XCTAssertLessThan(chi2, 18.47, "chi-square 4 dof, p=0.001; counts \(counts)")
    }

    func testGreedyIsUnchangedBySeeding() throws {
        let l = try logits()
        XCTAssertEqual(draws(sampler(temperature: 0, seed: 7), 8, l), draws(sampler(temperature: 0, seed: nil), 8, l))
    }

    func testWindowSeedIsContentOnlyAndAttemptSeedsDiffer() {
        let audio = (0..<16_000).map { Float(sin(Double($0) * 0.01)) }
        XCTAssertEqual(SamplingSeed.window(audio), SamplingSeed.window(Array(audio)))
        var nudged = audio
        nudged[8_000] = nudged[8_000].nextUp
        XCTAssertNotEqual(SamplingSeed.window(audio), SamplingSeed.window(nudged))
        let w = SamplingSeed.window(audio)
        XCTAssertNotEqual(SamplingSeed.attempt(window: w, temperature: 0.2),
                          SamplingSeed.attempt(window: w, temperature: 0.4))
    }
}
