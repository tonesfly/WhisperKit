//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 TonesFly. Fork addition (DET-2).

import Foundation

/// Stable seeds for temperature-fallback sampling: the same audio window at
/// the same temperature draws the same tokens on every run, so a long-form
/// transcript no longer depends on the process's random state. Compute
/// numerics (ANE/GPU) can still vary the logits; that is out of scope.
public enum SamplingSeed {

    /// FNV-1a 64 over the window's sample bit patterns (content, not position).
    public static func window(_ samples: [Float]) -> UInt64 {
        var hash: UInt64 = 0xCBF2_9CE4_8422_2325
        for sample in samples {
            var bits = sample.bitPattern
            for _ in 0..<4 {
                hash = (hash ^ UInt64(bits & 0xFF)) &* 0x0000_0100_0000_01B3
                bits >>= 8
            }
        }
        return hash
    }

    /// One fallback attempt's seed: the window hash mixed with the temperature.
    public static func attempt(window: UInt64, temperature: Float) -> UInt64 {
        Generator.mix(window ^ (UInt64(temperature.bitPattern) &* 0x9E37_79B9_7F4A_7C15))
    }

    /// SplitMix64: small, fast, full-period, uniform 64-bit output.
    public struct Generator: RandomNumberGenerator, Sendable {
        private var state: UInt64
        public init(seed: UInt64) { state = seed }

        public mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            return Self.mix(state)
        }

        static func mix(_ value: UInt64) -> UInt64 {
            var z = value
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }
}
