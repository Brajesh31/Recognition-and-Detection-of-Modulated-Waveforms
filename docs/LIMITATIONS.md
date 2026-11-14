# Limitations and roadmap

## Current limitations

- The constellation generator is intentionally simplified and does not claim DVB-S2/S2X compliance.
- No noise, fading, carrier-frequency offset, phase noise, timing error, pulse shaping, or matched filtering is simulated.
- The model sees aggregate statistics rather than a learned time-frequency representation.
- Holdout evaluation samples from the same synthetic generator used for training.
- Model probability estimates are not calibrated, and unknown modulation classes are not detected.
- The project has no graphical interface or streaming/real-time receiver.

## Recommended next steps

1. Add configurable AWGN and SNR sweeps with unit-tested noise-power calculations.
2. Add standards-based constellation parameters with references to the applicable specifications.
3. Add oversampling, pulse shaping, timing/carrier offsets, and receiver-side synchronization.
4. Compare statistical features with cumulants, spectral features, and time-frequency approaches.
5. Evaluate using scenario-separated synthetic captures and a documented real-world dataset where licensing permits.
6. Add calibration, unknown-class rejection, and model metadata/version compatibility.
7. Add plots only after the evaluation protocol and signal assumptions are explicit.

Do not present a single synthetic holdout score as evidence of field performance.
