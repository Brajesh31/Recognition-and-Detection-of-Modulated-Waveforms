# Modulation Recognition and Detection System

A Python project for generating synthetic complex baseband samples and classifying five digital modulation families: **QPSK, 8PSK, 16APSK, 32APSK, and 64APSK**. The classifier uses statistical signal features and a scikit-learn Random Forest.

> **Research limitation:** the included generator creates ideal, noiseless symbol sequences. Results from this synthetic dataset are a software demonstration, not evidence of reliable performance on real RF captures or DVB-S2X signals. Real-world use requires channel impairments, pulse shaping, timing/carrier offsets, realistic constellation parameters, and evaluation on independent measured data.

## Contents

- [Features](#features)
- [Supported modulation types](#supported-modulation-types)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Generate sample files](#generate-sample-files)
- [Train and evaluate](#train-and-evaluate)
- [Predict a signal file](#predict-a-signal-file)
- [Signal file format](#signal-file-format)
- [How the pipeline works](#how-the-pipeline-works)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Security and model files](#security-and-model-files)
- [Known limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

## Features

- Generate deterministic or random complex symbol sequences for five modulation classes.
- Export samples as portable two-column CSV text files containing real and imaginary values.
- Extract a fixed 24-value feature vector from each complex signal.
- Build balanced synthetic datasets and train a Random Forest classifier.
- Evaluate with a stratified holdout split, accuracy, per-class precision/recall/F1, and a confusion matrix.
- Save and load a trained model with Joblib.
- Classify a user-supplied signal file and display class probabilities.
- Run the complete workflow from a command-line interface.

## Supported modulation types

| Label | Family | Constellation order |
|---|---|---:|
| QPSK | Quadrature phase-shift keying | 4 |
| 8PSK | 8-ary phase-shift keying | 8 |
| 16APSK | 16-ary amplitude and phase-shift keying | 16 |
| 32APSK | 32-ary amplitude and phase-shift keying | 32 |
| 64APSK | 64-ary amplitude and phase-shift keying | 64 |

The APSK generator uses simplified concentric-ring constellations. These are useful for controlled software experiments but are not claimed to reproduce a particular DVB-S2/S2X standard's exact ring radii, phase offsets, or symbol probabilities.

## Project structure

```text
.
├── modulation_utils.py             # Shared constellation, feature, and input utilities
├── modulation_signal_generator.py  # Sample-file generation CLI
├── modulation_recognition.py       # Dataset, training, evaluation, prediction CLI
├── requirements.txt                # Runtime dependencies
├── pytest.ini                      # Pytest configuration
├── tests/
│   └── test_modulation.py           # Automated tests
├── docs/                            # Design, methodology, testing, and operations notes
└── .github/workflows/               # CI, security analysis, and documentation lint
```

Generated files such as `sample_inputs/` and `modulation_model.pkl` are local outputs and are not required to be committed.

## Requirements

- Python 3.10 or newer
- pip
- NumPy, SciPy, scikit-learn, and Joblib

## Installation

```bash
git clone https://github.com/Brajesh31/Recognition-and-Detection-of-Modulated-Waveforms.git
cd Recognition-and-Detection-of-Modulated-Waveforms
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Generate sample files

Generate 1,000 complex samples for each modulation type:

```bash
python modulation_signal_generator.py
```

Choose an output directory, sample count, and random seed:

```bash
python modulation_signal_generator.py --output-dir sample_inputs --samples 2000 --seed 42
```

The command creates one file per class, for example `sample_inputs/QPSK_sample.txt`.

## Train and evaluate

Train on 200 generated signal records per modulation class, evaluate on a stratified holdout set, and save the final model:

```bash
python modulation_recognition.py train
```

Customize the dataset and output model path:

```bash
python modulation_recognition.py train --samples-per-class 500 --signal-length 1000 --seed 42 --model models/modulation_model.pkl
```

The output includes holdout accuracy, a classification report, and a confusion matrix. Accuracy depends on the synthetic generator and chosen sample size. Do not treat it as a benchmark for real-world radio signals.

## Predict a signal file

After training, classify a file containing real and imaginary sample columns:

```bash
python modulation_recognition.py predict sample_inputs/16APSK_sample.txt
```

Use a model saved to a custom location:

```bash
python modulation_recognition.py predict sample_inputs/16APSK_sample.txt --model models/modulation_model.pkl
```

The command prints the predicted class and the model's per-class probability estimates. These probabilities are model scores, not calibrated physical certainty.

## Signal file format

Files use comma-delimited numeric columns with no header. Each row represents one complex sample, with the real value first and the imaginary value second. Files must contain exactly two finite numeric columns and at least two rows. The loader rejects malformed files rather than silently reshaping unexpected data.

## How the pipeline works

1. **Constellation generation:** draw symbols from the selected ideal PSK or simplified APSK constellation.
2. **Feature extraction:** calculate a fixed-length vector from real, imaginary, magnitude, phase, and adjacent-sample difference statistics.
3. **Dataset creation:** create the same number of signal records for each class.
4. **Evaluation:** split the generated records with stratification so each class is represented in both sets.
5. **Training:** fit a Random Forest on the complete dataset after holdout evaluation.
6. **Persistence and inference:** serialize the estimator locally and reuse it to classify compatible signal files.

See [the methodology notes](docs/METHODOLOGY.md) and [architecture](docs/ARCHITECTURE.md) for details and assumptions.

## Testing

Install dependencies, then run:

```bash
python -m pip install pytest
python -m pytest
```

The test suite covers all modulation generators, constellation cardinality, deterministic dataset generation, file round-tripping, feature shape, model persistence, prediction, and invalid input handling.

## Reproducibility

Pass `--seed` to the sample generator and training command to make random generation and model fitting repeatable within the same dependency environment. Exact serialized model bytes may still vary across library versions. Record Python and dependency versions when comparing experiments.

## Security and model files

- Never commit credentials, private captures, or sensitive RF data.
- Treat `.pkl` and Joblib model files as executable serialized content. Load only model files from trusted sources.
- Keep generated datasets and model artifacts out of Git unless there is a clear, documented reason to publish them.
- Do not use the classifier as a safety-critical spectrum-monitoring or interference-detection system without independent validation.

## Known limitations

- The default data is ideal and noiseless. There is no AWGN, fading, phase noise, frequency offset, timing offset, pulse shaping, or receiver synchronization model.
- The APSK constellations are simplified and are not standards-compliant reference implementations.
- Statistical features discard much of the temporal and spectral structure used by production automatic modulation recognition systems.
- The training and holdout sets come from the same synthetic generator. This measures internal consistency, not generalization to unseen hardware, channels, or environments.
- No calibrated confidence, model versioning, or robust out-of-distribution detection is provided.

See [known limitations and roadmap](docs/LIMITATIONS.md).

## Contributing

Bug reports, reproducible test cases, and improvements to the signal model are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

This repository includes an MIT license for original project material where the maintainer has the right to grant that license. See [LICENSE](LICENSE). Third-party dependencies retain their own licenses.

## Maintainer

Brajesh Kumar

Last updated: 2026-11-14 UTC
