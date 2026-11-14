# Architecture

## Components

- `modulation_utils.py` owns modulation labels, symbol generation, feature extraction, and signal-file parsing. It is the shared source of truth for the supported classes.
- `modulation_signal_generator.py` exposes a command-line entry point for writing one sample file per modulation type.
- `modulation_recognition.py` builds balanced datasets, trains and evaluates the estimator, persists trusted local model artifacts, and performs inference.
- `tests/test_modulation.py` checks the public behavior at the utility, file, and model boundaries.

## Data flow

Constellation generator -> complex samples -> feature extractor -> feature matrix -> Random Forest.
CSV signal file -> strict parser -> feature extractor -> saved model -> prediction.

## Design decisions

- The five supported labels are defined once in `modulation_utils.py` to prevent generator and classifier drift.
- Signal files contain explicit real and imaginary columns, avoiding Python-specific complex serialization.
- Feature extraction returns a fixed 24-value vector and normalizes undefined moments to finite values.
- Dataset generation accepts a NumPy random generator seed for reproducible experiments.
- The classifier is persisted locally. Deserializing Joblib files is only safe when their origin is trusted.

## Extension points

Add new constellations in the shared generator, then add tests for constellation cardinality and class coverage. If feature dimensions change, retrain all saved models and update compatibility checks. Channel simulation should be a separate, testable stage rather than hidden inside the classifier.
