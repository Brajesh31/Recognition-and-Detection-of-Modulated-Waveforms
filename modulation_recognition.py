"""Train and use a Random Forest classifier for digital modulation recognition."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from modulation_utils import MOD_TYPES, TIME_FRAMES, extract_features, generate_signal, load_signal_file

MODEL_FILE = Path("modulation_model.pkl")

def generate_dataset(n_samples_per_mod: int, samples_per_signal: int = TIME_FRAMES,
                     seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build a balanced feature dataset for all supported modulation classes."""
    if n_samples_per_mod < 2 or samples_per_signal < 2:
        raise ValueError("Sample counts must both be at least 2.")
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for modulation in MOD_TYPES:
        for _ in range(n_samples_per_mod):
            features.append(extract_features(generate_signal(modulation, samples_per_signal, rng)))
            labels.append(modulation)
    return np.vstack(features), np.asarray(labels)

def train_model(features: np.ndarray, labels: np.ndarray, seed: int = 42) -> RandomForestClassifier:
    """Fit a deterministic Random Forest classifier."""
    if features.ndim != 2 or features.shape[0] != len(labels):
        raise ValueError("features must be a 2D array with one label per row.")
    if len(np.unique(labels)) < 2:
        raise ValueError("Training requires at least two modulation classes.")
    model = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=1)
    model.fit(features, labels)
    return model

def save_model(model: RandomForestClassifier, path: str | Path = MODEL_FILE) -> Path:
    """Persist a trained model."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path

def load_model(path: str | Path = MODEL_FILE) -> RandomForestClassifier:
    """Load a previously saved model from a trusted local path."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}. Train a model first.")
    model = joblib.load(model_path)
    if not hasattr(model, "predict") or not hasattr(model, "n_features_in_"):
        raise ValueError("The selected file does not contain a compatible classifier.")
    return model

def evaluate_model(model: RandomForestClassifier, features: np.ndarray, labels: np.ndarray) -> dict:
    """Return test metrics and a classification report."""
    predictions = model.predict(features)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=MOD_TYPES),
        "classification_report": classification_report(labels, predictions, labels=MOD_TYPES, zero_division=0),
    }

def train_and_evaluate(n_samples_per_mod: int = 200, samples_per_signal: int = TIME_FRAMES,
                       seed: int = 42) -> tuple[RandomForestClassifier, dict]:
    """Evaluate on a stratified holdout, then fit the final model on all records."""
    features, labels = generate_dataset(n_samples_per_mod, samples_per_signal, seed)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=seed, stratify=labels
    )
    evaluation_model = train_model(x_train, y_train, seed)
    metrics = evaluate_model(evaluation_model, x_test, y_test)
    return train_model(features, labels, seed), metrics

def predict_file(signal_path: str | Path, model_path: str | Path = MODEL_FILE) -> tuple[str, dict[str, float]]:
    """Predict the modulation class for one real/imaginary CSV signal file."""
    model = load_model(model_path)
    features = extract_features(load_signal_file(signal_path)).reshape(1, -1)
    if features.shape[1] != model.n_features_in_:
        raise ValueError("Signal feature count does not match the saved model. Retrain the model.")
    prediction = str(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else []
    confidence = {str(label): float(probability) for label, probability in zip(model.classes_, probabilities)}
    return prediction, confidence

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Generate synthetic data, evaluate, and save a model.")
    train_parser.add_argument("--samples-per-class", type=int, default=200)
    train_parser.add_argument("--signal-length", type=int, default=TIME_FRAMES)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--model", default=str(MODEL_FILE))
    predict_parser = subparsers.add_parser("predict", help="Classify a CSV file with real,imaginary columns.")
    predict_parser.add_argument("signal_file")
    predict_parser.add_argument("--model", default=str(MODEL_FILE))
    args = parser.parse_args()
    if args.command == "train":
        model, metrics = train_and_evaluate(args.samples_per_class, args.signal_length, args.seed)
        saved_path = save_model(model, args.model)
        print(f"Holdout accuracy: {metrics['accuracy']:.2%}")
        print(metrics["classification_report"])
        print("Confusion matrix (rows=true, columns=predicted, order=MOD_TYPES):")
        print(metrics["confusion_matrix"])
        print(f"Model saved to: {saved_path}")
    else:
        prediction, confidence = predict_file(args.signal_file, args.model)
        print(f"Predicted modulation: {prediction}")
        if confidence:
            print("Class probabilities:")
            for label, probability in sorted(confidence.items(), key=lambda item: item[1], reverse=True):
                print(f"  {label}: {probability:.2%}")

if __name__ == "__main__":
    main()
