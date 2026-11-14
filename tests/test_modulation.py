from pathlib import Path
import numpy as np
import pytest
from modulation_recognition import generate_dataset, load_model, predict_file, save_model, train_model
from modulation_signal_generator import generate_sample_files
from modulation_utils import MOD_TYPES, extract_features, generate_signal, load_signal_file

def test_all_supported_generators_return_expected_length_and_finite_values():
    rng = np.random.default_rng(7)
    for modulation in MOD_TYPES:
        signal = generate_signal(modulation, 128, rng)
        assert signal.shape == (128,)
        assert np.iscomplexobj(signal)
        assert np.all(np.isfinite(signal))
        assert extract_features(signal).shape == (24,)

def test_apsk_constellations_have_exact_order():
    from modulation_utils import generate_apsk
    for order in (16, 32, 64):
        signal = generate_apsk(5000, order, np.random.default_rng(order))
        assert len(np.unique(signal)) == order

def test_dataset_is_balanced_and_reproducible():
    x1, y1 = generate_dataset(4, 80, seed=10)
    x2, y2 = generate_dataset(4, 80, seed=10)
    assert x1.shape == (20, 24)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert set(y1) == set(MOD_TYPES)

def test_generator_writes_loadable_two_column_files(tmp_path: Path):
    files = generate_sample_files(tmp_path, n_samples=64, seed=2)
    assert len(files) == len(MOD_TYPES)
    for file_path in files:
        assert load_signal_file(file_path).shape == (64,)

def test_training_save_load_and_prediction(tmp_path: Path):
    x, y = generate_dataset(12, 120, seed=11)
    model_path = save_model(train_model(x, y), tmp_path / "model.pkl")
    signal_path = generate_sample_files(tmp_path / "signals", 120, seed=12)[0]
    loaded = load_model(model_path)
    assert loaded.n_features_in_ == 24
    prediction, probabilities = predict_file(signal_path, model_path)
    assert prediction in MOD_TYPES
    assert set(probabilities) == set(MOD_TYPES)
    assert sum(probabilities.values()) == pytest.approx(1.0)

def test_invalid_signal_inputs_are_rejected():
    with pytest.raises(ValueError):
        extract_features(np.array([1 + 1j]))
    with pytest.raises(ValueError):
        generate_signal("UNKNOWN", 10)
    with pytest.raises(ValueError):
        generate_signal("QPSK", 0)

def test_missing_model_and_signal_files_raise_clear_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.pkl")
    with pytest.raises(FileNotFoundError):
        load_signal_file(tmp_path / "missing.csv")
