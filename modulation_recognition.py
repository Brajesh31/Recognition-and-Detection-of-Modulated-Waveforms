import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import skew, kurtosis
import seaborn as sns
import joblib
import os

# Constants
MOD_TYPES = ['QPSK', '8PSK', '16APSK', '32APSK']  # List of supported modulations
MODEL_FILE = 'modulation_model.pkl'  # File to save/load trained model
TIME_FRAMES = 1000  # Number of time samples per signal

# Functions to generate modulation signals (same as in the generator script)
def generate_qpsk(n_samples):
    """Generate QPSK modulated signal."""
    symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    return np.random.choice(symbols, n_samples)

def generate_psk(n_samples, M):
    """Generate M-PSK modulated signal."""
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    symbols = np.exp(1j * angles)
    return np.random.choice(symbols, n_samples)

def generate_apsk(n_samples, M):
    """Generate M-APSK modulated signal."""
    if M == 16:
        radii = [1, 2]
    elif M == 32:
        radii = [1, 2, 3]
    else:
        raise ValueError("Unsupported M for APSK. Use 16 or 32.")

    symbols = []
    for r in radii:
        angles = np.linspace(0, 2 * np.pi, M // len(radii), endpoint=False)
        symbols.extend(r * np.exp(1j * angles))

    return np.random.choice(symbols, n_samples)

# Feature extraction
def extract_features(data):
    """Extract statistical features from the real and imaginary parts of a signal."""
    features = []
    real_part = data.real
    imag_part = data.imag
    magnitude = np.abs(data)
    phase = np.angle(data)

    # Statistical features
    features += [
        np.mean(real_part), np.std(real_part), np.mean(imag_part), np.std(imag_part),
        np.mean(magnitude), np.std(magnitude), np.mean(phase), np.std(phase),
        skew(real_part), kurtosis(real_part), skew(imag_part), kurtosis(imag_part),
        np.min(real_part), np.max(real_part), np.min(imag_part), np.max(imag_part),
        np.min(magnitude), np.max(magnitude), np.min(phase), np.max(phase)
    ]
    
    return features

# Generate dataset
def generate_dataset(n_samples_per_mod):
    """Create a dataset of modulated signals and their corresponding labels."""
    data, labels = [], []
    for mod_type in MOD_TYPES:
        if mod_type == 'QPSK':
            signals = [generate_qpsk(TIME_FRAMES) for _ in range(n_samples_per_mod)]
        elif mod_type == '8PSK':
            signals = [generate_psk(TIME_FRAMES, 8) for _ in range(n_samples_per_mod)]
        elif mod_type == '16APSK':
            signals = [generate_apsk(TIME_FRAMES, 16) for _ in range(n_samples_per_mod)]
        elif mod_type == '32APSK':
            signals = [generate_apsk(TIME_FRAMES, 32) for _ in range(n_samples_per_mod)]
        else:
            raise ValueError("Unsupported modulation type.")

        for signal in signals:
            data.append(extract_features(signal))
            labels.append(mod_type)
    
    return np.array(data), np.array(labels)

# Model training
def train_model(X_train, y_train):
    """Train a RandomForest model for modulation classification."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Save and load the model
def save_model(model):
    """Save the trained model to a file."""
    joblib.dump(model, MODEL_FILE)

def load_model():
    """Load the trained model if it exists."""
    return joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None

# Main interactive function
def interactive_program():
    """Command-line interface for training and testing the model."""
    print("Welcome to Modulation Detection System")

    while True:
        print("\nOptions:")
        print("1. Generate dataset and train model")
        print("2. Test model with an input file")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            n_samples = int(input("Enter number of samples per modulation type: "))
            data, labels = generate_dataset(n_samples)
            model = train_model(data, labels)
            save_model(model)
            print("Training complete. Model saved.")

        elif choice == '2':
            print("Feature extraction and classification for input signal will be implemented here.")

        elif choice == '3':
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    interactive_program()
