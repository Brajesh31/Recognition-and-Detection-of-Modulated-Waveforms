import numpy as np
import os

# Constants
TIME_FRAMES = 1000  # Number of time samples per signal
OUTPUT_DIR = 'sample_inputs'  # Directory to save generated modulation files
MOD_TYPES = ['QPSK', '8PSK', '16APSK', '32APSK', '64APSK']  # List of supported modulation types

# Functions to generate modulation signals

def generate_qpsk(n_samples):
    """Generate QPSK (Quadrature Phase Shift Keying) modulated signal."""
    symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)  # Normalized QPSK symbols
    return np.random.choice(symbols, n_samples)

def generate_psk(n_samples, M):
    """Generate M-PSK (Phase Shift Keying) modulated signal."""
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    symbols = np.exp(1j * angles)  # Generate unit circle symbols
    return np.random.choice(symbols, n_samples)

def generate_apsk(n_samples, M):
    """Generate M-APSK (Amplitude Phase Shift Keying) modulated signal."""
    if M == 16:
        radii = [1, 2]  # Two concentric rings
    elif M == 32:
        radii = [1, 2, 3]  # Three concentric rings
    elif M == 64:
        radii = [1, 2, 3, 4]  # Four concentric rings
    else:
        raise ValueError("Unsupported M for APSK. Use 16, 32, or 64.")

    symbols = []
    for r in radii:
        angles = np.linspace(0, 2 * np.pi, M // len(radii), endpoint=False)
        symbols.extend(r * np.exp(1j * angles))  # Create APSK symbols

    return np.random.choice(symbols, n_samples)

def generate_sample_files():
    """Generate modulation samples and save them as text files."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)  # Create output directory if not exists

    for mod_type in MOD_TYPES:
        # Generate signal based on modulation type
        if mod_type == 'QPSK':
            signal = generate_qpsk(TIME_FRAMES)
        elif mod_type == '8PSK':
            signal = generate_psk(TIME_FRAMES, 8)
        elif mod_type == '16APSK':
            signal = generate_apsk(TIME_FRAMES, 16)
        elif mod_type == '32APSK':
            signal = generate_apsk(TIME_FRAMES, 32)
        elif mod_type == '64APSK':
            signal = generate_apsk(TIME_FRAMES, 64)
        else:
            raise ValueError("Unsupported modulation type.")

        # Save the generated signal to a text file
        file_path = os.path.join(OUTPUT_DIR, f"{mod_type}_sample.txt")
        np.savetxt(file_path, signal.view(float).reshape(-1, 2), delimiter=",")
        print(f"Sample file generated for {mod_type}: {file_path}")

if __name__ == "__main__":
    generate_sample_files()  # Execute the function when running the script
