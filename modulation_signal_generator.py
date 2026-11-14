"""Generate reproducible sample files for supported digital modulations."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from modulation_utils import MOD_TYPES, TIME_FRAMES, generate_signal

def generate_sample_files(output_dir: str | Path = "sample_inputs", n_samples: int = TIME_FRAMES,
                          seed: int | None = None) -> list[Path]:
    """Generate one two-column CSV signal file per modulation type."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    generated = []
    for modulation in MOD_TYPES:
        signal = generate_signal(modulation, n_samples, rng)
        file_path = output_path / f"{modulation}_sample.txt"
        np.savetxt(file_path, np.column_stack((signal.real, signal.imag)), delimiter=",", fmt="%.10g")
        generated.append(file_path)
        print(f"Generated {modulation}: {file_path} ({n_samples} samples)")
    return generated

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="sample_inputs")
    parser.add_argument("--samples", type=int, default=TIME_FRAMES)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    generate_sample_files(args.output_dir, args.samples, args.seed)

if __name__ == "__main__":
    main()
