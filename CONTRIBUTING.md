# Contributing

Thanks for considering a contribution.

## Before changing code

- Open an issue for substantial changes so the scope and assumptions can be discussed.
- Keep modulation labels and shared signal behavior centralized in `modulation_utils.py`.
- Do not claim standards compliance or real-world accuracy without reproducible evidence.

## Development setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest
```

## Pull request expectations

- Explain the problem and the proposed change.
- Add or update tests for behavior changes.
- Include the random seed and exact command for reproducible experiments.
- Document any new signal assumptions, dependencies, or model compatibility changes.
- Keep generated datasets, model binaries, credentials, and unrelated IDE files out of commits.
