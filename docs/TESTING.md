# Testing

Run the complete suite with:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
```

The tests verify:

- All declared modulation generators return finite complex arrays.
- APSK constellation cardinality matches 16, 32, and 64 points.
- Feature extraction returns 24 finite values.
- Dataset generation is balanced and repeatable for a fixed seed.
- The generator writes files that the strict CSV loader can read.
- A Random Forest can be trained, saved, loaded, and used for prediction.
- Invalid modulation labels, malformed sample lengths, and missing files fail with useful errors.

The GitHub Actions workflow runs the same test suite on pushes and pull requests to `main`, and performs a small CLI smoke test.
