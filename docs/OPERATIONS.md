# Operations and maintenance

## Local workflow

1. Create and activate a virtual environment.
2. Install `requirements.txt`.
3. Run `python -m pytest` before committing.
4. Generate sample files or train a model only when needed. These artifacts are local outputs by default.
5. Record the random seed and dependency versions for experiments intended to be compared.

## Dependency maintenance

Update dependencies deliberately and run the complete test suite after upgrades. Scikit-learn and Joblib model serialization is version-sensitive, so retrain models after major dependency changes rather than assuming old model files remain compatible.

## CI and security

The Python workflow validates dependencies, tests, and CLI behavior. CodeQL scans Python code. Markdown lint checks documentation. Dependabot proposes weekly dependency and Actions updates. Review automated updates and their test results before merging.

## Generated artifacts

Do not commit local virtual environments, caches, generated sample data, or serialized models by default. Treat externally supplied model files as untrusted executable input.
