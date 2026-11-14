# Security Policy

## Supported versions

Security fixes are applied to the current default branch. There are no separately maintained release branches at this time.

## Reporting a vulnerability

Please do not publish exploit details in a public issue. Use GitHub's private vulnerability reporting feature for this repository when available. Include the affected file or version, impact, and a minimal reproduction. Do not attach private signal captures or credentials.

## Security considerations

Serialized Joblib models must be treated as trusted executable content. Never load a model file from an untrusted source. Public-facing deployments should impose upload size limits and isolate model inference from sensitive systems.
