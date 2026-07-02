# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-07-01

### Added
- Type hints to arrays using `jaxtyping` in `diff_fret.kernels`.
- Missing tests to reach 100% test coverage.

### Changed
- Required explicit PRNG keys in stochastic accessible volume (AV) simulations for deterministic safety.
- Updated documentation URLs and tutorials.

### Fixed
- Fixed mypy and numpy version conflicts in pre-commit and GitHub actions CI.

## [0.1.1] - 2026-06-07

### Security
- Removed compromised `polyfill.io` CDN script from MkDocs configuration to resolve supply-chain vulnerability.
