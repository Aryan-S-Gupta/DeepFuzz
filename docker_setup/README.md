# Docker setup for doc-driven DL API fuzzing + coverage

This bundle gives you a reproducible CPU-first Docker workflow for:
- building an instrumented source checkout
- running your Stage 4 coverage runner
- exporting per-API and campaign coverage

Included files:
- Dockerfile.pytorch
- Dockerfile.tensorflow
- Dockerfile.jax
- run_pytorch.sh
- run_tensorflow.sh
- run_jax.sh

These are meant to be used one library at a time.
