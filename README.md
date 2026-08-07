# Towards-Verifiable-AI-with-Proofs-of-Inference

Research code exploring **verifiable machine-learning inference**: how to attack, separate, and cryptographically support proofs that a claimed model computation actually occurred.

## Motivation

Proofs of inference aim to let a verifier check that a model produced a given output for a given input — without naively re-running the full model. This repository studies complementary pieces of that problem:

1. **Attacks** that try to reconstruct or forge inputs/activations (stress-testing what a proof must bind).
2. **Model separation** experiments that quantify how distinguishable independently trained models are (relevant to “which model was used?”).
3. **Performance / ZK primitives** (CUDA field arithmetic, Keccak, Merkle commitments) used in proof-system style pipelines.

## Repository Structure

| Folder | Description |
|--------|-------------|
| `Gradient Descent Attack on Basic Models/` | GD-based input reconstruction against small ANNs (Iris/tabular) |
| `Gradient Descent Attack on LLMs/` | Gradient input-reconstruction attack on Llama-2 with analyzers/visualizers |
| `Inverse Transform Attack/` | Inverse-transform style input cracking on basic models |
| `Swap Attack on LLMs/` | Activation/weight swap attacks probing integrity assumptions |
| `Model Separation on Basic Models/` | Train/test/visualize separation for simple networks |
| `Model Separation in Classifiers/` (+ `v2`, `v3`) | Image-classifier separation studies across iterations |
| `Model Separation in LLMs/` | Separation analysis for LLM instances (weights, results, plots) |
| `Performance/` | CUDA/C++ kernels and headers (BLS12-381, Goldilocks, Keccak, Merkle) for proof-related benchmarks |

## Typical Experiment Pattern

Many folders follow a similar layout:

- `model_structure.py` / trainer notebook — define and train models
- `tester.py` / `*_attack.py` — run evaluation or attack
- `*_analyzer.py` / `*_result_checker.py` — aggregate large CSV outputs
- `visualizer*.py` — publication-style plots

## Notes

- LLM attack scripts expect substantial compute/GPU memory and large intermediate CSVs.
- `Performance/` mixes project-specific CUDA sources with third-party/Apache-licensed field-arithmetic headers (see file copyright banners).
