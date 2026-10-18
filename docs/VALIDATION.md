# Tests

```bash
python scripts/check_release.py
python scripts/reproduce.py --validate
python -m unittest discover -s tests -v
```

The suite covers response statistics, BN folding, Gaussian accounting,
integer-clipping bounds, share-cache consistency, paired-release noise,
MP-SPDZ serialization, configuration digests, and a synthetic training round.
`check_release.py` verifies source hashes and repository contents.

Validation on 2026-10-18 used Python 3.12, PyTorch 2.8, and torchvision 0.23
on macOS/Apple Silicon: 47 tests passed and 3 CUDA tests were skipped.
All 278 configuration checks passed. Isolated clones passed CryptGPU import
and ONNX conversion. Linux MP-SPDZ tests executed three parties on signed and
decimal vectors; maximum error against the quantized reference was `1.18e-7`.

On Linux with an RTX 5090 and NVIDIA 580.178.04, all three CUDA tests executed
and passed without skips. The integer CryptGPU recipe completed one- and
two-round runs with three local MPC processes. Checks covered integer clipping
inside MPC, secret-mask aggregation, clipped-share cache reuse, three malicious
submissions, and use of the filtered reference in the following round. These
short runs verify execution and artifact consistency; full 200-round results
were not reproduced in this validation.

CUDA enables the three-party integer-clipping and cache tests. MP-SPDZ execution
requires the external runtime described in [backend setup](BACKENDS.md).
