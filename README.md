# PrivFilter

Implementation of **PrivFilter: Backdoor Filtering for Privacy-Preserving
Federated Learning**. The repository includes training, response-based filtering,
CryptGPU computation, baseline comparisons, and privacy evaluations.

## Installation

Use Python 3.12. CryptGPU execution requires Linux, NVIDIA CUDA, PyTorch 2.8,
and torchvision 0.23. CPU execution supports utility profiles and unit tests.

```bash
git submodule update --init external/CryptGPU
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU execution, install the matching CUDA builds of PyTorch and torchvision,
then prepare the pinned CryptGPU runtime:

```bash
python tools/prepare_cryptgpu_runtime.py --verify-conversion
```

See [backend setup](docs/BACKENDS.md) for data paths, party seeds, and MP-SPDZ.

## Experiments

List the main configurations and run one utility experiment:

```bash
python scripts/reproduce.py --suite main --list
python scripts/reproduce.py --run mmtable_sim_cifar10_seed1 \
  --execute --local-test-seeds --output-dir outputs/cifar10-seed1
```

The `main` suite uses local numerical-reference execution for utility evaluation.
The `mpc-integer` suite uses CryptGPU with integer clipping and share caching. See the
[experiment guide](reproduction/README.md) for profiles, data locations,
seed provisioning, and privacy evaluations. Generated files are stored in `outputs/`.

## Tests

```bash
python scripts/check_release.py
python scripts/reproduce.py --validate
python -m unittest discover -s tests -v
```

[Test coverage](docs/VALIDATION.md) lists the available checks and hardware requirements.
Source and configuration provenance are recorded in `SOURCE_MANIFEST.json` and
`reproduction/manifest.json`. See [attribution and licensing](THIRD_PARTY.md)
for dependency details.
