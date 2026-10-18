# Backends and data

## Data

CIFAR-10, CIFAR-100, and GTSRB download through torchvision into `data/`.
Use `--data-root PATH` to select another cache. The fallback GTSRB folder format
is defined in `datasets/GTSRB.py`.

## CryptGPU

Use Linux with NVIDIA CUDA and matching PyTorch 2.8/torchvision 0.23 builds.

```bash
git submodule update --init external/CryptGPU
python tools/prepare_cryptgpu_runtime.py --verify-conversion
```

The preparer applies `patches/cryptgpu-pytorch2-runtime.patch` to the pinned
upstream revision and records the resulting hashes in `.cache/cryptgpu-runtime/`.
Import and ONNX conversion checks run on CPU. GPU tests require CUDA.

The launcher runs three local party processes. Each party uses a private noise
seed and share directory. Independent server deployment requires separate party
execution and private storage. Seed provisioning commands are in the
[experiment guide](../reproduction/README.md#party-seeds).

## MP-SPDZ

RoFL/ELSA baseline profiles use MP-SPDZ for final aggregation. Install
[MP-SPDZ 0.4.1](https://mp-spdz.readthedocs.io/en/v0.4.1/readme.html) and build
`ps-rep-ring-party.x`. Keep its compiler dependencies in the Python environment.

```bash
export MP_SPDZ_ROOT=/path/to/MP-SPDZ
# Set this only when the executable is outside the runtime directory.
export MP_SPDZ_PARTY_BINARY=/path/to/ps-rep-ring-party.x
```

The runtime requires `compile.py`, `Programs/`, `Scripts/`, and writable
compilation and party-input directories. The runner compiles with `-R 64`,
uses fixed-point scale `10^4`, and launches three local parties on base port 37000.
It accepts the source-build layout and `bin/Linux-amd64/`.
An explicit `mp_spdz_root` parameter takes precedence over `MP_SPDZ_ROOT`.
Failed aggregation stops the run.
