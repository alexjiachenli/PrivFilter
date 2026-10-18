# Experiments

`manifest.json` defines 278 runs using the profiles in `configs/`. Each run
records its source configuration, seed, backend, and parameter digest.

## Running experiments

```bash
python scripts/reproduce.py --list
python scripts/reproduce.py --run mmtable_sim_cifar10_seed1
python scripts/reproduce.py --run mmtable_sim_cifar10_seed1 \
  --execute --local-test-seeds --output-dir outputs/cifar10-seed1
```

Without `--execute`, the runner prints the plan. Use `--materialize-only` to
write resolved configurations. `--suite main` selects the three datasets and
seeds 1–3. Output directories must be new.

| Option | Purpose |
| --- | --- |
| `--rounds N` | Override training rounds for a short run |
| `--data-root PATH` | Select the dataset cache |
| `--output-dir PATH` | Select the output directory |
| `--seed-root PATH` | Select the private party-seed directory |
| `--validate` | Check configurations before training |

Each run saves its configuration, metrics, participation schedule, environment,
code revision, and execution receipt. Noise seeds and generated outputs are
excluded from Git.
Successful runs record `return_code: 0` in `run_receipt.json` and write
`summary.json` and `final_metrics.csv`.

## Profiles

| Suite | Evaluation |
| --- | --- |
| `main` | PrivFilter utility on CIFAR-10, CIFAR-100, and GTSRB |
| `baselines` | RoFL, ELSA, ShieldFL, CrowdGuard, and FLTrust comparisons |
| `references` | No defense and plaintext FilterFL |
| `rule-control` | FilterFL's original decision rule in the PrivFilter pipeline |
| `privacy-control` | Client-noise DP-FilterFL control |
| `supplementary-baselines` | Krum, Trimmed-Mean, FLAME, FedREDefense, and Lockdown |
| `heterogeneity-references` | Reference runs across Dirichlet settings |
| `epsilon`, `grid`, `dpcbf` | Noise, alpha/q, and BN-statistics ablations |
| `dba`, `adaptive`, `canary` | Distributed attacks, adaptive attacks, and canary responses |
| `cost` | CryptGPU timing and scaling |
| `mpc-integer` | Integer MPC clipping with share caching |

The `main` suite uses local numerical-reference execution with floating-point
clipping. The `cost` suite uses precomputed bounded inputs. The `mpc-integer` suite executes integer
clipping and uses a secret acceptance mask. Their backend and clipping settings
are recorded in the manifest.

The Gaussian accountant evaluates the fixed-state, fixed-partition reference
mechanism. Client-noise controls with uniform sampling retain that schedule and
report their budget label as invalid.

## Party seeds

Provision a seed for each party using its own account:

```bash
python scripts/provision_noise_seed.py \
  --run privfilter_mpc_integer_cifar10_seed1 --party 0
```

Repeat for parties 1 and 2. Pass the same seed root to provisioning (`--root`)
and execution (`--seed-root`). Existing seeds are preserved. `--local-test-seeds`
is available for simulator diagnostics. See [backend setup](../docs/BACKENDS.md)
for the local three-process runtime.

```bash
python scripts/reproduce.py --run privfilter_mpc_integer_cifar10_seed1 \
  --execute --output-dir outputs/mpc-cifar10-seed1
```

## Privacy evaluations

```bash
python reproduction/privacy/collect_updates.py \
  --params configs/privfilter_cifar10.json --rounds 12 --seed 1 --alpha 1 \
  --keep-local-models 4,8,12 --out outputs/privacy/updates

python reproduction/privacy/attack_differential.py \
  --updates outputs/privacy/updates --out outputs/privacy/differential.json

python reproduction/privacy/attack_membership_audit.py \
  --updates outputs/privacy/updates --arm partition \
  --out outputs/privacy/membership_audit.json

python reproduction/privacy/attack_membership_inference.py \
  --updates outputs/privacy/updates --rounds 4,8,12 \
  --out outputs/privacy/mia.json

python reproduction/privacy/attack_linkage_ladder.py \
  --raw-glob 'outputs/full-run/*/summary.json' \
  --out outputs/privacy/linkage.json
```

Use `--arm partition_collude` for the collusion arm. Linkage evaluation reads
transcript fields from diagnostic run summaries. Collected client models and
sample memberships are private audit inputs.

Membership inference compares complete model views: local models before
clipping, a clipped aggregate, and its quantized noisy release. Nonfinite losses
trigger BN-statistics re-estimation on the selected nonmembers before rescoring.
