# PyHealth core baseline on released SUPPORT

This example trains a `SurvivalMDN` model on the preprocessed SUPPORT dataset
distributed with the Survival-MDN repository. It uses a fixed training
configuration, selects weights using validation NLL, restores the best
checkpoint, and exports survival curves for every test row.
Related work: [issue #1297](https://github.com/sunlabuiuc/PyHealth/issues/1297).

## Source and audit

Obtain the released `data/support` archives from the
[Survival-MDN repository](https://github.com/XintianHan/Survival-MDN/tree/71c61f649f3c8784fe13b092b38e0cbb3a184dbd/data/support),
at revision `71c61f649f3c8784fe13b092b38e0cbb3a184dbd`.
Supply a local directory explicitly; the example never downloads data or imports
the reference Python code. Keep benchmark records and generated artifacts out of
Git and public documentation.

The adapter retains the released train/valid/test partitions, feature columns,
and archive row order. Only training batches shuffle. Record IDs such as
`support-split1-test-row000000` identify archive rows; original patient IDs are
unavailable, so identity is tracked within each archive. All archive rows are
retained, including identical numeric rows.

The split-1 audit found:

| Archive | `arr_0` shape | `arr_1` shape | Events | Censored |
| --- | --- | --- | --- | --- |
| train_1.npz | (5370, 27) | (5370, 2) | 3609 | 1761 |
| valid_1.npz | (1776, 27) | (1776, 2) | 1195 | 581 |
| test_1.npz | (1727, 27) | (1727, 2) | 1232 | 495 |

All selected values are finite. `arr_0` is float32; `arr_1` is float64.
All three archives also contain finite numeric float32 `arr_2` with shape
`[N, 27]`, which the reference loader and this adapter ignore. Fingerprints:

```text
train_1.npz 98ec7c2960e6536ca76ca6957bff19a24285440d4f9749c985e0ec7e5f9b0b54
valid_1.npz 01622f42124fc1115d366b058d4acd30e319256061e47426629317900734abab
test_1.npz  94cb108095991302113c439c7e92f40584dc8670570624f96357a256be328409
```

Training `arr_0` has maximum absolute column mean `5.38e-7` and population
standard deviations from `0.999999896` to `1.000000134` (float64 audit reductions).
Validation columns 15, 16, 17 and test columns 16, 17 are constant (zero-based
indices); every column is preserved. Each run records all 27 feature means and
standard deviations, shapes, dtypes, finite-value checks, duration ranges, counts,
archive keys, and hashes in its manifest.

An audit using **training only** `m = train_arr_2.mean(axis=0)` and
`s = train_arr_2.std(axis=0)` (NumPy float32 reductions, `ddof=0`) found no zero
training scales. Applying `(arr_2 - m) / s` across partitions approximates
`arr_0`: train/valid/test maximum absolute errors are respectively `7.63e-6`,
`1.43e-6`, `1.43e-6`, with RMSE `2.58e-7`, `2.57e-7`, `2.57e-7`.
This verifies a numerical relationship, not original archive-generation
provenance. The adapter does not fit or apply this transform. `arr_2` may already
be encoded or imputed; it is not established as raw clinical data.

Inspection of the pinned repository's README, `data.py`, `main.py`, configs, and
Python source found no SUPPORT archive-generation pipeline establishing feature
names, imputation choices, cohort filtering, or inverse time normalization.
Those details remain unknown. The README's `configs/support.json` is absent;
`configs/train.json` specifies RMSprop and gradient clip 100 but is not a complete
chosen benchmark configuration. The table below defines the PyHealth training configuration.

## Exact label and time contract

The verified reference entry point is `data.py:get_mimic_dataloader`, used for
SUPPORT in the `main.py` partition loop. The adapter maps each row as:

```python
features = arr_0[row].astype("float32")
duration = (arr_1[:, 0] + 0.001).astype("float32")[row]
event = arr_1[row, 1].astype("float32")  # 1 observed, 0 right censored
```

The offset is added in archive floating precision **before** conversion, once
per partition. No further offset is applied on checkpoint reload or prediction.
The input schema is `{"features": "tensor"}`; the output schema is
`{"duration": "tensor", "event": "tensor"}`. Batches have `[B, 27]` features
and scalar `[B]` labels. The model uses `mode=None`. Fitted training tensor
processors are reused for validation/test and perform no normalization.
Partition and source row metadata are never features.

Raw duration ranges in train and validation are
`[0.00821355264633894, 5.555099010467529]`; test has the same minimum and maximum
`5.5359344482421875`. The loader exposes no reliable inverse mapping to calendar
days. `original_calendar_unit_mapping` is `unknown`. Curves use model units:
**released SUPPORT time units + 0.001**. A future query at raw released horizon
`h` must query the curve at `h + 0.001`, once. Do not label these curves as days
or interpret their grid as clinical month horizons.

## Frozen baseline and reference differences

| Setting | Value |
| --- | --- |
| Split; seed | 1; 0 |
| Device; dtype; intra-op threads | CPU; float32; 1 |
| Hidden dimensions; components | [64, 64, 64]; 5 |
| Activation; BatchNorm; minimum scale | PReLU; false; 1e-4 |
| Optimizer; learning rate; weight decay | Adam; 0.001; 0 |
| Training batch size; drop last | 256; false |
| Maximum epochs; patience; gradient norm cap | 100; 10; 100 |
| Monitor; direction; restore | Validation NLL; minimize; best checkpoint |
| Validation batch | Entire validation partition (1776 rows for split 1) |

Python, NumPy, PyTorch, and the private training-loader generator are seeded;
PyTorch deterministic algorithms are enabled. The manifest records software,
platform, intra/inter-op thread counts, PyHealth Git SHA and dirty status.
Determinism claims apply to the same software/hardware environment, not arbitrary
platforms. The CLI permits split/seed/epoch/patience changes for integration
checks, and records them; the final baseline uses the table above.

The core network/numerics remain unchanged. Like the reference default network,
it initializes mean-head weights to zero and biases evenly over [-3, 3]; other
heads use PyTorch linear initialization rather than the reference's optional
`fixed` initialization or residual-network initialization. Reference default
hidden layers use BatchNorm, while this baseline disables it. The reference
clamps hidden inputs for the weight head to [-1000, 1000], mean outputs to
[-30, 30], and raw scale outputs to [0.01, 100]; the core omits these clamps and
uses `softplus(raw_scale) + 1e-4`.

The core uses stable inverse-softplus, log-density, and log-survival operations
instead of the reference's CDF subtraction, epsilon adjustments, and hazard
floor. See the [model documentation](../docs/api/models/pyhealth.models.SurvivalMDN.rst)
for the precise numerical comparison. The reference sorts by duration and uses
an ordered random batch sampler (default seed 13, sorted indices within batches),
dropping its final training batch. This example retains archive order, shuffles
training via PyHealth's seeded loader, keeps every row, and uses Adam rather
than RMSprop. The recorded configuration and these implementation differences define the
PyHealth baseline used here.

Only validation loss selects checkpoints and triggers early stopping. Test
labels never reach `Trainer.train` or prediction calls. One full validation
batch makes Trainer's mean-of-batch-loss reduction exactly a row mean; the
initial training NLL also uses one full batch. No epoch number is guessed from
the requested epoch budget. The reported final validation NLL is evaluated
after best-state restoration.

## Commands and artifacts

From the PyHealth repository root, using a compatible installed Python:

```bash
python examples/survival_mdn_support.py --data-dir ../Survival-MDN/data/support --split 1 --seed 0 --epochs 2 --output-dir output/support-smoke
python examples/survival_mdn_support.py --data-dir ../Survival-MDN/data/support --split 1 --seed 0 --output-dir output/support-baseline
python examples/survival_mdn_support.py --data-dir ../Survival-MDN/data/support --split 1 --seed 0 --output-dir output/support-repeat
python examples/survival_mdn_support.py --data-dir ../Survival-MDN/data/support --predict-only output/support-baseline --output-dir output/support-reloaded
```

Use a new/empty output directory for each run. Prediction-only mode uses the
saved split, seed, model settings, units, and exact grid, verifies source archive
hashes/metadata, reconstructs compatible tensor processors, and copies the
selected checkpoint into the new output directory. It never trains.

A run produces:

- `checkpoints/best.ckpt`: selected state dictionary; no full model or dataset
  serialization. Trainer also writes `last.ckpt` and its log during training.
- `manifest.json`: source revision, partition hashes/audit/schema, complete
  model/training configuration, seed/device/dtype, software/Git/thread metadata,
  relative checkpoint path, initial NLLs, restored validation NLL, training
  elapsed time, and the exact prediction grid. Prediction-only runs preserve
  the training provenance and add their replay environment.
- `test_predictions.npz`: `survival [N_test, 128]`, `times [128]`, transformed
  `duration`, original `released_duration`, `event`, integer `source_row_index`,
  ordinary Unicode `record_id`, and `partition`. Load with `allow_pickle=False`.
  Arrays retain archive test-row order; labels are exported only for later
  evaluation.

The float32 grid is `linspace(0, max(transformed_training_duration), 128)`,
computed from training alone and saved exactly. It demonstrates prediction;
it covers the range of transformed training durations. Predictions are
batched in groups of at most 256 in eval/no-grad mode, with features only.
Every run checks row coverage/order, shape, finite probabilities, [0, 1] bounds,
nonincreasing curves within `1e-6`, and exact ones at zero. It reloads the best
checkpoint into a fresh model and requires exact agreement for all test rows.

## Offline validation

Tests generate tiny artificial archives and require no clinical data or
reference repository. They cover the mapping/precision/offset, processors,
identity, invalid contracts, ignored extra arrays, reproducibility, complete
validation batches, partial training/prediction batches, features-only
prediction, manifest compatibility, overwrite refusal, and end-to-end replay.

```bash
python -m unittest discover -t tests -s tests/core -p 'test_survival_mdn_support.py' -v
python -m unittest discover -t tests -s tests/core -p 'test_survival_mdn.py' -v
python -m ruff check examples/survival_mdn_support.py tests/core/test_survival_mdn_support.py
git diff --check
python tools/check_pr_rules.py --base feat/survival-mdn-core --head HEAD
```

The adapter branch boundary is core commit
`8a268b02d384c1edd8d5429f429fda8d9cd0d05c`. The intended initial PR base is `feat/survival-mdn-core`. After core
lands, verify the actual destination branch and preserve this boundary when
rebasing adapter-only commits after a squash/rebase merge. No PR is created or
pushed by this example.
