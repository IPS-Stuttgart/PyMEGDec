# CLI reference

PyMEGDec exposes one grouped command plus compatibility entry points. Prefer the grouped `pymegdec` command for new documentation and scripts.

## Grouped command

```bash
pymegdec --help
pymegdec alpha --help
pymegdec stimulus --help
```

Core workflows:

```bash
pymegdec cross-validate --participant 2
pymegdec transfer --participant 2 --classifier multiclass-svm
```

Stimulus workflows:

```bash
pymegdec stimulus decoding --participants 2 --output outputs/part2_stimulus_decoding.csv
pymegdec stimulus predictions --participants 2 --output outputs/part2_stimulus_predictions.csv
pymegdec stimulus robustness --participants 2
pymegdec stimulus temporal-generalization --participants 2 --output outputs/part2_temporal_generalization.csv
pymegdec stimulus onset-scan --participants 2 --output outputs/part2_onset_scan.csv
```

Alpha workflows:

```bash
pymegdec alpha metrics --participant 2 --output outputs/part2_alpha_metrics.csv
pymegdec alpha movement --participants 2 --trajectory-output outputs/part2_alpha_movement.csv
pymegdec alpha movement-results --movement-summary outputs/part2_alpha_movement_summary.csv --effect-output outputs/part2_alpha_movement_effects.csv --condition-summary-output outputs/part2_alpha_movement_condition_summary.csv
pymegdec alpha reaction-time --participants 2 --joined-output outputs/part2_alpha_rt_joined.csv --summary-output outputs/part2_alpha_rt_summary.csv
```

## Compatibility aliases

Single-token aliases remain available for existing shell scripts:

```bash
pymegdec stimulus-decoding ...
pymegdec stimulus-predictions ...
pymegdec stimulus-robustness ...
pymegdec stimulus-temporal-generalization ...
pymegdec stimulus-onset-scan ...
pymegdec alpha-metrics ...
pymegdec alpha-movement ...
pymegdec alpha-movement-results ...
pymegdec alpha-reaction-time ...
```

The package also installs direct console scripts:

```bash
pymegdec-cross-validate
pymegdec-transfer
pymegdec-stimulus-decoding
pymegdec-stimulus-predictions
pymegdec-stimulus-robustness
pymegdec-stimulus-temporal-generalization
pymegdec-stimulus-onset-scan
pymegdec-alpha-metrics
pymegdec-alpha-movement
pymegdec-alpha-movement-results
pymegdec-alpha-reaction-time
```

Top-level Python scripts remain available for old workflows:

```bash
python export_alpha_metrics.py --participant 2 --output outputs/part2_alpha_metrics.csv
python analyze_alpha_movement.py --participants 2 --trajectory-output outputs/part2_alpha_movement.csv --summary-output outputs/part2_alpha_movement_summary.csv
python analyze_alpha_reaction_time.py --participants 2 --joined-output outputs/part2_alpha_rt_joined.csv --summary-output outputs/part2_alpha_rt_summary.csv
python scripts/export_stimulus_predictions.py --participants 2 --output outputs/part2_predictions.csv
python scripts/export_stimulus_robustness.py --participants 2
python scripts/export_stimulus_temporal_generalization.py --participants 2
python scripts/export_stimulus_onset_scan.py --participants 2
```

## Shared decoding options

The cross-validation, transfer, and stimulus commands share the core decoding options where applicable:

| Option | Meaning | Typical value |
| --- | --- | --- |
| `--data-dir` | Directory containing participant MAT files. | `/path/to/MEG-Data` |
| `--participant` / `--participants` | Participant id or participant-id range. | `2`, `1-4,6,8` |
| `--window-size` | Window duration in seconds. | `0.1` |
| `--null-window-center` | Null-window center, or `nan`. | `nan` or `-0.2` |
| `--new-framerate` | Target frame rate, or `inf`. | `inf` |
| `--classifier` | Classifier registry name. | `multiclass-svm` |
| `--classifier-param` | Numeric, JSON, or Python literal parameter. | `1.0` |
| `--components-pca` | PCA component count, or `inf`. | `100` |
| `--frequency-range LOW HIGH` | Frequency range in Hz. | `0 inf` |
| `--transfer-direction` | Direction for main/cue transfer. | `main-to-cue` |

## Examples

Run train-main / validate-cue decoding across a time range:

```bash
pymegdec stimulus decoding \
  --data-dir /path/to/MEG-Data \
  --participants 2 \
  --time-window=-0.2,0.6 \
  --window-step-s 0.05 \
  --output outputs/part2_stimulus_decoding.csv \
  --summary-output outputs/part2_stimulus_decoding_summary.csv \
  --plots-dir outputs/part2_stimulus_decoding_plots
```

Export trial-level diagnostics at selected windows:

```bash
pymegdec stimulus predictions \
  --data-dir /path/to/MEG-Data \
  --participants 2 \
  --window-centers=-0.175,0.175 \
  --output outputs/part2_stimulus_predictions.csv \
  --summary-output outputs/part2_stimulus_prediction_summary.csv \
  --confusion-output outputs/part2_stimulus_confusion.csv \
  --per-stimulus-output outputs/part2_stimulus_per_class.csv
```

Analyze alpha metrics against reaction time:

```bash
pymegdec alpha reaction-time \
  --data-dir /path/to/MEG-Data \
  --participants 2 \
  --joined-output outputs/part2_alpha_rt_joined.csv \
  --summary-output outputs/part2_alpha_rt_summary.csv \
  --plots-dir outputs/part2_alpha_rt_plots
```
