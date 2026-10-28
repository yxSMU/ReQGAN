# Experiment Guide

Run commands from the project root. Inspect an entry's arguments with `python run.py NAME --help`. For each evaluation, supply the checkpoint created by the matching training run and the same circuit depth.

| Paper experiment | Training entry | Evaluation entry | Change from main model |
| --- | --- | --- | --- |
| MNIST digits 0–9 | `mnist_main --classes N` | `mnist_eval --classes N` | Main model, one run per digit |
| Fashion-MNIST classes 0–9 | `fashion_main --classes N` | `fashion_eval --classes N` | Dataset |
| Digit 2, uniform noise | `noise_uniform_digit2` | `noise_ablation_eval_digit2` | Replace learned circuit noise with uniform noise |
| Digit 2, Gaussian noise | `noise_gaussian_digit2` | `noise_ablation_eval_digit2` | Replace learned circuit noise with Gaussian noise |
| Digit 7, direct `/max` mapping | `mapping_max_digit7` | `mapping_max_eval_digit7` | Replace calibrated intensity mapping |
| Digit 6, no temperature smoothing | `without_temperature_digit6` | `without_temperature_eval_digit6` | Remove temperature smoothing |
| Digit 6, no deviation response | `without_deviation_digit6` | `without_deviation_eval_digit6` | Remove nonlinear deviation response |
| Digit 6, no contrast normalization | `without_contrast_digit6` | `without_contrast_eval_digit6` | Remove per-image contrast normalization |
| Digit 6, no affine correction | `without_affine_digit6` | `without_affine_eval_digit6` | Remove learned affine correction |
| PQWGAN, Patch=1, 32×32 | `pqwgan_patch1_32` | `pqwgan_patch1_32_eval` | Separate baseline; evaluation uses `--config TRAIN_OUT/model_config.json` |

Example ablation:

```bash
python run.py without_temperature_digit6 --layers 20 --per_class_size 1000 --epochs 50 --out outputs/no_temperature_digit6
python run.py without_temperature_eval_digit6 --ckpt outputs/no_temperature_digit6/weights_batch_000050.pt --layers 20 --classes 6 --n_images 250 --out outputs/no_temperature_digit6_eval
```

For the PQWGAN baseline:

```bash
python run.py pqwgan_patch1_32 --classes 0 --layers 20 --per_class_size 1000 --epochs 50 --out outputs/pqwgan_digit0
python run.py pqwgan_patch1_32_eval --config outputs/pqwgan_digit0/model_config.json --n_images 250 --out outputs/pqwgan_digit0_eval
```

The PQWGAN baseline adapts [PQWGAN](https://github.com/jasontslxd/PQWGAN) to a 32×32 image with one global generator, 10 data qubits, and one ancilla. The paper does not specify every baseline hyperparameter. FID/KID in the supplied evaluation scripts use flattened image pixels; label results as **pixel-space FID/KID**.
