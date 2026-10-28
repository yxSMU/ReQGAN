# ReQGAN

Code for **End-to-End QGAN-Based Image Synthesis via Neural Noise Encoding and Intensity Calibration** ([paper](https://arxiv.org/html/2603.18554v1)).

## Setup

Use Python 3.10 with TensorFlow 2.17, TensorCircuit 0.12, PyTorch 2.14, torchvision 0.29, NumPy, SciPy, Pillow, Matplotlib, and tqdm. MNIST and Fashion-MNIST download automatically on first use. Run all commands from this project directory. Outputs should be placed under `outputs/`, which is ignored by Git.

## Find an experiment

```bash
python run.py list
python run.py mnist_main --help
```

`run.py` resolves the experiment names in `experiment_manifest.json`. Command-line arguments placed after the experiment name override preset arguments.

## Main experiment

Train one MNIST class at a time (`0` through `9`):

```bash
python run.py mnist_main --classes 0 --layers 20 --per_class_size 1000 --batch_size 5 --epochs 50 --out outputs/mnist_digit0
python run.py mnist_eval --classes 0 --ckpt outputs/mnist_digit0/weights_batch_000050.pt --layers 20 --n_images 250 --out outputs/mnist_digit0_eval
```

Choose an existing checkpoint file from the training output. The evaluator uses 250 examples after the first 1000 examples of each class, avoiding training overlap. For Fashion-MNIST, replace `mnist_main`/`mnist_eval` with `fashion_main`/`fashion_eval` and set the matching checkpoint path.

## Change parameters

| Argument | Meaning | Main setting |
| --- | --- | --- |
| `--classes` | Digit or class ID; train classes separately for class-wise results | `0` to `9` |
| `--layers` | Quantum circuit depth; use the same value for training and evaluation | `20` in the provided scripts |
| `--n_data_qubits` | Data qubits for a 32×32 image | `10` |
| `--latent_dim` | Neural encoder input dimension | `10` |
| `--per_class_size` | Training samples per selected class | `1000` |
| `--batch_size`, `--epochs` | Training schedule | `5`, `50` |
| `--out` | Output directory | user chosen |

## Ablations and baseline

See [Experiment Guide](docs/experiments.md) for exact commands, fixed changes, and evaluation entry points. Use the same `--classes`, `--layers`, sample count, and training schedule when comparing variants. The evaluators retain the original projects' pixel-vector FID/KID computation.
