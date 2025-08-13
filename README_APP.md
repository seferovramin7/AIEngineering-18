# App: Keras Transfer Learning & Data Augmentation (CLI)

This app is a small, readable CLI that teaches the core mechanics behind transfer learning and image data augmentation in Keras. It is intentionally minimal so students can map each line of code to the conceptual steps.

## Purpose and scope

- Show how augmentation layers are composed and applied to image tensors.
- Demonstrate the two-phase transfer-learning workflow (freeze base → fine-tune top layers).
- Keep a clear separation between explanation-only output and compute-heavy demos.

## Project layout

- `app/main.py`: single-file CLI containing all logic
- `outputs/`: folder where artifacts are written (augmented images, trained model)

## CLI modes (high level behavior)

- `explain`
  - Prints a concise overview of the topic and the typical Keras workflow.
  - No ML frameworks required to run; designed to always work.

- `demo-augment`
  - Builds a small `tf.keras.Sequential` of preprocessing layers:
    - `RandomFlip('horizontal')`, `RandomRotation(0.1)`, `RandomZoom(0.1)`
  - Applies these to a synthetic 128×128 RGB tensor to make the behavior observable without an external dataset.
  - Saves several augmented samples into `outputs/` as PNGs so students can visually inspect the effects.

- `demo-train`
  - Loads a lightweight subset of cats-vs-dogs (fetched via `keras.utils.get_file`).
  - Constructs a model with `MobileNetV2` (ImageNet weights) as a frozen feature extractor and a small classification head.
  - Trains the head first, then unfreezes only the top ~20 layers and fine-tunes with a lower learning rate.
  - Saves the resulting Keras model into `outputs/transfer_learning_model`.

## Internal design notes

- Single entrypoint (`main()`): parses a `mode` plus a few optional flags (`--epochs`, `--image-size`, `--samples`).
- Pure-explanation path: the `explain()` function has no heavy imports, guaranteeing a successful run even without TensorFlow installed.
- Optional dependencies: augmentation/training paths attempt to import TensorFlow (and NumPy/Pillow for saving images) and fail gracefully with a helpful message if missing.
- Data augmentation is applied as model layers, not offline preprocessing, so it naturally integrates into the input pipeline and only runs during training.
- Transfer learning is split into two explicit stages to make the transition (freeze → unfreeze) visible and easy to modify.

## Data flow by mode

- `demo-augment`
  1) Create a synthetic image tensor (random noise) to avoid external data.
  2) Apply augmentation layers with `training=True` to ensure random transforms are active.
  3) Convert to `PIL.Image` and write PNGs under `outputs/`.

- `demo-train`
  1) Download and load image folders as `tf.data.Dataset` objects.
  2) Cache/shuffle/prefetch for simple performance hygiene.
  3) Forward pass through augmentation → preprocessing → base CNN → pooling → dense head.
  4) Stage 1: compile with higher LR; train only the head (base frozen).
  5) Stage 2: unfreeze top layers; compile with lower LR; fine-tune briefly.
  6) Save the model artifact under `outputs/`.

## Key functions and responsibilities

- `explain()` — prints a short curriculum-style summary of the topic and workflow.
- `demo_augment(samples)` — constructs augmentation stack, generates and saves `samples` images.
- `demo_train(epochs, image_size)` — builds datasets and the transfer-learning model, performs two-phase training, saves the model.

## Error handling and exit codes

- If TensorFlow or dependencies are unavailable for demos, the app prints a clear message and exits with non-zero status (2) to signal missing prerequisites.
- Network errors during dataset download will surface as exceptions from Keras utilities; students can substitute their own dataset directory by adapting the loader.

## Extensibility paths for students

- Swap backbones: try `EfficientNetB0` or `ResNet50` and compare.
- Tweak augmentation strengths or add color jitter (e.g., `RandomContrast`).
- Change fine-tuning depth by adjusting how many layers remain trainable.
- Replace the dataset loader with a custom directory structure or TFRecords.

## Outputs

- Augmented samples: `outputs/augmented_#.png` (from `demo-augment`).
- Saved model: `outputs/transfer_learning_model` (from `demo-train`).

## How to run

- From the project root, invoke the CLI as:
```bash
python app/main.py <mode> [flags]
```
- Modes:
  - `explain`: prints the conceptual overview (no TensorFlow required).
  - `demo-augment`: saves augmented images to `outputs/`.
    - Flags: `--samples <int>` number of images to generate (default 4).
  - `demo-train`: runs transfer learning + fine-tuning and saves a model to `outputs/transfer_learning_model`.
    - Flags: `--epochs <int>` (default 1), `--image-size <int>` (default 160), `--data-dir <path>` to use a local dataset with `train/` and `validation/` subfolders.
- Requirements:
  - `explain` works in any basic Python 3.9+ environment.
  - `demo-augment` and `demo-train` require TensorFlow (and Pillow/NumPy for saving images).

### Ready-to-run commands

```bash
# Explain the topic (no heavy dependencies required)
python app/main.py explain
```

```bash
# Augmentation demo: generate 6 augmented samples into outputs/
python app/main.py demo-augment --samples 6
```

```bash
# Transfer-learning demo: 1 epoch, image size 160 (downloads small dataset)
python app/main.py demo-train --epochs 1 --image-size 160

# Transfer-learning demo using a local dataset directory with train/ and validation/
python app/main.py demo-train --epochs 1 --image-size 160 --data-dir /path/to/dataset
```
