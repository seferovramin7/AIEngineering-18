# app/main.py
# -------------------------------------------------------
# Transfer Learning demo with Fine-Tuning depth control
# Examples:
#   python app/main.py demo-train --epochs 2 --image-size 160
#   python app/main.py demo-train --epochs 2 --image-size 160 --train-last-n 40
# If no dataset paths are provided, it falls back to CIFAR-10.
# Metrik akhir disimpan ke JSON di:
#   <project_root>/outputs/ftda/milzon18/runs/   (jika ada)
#   atau ./runs/ (fallback)
# -------------------------------------------------------

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

AUTOTUNE = tf.data.AUTOTUNE
tf.get_logger().setLevel("ERROR")


# --------------------------
# Output directory resolver
# --------------------------
def resolve_runs_dir() -> Path:
    """Cari folder runs yang 'benar' di dalam repo, fallback ke CWD/runs."""
    # __file__ = app/main.py → project_root = parent of 'app'
    project_root = Path(__file__).resolve().parents[1]

    # Prefer: outputs/ftda/milzon18/runs
    candidate = project_root / "outputs" / "ftda" / "milzon18" / "runs"
    if candidate.parent.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    # Legacy fallback: <project_root>/milzon18/runs
    legacy = project_root / "milzon18" / "runs"
    if legacy.parent.exists():
        legacy.mkdir(parents=True, exist_ok=True)
        return legacy

    # Last resort: ./runs (relative to CWD)
    fallback = Path.cwd() / "runs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# --------------------------
# Dataset utilities
# --------------------------
def make_dir_datasets(
    train_dir: Path,
    val_dir: Optional[Path],
    image_size: int,
    batch_size: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Load datasets from folders via image_dataset_from_directory."""
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    common_kwargs = dict(
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True, **common_kwargs
    )

    if val_dir and val_dir.exists():
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir, shuffle=False, **common_kwargs
        )
    else:
        # 90/10 split from train if no explicit val dir
        card = int(tf.data.experimental.cardinality(train_ds).numpy())
        val_batches = max(1, int(round(card * 0.1)))
        val_ds = train_ds.take(val_batches)
        train_ds = train_ds.skip(val_batches)

    num_classes = len(train_ds.class_names)

    # Cache & prefetch
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, num_classes


def make_cifar10_datasets(
    image_size: int,
    batch_size: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Fallback dataset to guarantee the demo runs: CIFAR-10."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze().astype("int32")
    y_test = y_test.squeeze().astype("int32")

    # build val split from train
    val_n = 5000
    x_val, y_val = x_train[-val_n:], y_train[-val_n:]
    x_train, y_train = x_train[:-val_n], y_train[:-val_n]

    def _prep(x, y):
        x = tf.image.resize(tf.cast(x, tf.float32), (image_size, image_size))
        return x, y

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .map(_prep, num_parallel_calls=AUTOTUNE)
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .map(_prep, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return train_ds, val_ds, 10


# --------------------------
# Model & augmentation
# --------------------------
def data_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def set_trainable_layers(base_model: tf.keras.Model, train_last_n: int):
    """Freeze semua layer kecuali N terakhir."""
    total = len(base_model.layers)
    if train_last_n <= 0:
        for l in base_model.layers:
            l.trainable = False
        print(f"[freeze] All {total} layers frozen (train_last_n={train_last_n})")
        return
    freeze_upto = max(0, total - train_last_n)
    for l in base_model.layers[:freeze_upto]:
        l.trainable = False
    for l in base_model.layers[freeze_upto:]:
        l.trainable = True
    print(f"[freeze] Frozen {freeze_upto}/{total} → training last {train_last_n}")


def build_model(
    num_classes: int,
    image_size: int,
    backbone: str = "mobilenetv2",
    train_last_n: int = 20,
) -> tf.keras.Model:
    backbone = backbone.lower()
    if backbone == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import (
            MobileNetV2,
            preprocess_input,
        )
        base = MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
        )
        preprocess = preprocess_input
    elif backbone in {"efficientnetb0", "efficientnet"}:
        from tensorflow.keras.applications.efficientnet import (
            EfficientNetB0,
            preprocess_input,
        )
        base = EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
        )
        preprocess = preprocess_input
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    set_trainable_layers(base, train_last_n)

    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = data_augmentation()(inputs)
    x = layers.Lambda(preprocess, name="preprocess")(x)
    # Call backbone with training=False for BN stability in short runs
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name=f"tl_{backbone}")


# --------------------------
# Train routine
# --------------------------
def run_training(args):
    # Load data
    if args.dataset_dir:
        root = Path(args.dataset_dir)
        train_ds, val_ds, num_classes = make_dir_datasets(
            root / "train",
            (root / "val") if (root / "val").exists() else None,
            args.image_size,
            args.batch_size,
        )
    elif args.train_dir:
        train_ds, val_ds, num_classes = make_dir_datasets(
            Path(args.train_dir),
            Path(args.val_dir) if args.val_dir else None,
            args.image_size,
            args.batch_size,
        )
    else:
        print("[data] No dataset path provided → using CIFAR-10 fallback.")
        train_ds, val_ds, num_classes = make_cifar10_datasets(
            args.image_size, args.batch_size
        )

    # Build & compile
    model = build_model(
        num_classes=num_classes,
        image_size=args.image_size,
        backbone=args.backbone,
        train_last_n=args.train_last_n,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = []
    if args.early_stop:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=2, restore_best_weights=True
            )
        )

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    # Final metrics
    tr_hist = hist.history.get("accuracy", [None])
    va_hist = hist.history.get("val_accuracy", [None])
    final_train = float(tr_hist[-1]) if tr_hist[-1] is not None else None
    final_val = float(va_hist[-1]) if va_hist[-1] is not None else None
    print(f"\n[final] train_accuracy={final_train:.4f}  val_accuracy={final_val:.4f}")

    # Save metrics JSON
    out_dir = resolve_runs_dir()
    payload = {
        "train_accuracy": final_train,
        "val_accuracy": final_val,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "train_last_n": args.train_last_n,
        "backbone": args.backbone,
        "learning_rate": args.learning_rate,
    }
    out_path = out_dir / f"final_metrics_last{args.train_last_n}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] Metrics saved → {out_path.resolve()}")


# --------------------------
# CLI
# --------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Transfer Learning demo: fine-tuning depth & augmentation"
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo-train", help="Run a short training demo")
    d.add_argument("--epochs", type=int, default=2)
    d.add_argument("--image-size", type=int, default=160)
    d.add_argument("--batch-size", type=int, default=32)
    d.add_argument("--learning-rate", type=float, default=1e-4)
    d.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv2",
        choices=["mobilenetv2", "efficientnetb0", "efficientnet"],
    )
    d.add_argument(
        "--train-last-n",
        type=int,
        default=20,
        help="Number of final layers kept trainable",
    )
    # Dataset args
    d.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Folder with subfolders 'train' and optional 'val'",
    )
    d.add_argument("--train-dir", type=str, default=None, help="Explicit train dir")
    d.add_argument("--val-dir", type=str, default=None, help="Explicit val dir")
    d.add_argument("--early-stop", action="store_true", help="Enable EarlyStopping")
    return p


def main():
    # QoL: enable GPU memory growth if possible
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "demo-train":
        run_training(args)
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
