#!/usr/bin/env python3
"""Train or evaluate an isolated EfficientNetV2 Stanford Cars candidate.

This tool never writes the deployed ``best_car_model.keras`` artifact. Training
selects checkpoints only by validation accuracy, then evaluates the selected
candidate on the test split once for reporting. Every output stays below the
ignored ``training_runs/<run-name>/`` directory.

TensorFlow is loaded only after an explicit ``--train`` or ``--evaluate-run``
operation has been parsed, so ``--help`` and the model-free tests remain light.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
MAPPING_PATH = ROOT / "class_mapping.json"
RUNS_DIR = ROOT / "training_runs"

IMAGE_SIZE = 224
TRAIN_LOAD_SIZE = 256
NUM_CLASSES = 196
RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")

# Populated only by load_ml_stack(), after the CLI operation is explicit.
tf: Any = None
keras: Any = None
layers: Any = None


@dataclass(frozen=True)
class RunPaths:
    directory: Path
    checkpoint: Path
    history: Path
    metrics: Path
    status: Path
    epoch_log: Path
    evaluation: Path


def resolve_run_paths(run_name: str) -> RunPaths:
    """Return fixed outputs for one safe, single-component run name."""
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "run name must be 1-64 letters, digits, dots, underscores, or hyphens"
        )
    resolved_root = RUNS_DIR.resolve()
    if resolved_root.parent != ROOT.resolve():
        raise ValueError("training run root must stay directly below the repository")
    directory = RUNS_DIR / run_name
    if directory.resolve().parent != resolved_root:
        raise ValueError("run name must stay directly below the training run root")
    return RunPaths(
        directory=directory,
        checkpoint=directory / "best_val.keras",
        history=directory / "history.json",
        metrics=directory / "metrics.json",
        status=directory / "status.json",
        epoch_log=directory / "epochs.jsonl",
        evaluation=directory / "evaluation.json",
    )


def load_ml_stack() -> None:
    """Load the heavy training stack only when an operation needs it."""
    global tf, keras, layers
    try:
        import tensorflow as tensorflow_module
        from tensorflow import keras as keras_module
        from tensorflow.keras import layers as layers_module
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow training dependencies are missing; install requirements.txt"
        ) from exc
    tf = tensorflow_module
    keras = keras_module
    layers = layers_module


def write_json(path: Path, payload: Any) -> None:
    """Replace a run-owned JSON file atomically."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_status(paths: RunPaths, **fields: Any) -> None:
    payload: dict[str, Any] = {}
    if paths.status.exists():
        try:
            payload = json.loads(paths.status.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    if fields.get("state") in {"starting", "running", "completed"}:
        payload.pop("error", None)
    payload.update({"updated_at": time.time(), **fields})
    write_json(paths.status, payload)


def load_ordered_class_names(path: Path = MAPPING_PATH) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    index_to_class = mapping["index_to_class"]
    names = [index_to_class[str(index)] for index in range(len(index_to_class))]
    if len(names) != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} classes, found {len(names)}")
    return names


def validate_dataset(class_names: list[str]) -> None:
    if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"dataset directories must exist at {TRAIN_DIR} and {TEST_DIR}"
        )
    train_folders = sorted(path.name for path in TRAIN_DIR.iterdir() if path.is_dir())
    test_folders = sorted(path.name for path in TEST_DIR.iterdir() if path.is_dir())
    if train_folders != class_names or test_folders != class_names:
        raise ValueError(
            "class_mapping.json must exactly match the data/train and data/test folders"
        )


def configure_runtime(allow_cpu: bool) -> str:
    """Use mixed precision on GPU and an explicit float32 opt-in on CPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if not allow_cpu:
            raise RuntimeError(
                "no GPU detected; pass --allow-cpu to accept slow float32 execution"
            )
        keras.mixed_precision.set_global_policy("float32")
        print("GPUs: none (explicit CPU run)")
        print("Policy: float32")
        return "cpu_float32"

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"memory growth skipped: {exc}")
    keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"GPUs: {gpus}")
    print("Policy: mixed_float16")
    return "gpu_mixed_float16"


def build_model() -> tuple[Any, Any]:
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    # The API supplies RGB float32 in [0, 1]. Keep the conversion inside the
    # graph and disable the application's built-in [0, 255] preprocessing.
    x = layers.Rescaling(2.0, offset=-1.0, name="efficientnet_v2_preprocess")(inputs)
    backbone = keras.applications.EfficientNetV2S(
        include_top=False,
        include_preprocessing=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling=None,
        name="efficientnetv2-s",
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)
    model = keras.Model(inputs, outputs, name="stanford_cars_efficientnetv2s")
    return model, backbone


def compile_model(model: Any, learning_rate: float, weight_decay: float = 0.0) -> None:
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        ],
    )


def freeze_batch_norm(backbone: Any) -> None:
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def unfreeze_backbone(backbone: Any, freeze_ratio: float = 0.45) -> None:
    backbone.trainable = True
    freeze_until = int(len(backbone.layers) * freeze_ratio)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= freeze_until
    freeze_batch_norm(backbone)
    trainable = sum(int(layer.trainable) for layer in backbone.layers)
    print(
        f"Fine-tune: froze first {freeze_until}/{len(backbone.layers)} layers; "
        f"{trainable} trainable (batch normalization frozen)"
    )


_AUGMENTERS: tuple[Any, ...] | None = None


def get_augmenters() -> tuple[Any, ...]:
    global _AUGMENTERS
    if _AUGMENTERS is None:
        _AUGMENTERS = (
            layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03),
            layers.RandomContrast(0.15),
        )
    return _AUGMENTERS


def augment_batch(images: Any, labels: Any) -> tuple[Any, Any]:
    crop, flip, rotation, contrast = get_augmenters()
    images = crop(images, training=True)
    images = flip(images, training=True)
    images = rotation(images, training=True)
    images = contrast(images, training=True)
    images = tf.image.random_brightness(images, 0.1)
    images = tf.image.random_saturation(images, 0.9, 1.1)
    images = tf.cast(tf.clip_by_value(images, 0.0, 255.0) / 255.0, tf.float32)
    return images, tf.cast(labels, tf.float32)


def center_resize(images: Any, labels: Any) -> tuple[Any, Any]:
    images = tf.image.resize(images, (IMAGE_SIZE, IMAGE_SIZE))
    return tf.cast(images / 255.0, tf.float32), tf.cast(labels, tf.float32)


def mixup_batch(images: Any, labels: Any, alpha: float = 0.2) -> tuple[Any, Any]:
    batch_size = tf.shape(images)[0]
    gamma1 = tf.random.gamma([], alpha, dtype=tf.float32)
    gamma2 = tf.random.gamma([], alpha, dtype=tf.float32)
    mixture = gamma1 / (gamma1 + gamma2)
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = mixture * images + (1.0 - mixture) * tf.gather(images, indices)
    mixed_labels = mixture * labels + (1.0 - mixture) * tf.gather(labels, indices)
    return mixed_images, mixed_labels


def make_datasets(class_names: list[str], batch_size: int, seed: int):
    common = {
        "labels": "inferred",
        "label_mode": "categorical",
        "class_names": class_names,
        "batch_size": batch_size,
    }
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(TRAIN_LOAD_SIZE, TRAIN_LOAD_SIZE),
        shuffle=True,
        seed=seed,
        validation_split=0.1,
        subset="training",
        **common,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        shuffle=False,
        seed=seed,
        validation_split=0.1,
        subset="validation",
        **common,
    )
    test_ds = make_test_dataset(class_names, batch_size)
    autotune = tf.data.AUTOTUNE
    train_ds = (
        train_ds.map(augment_batch, num_parallel_calls=autotune)
        .map(mixup_batch, num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    val_ds = val_ds.map(center_resize, num_parallel_calls=autotune).prefetch(autotune)
    return train_ds, val_ds, test_ds


def make_test_dataset(class_names: list[str], batch_size: int):
    dataset = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        shuffle=False,
    )
    return dataset.map(center_resize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )


def epoch_status_callback(paths: RunPaths, phase: str):
    class EpochStatus(keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            values = logs or {}
            row = {
                "phase": phase,
                "epoch": int(epoch) + 1,
                "accuracy": float(values.get("accuracy", 0)),
                "val_accuracy": float(values.get("val_accuracy", 0)),
                "top_5_accuracy": float(values.get("top_5_accuracy", 0)),
                "val_top_5_accuracy": float(values.get("val_top_5_accuracy", 0)),
                "loss": float(values.get("loss", 0)),
                "val_loss": float(values.get("val_loss", 0)),
            }
            with paths.epoch_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            write_status(paths, phase=phase, epoch=row["epoch"], last_epoch=row)

    return EpochStatus()


def evaluate(model: Any, dataset: Any, split: str) -> dict[str, float]:
    results = model.evaluate(dataset, verbose=1, return_dict=True)
    values = {key: float(value) for key, value in results.items()}
    print(
        f"{split}: loss={values['loss']:.4f} "
        f"top1={values['accuracy'] * 100:.2f}% "
        f"top5={values['top_5_accuracy'] * 100:.2f}%"
    )
    return values


def merge_histories(histories: list[dict[str, Sequence[Any]]]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def build_metrics(
    val_results: dict[str, float],
    test_results: dict[str, float],
    history: dict[str, list[float]],
) -> dict[str, Any]:
    return {
        "backbone": "EfficientNetV2S",
        "image_size": IMAGE_SIZE,
        "selection_metric": "val_accuracy",
        "test_role": "final_report_only",
        "val": val_results,
        "test": test_results,
        "history_epochs": {key: len(values) for key, values in history.items()},
        "deployment_artifact_changed": False,
    }


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument(
        "--train",
        action="store_true",
        help="train a new isolated candidate; never changes the deployed model",
    )
    operation.add_argument(
        "--evaluate-run",
        metavar="RUN_NAME",
        help="evaluate one existing candidate checkpoint on the test split",
    )
    parser.add_argument(
        "--run-name",
        help="new run directory name (default: a UTC timestamp)",
    )
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--phase1-epochs", type=positive_int, default=12)
    parser.add_argument("--phase2-epochs", type=positive_int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="explicitly accept slow float32 execution when no GPU is available",
    )
    return parser.parse_args(argv)


def new_run_name() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def prepare_new_run(paths: RunPaths) -> None:
    if paths.directory.exists():
        raise FileExistsError(
            f"run directory already exists: {paths.directory}; choose another --run-name"
        )
    paths.directory.mkdir(parents=True)


def run_training(args: argparse.Namespace, paths: RunPaths) -> dict[str, Any]:
    class_names = load_ordered_class_names()
    validate_dataset(class_names)
    train_ds, val_ds, test_ds = make_datasets(class_names, args.batch_size, args.seed)
    model, backbone = build_model()
    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(paths.checkpoint),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )

    write_status(paths, state="running", phase="head")
    compile_model(model, learning_rate=1e-3, weight_decay=1e-5)
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.phase1_epochs,
        callbacks=[checkpoint, epoch_status_callback(paths, "head")],
        verbose=1,
    )

    write_status(paths, state="running", phase="finetune")
    unfreeze_backbone(backbone)
    compile_model(model, learning_rate=3e-5, weight_decay=1e-5)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.phase2_epochs,
        callbacks=[
            checkpoint,
            early,
            reduce_lr,
            epoch_status_callback(paths, "finetune"),
        ],
        verbose=1,
    )

    # Validation selected the checkpoint. Test is consulted only after selection
    # and can never trigger a write to the deployed artifact.
    selected = keras.models.load_model(paths.checkpoint, compile=False)
    compile_model(selected, learning_rate=1e-5)
    val_results = evaluate(selected, val_ds, "val")
    test_results = evaluate(selected, test_ds, "test")
    history = merge_histories([history1.history, history2.history])
    write_json(paths.history, history)
    metrics = build_metrics(val_results, test_results, history)
    write_json(paths.metrics, metrics)
    return metrics


def run_evaluation(args: argparse.Namespace, paths: RunPaths) -> dict[str, Any]:
    if not paths.checkpoint.is_file():
        raise FileNotFoundError(f"candidate checkpoint not found: {paths.checkpoint}")
    class_names = load_ordered_class_names()
    validate_dataset(class_names)
    model = keras.models.load_model(paths.checkpoint, compile=False)
    compile_model(model, learning_rate=1e-5)
    results = {
        "test_role": "final_report_only",
        "test": evaluate(model, make_test_dataset(class_names, args.batch_size), "test"),
        "deployment_artifact_changed": False,
    }
    write_json(paths.evaluation, results)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.evaluate_run and args.run_name:
        raise SystemExit("--run-name is only valid with --train")

    run_name = args.run_name or (new_run_name() if args.train else args.evaluate_run)
    paths = resolve_run_paths(run_name)
    if args.train:
        prepare_new_run(paths)

    try:
        load_ml_stack()
        runtime = configure_runtime(args.allow_cpu)
        tf.keras.utils.set_random_seed(args.seed)
        write_status(paths, state="starting", phase="setup", runtime=runtime)
        metrics = run_training(args, paths) if args.train else run_evaluation(args, paths)
        write_status(paths, state="completed", metrics=metrics)
    except Exception as exc:
        if paths.directory.exists():
            write_status(paths, state="failed", error=f"{type(exc).__name__}: {exc}")
        raise

    print(f"DONE run={run_name} deployed_artifact_changed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
