#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np


EXPECTED_ACTIVATIONS = ("relu", "relu", "relu", "relu", "linear")
EXPECTED_KERNEL_SHAPES = (
    (20, 10),
    (10, 10),
    (10, 10),
    (10, 10),
    (10, 1),
)
ALLOWED_INPUT_DIMENSIONS = (11, 20)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    package_dir = script_dir
    parser = argparse.ArgumentParser(
        description="Export the released ML-CO Keras model for NumPy-only inference."
    )
    parser.add_argument(
        "--saved-model",
        type=Path,
        default=package_dir / "assets" / "mlco_official_saved_model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=package_dir / "assets" / "mlco_official_weights.npz",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    model = tf.keras.models.load_model(str(args.saved_model), compile=False)
    layers = [layer for layer in model.layers if layer.get_weights()]
    activations = tuple(layer.get_config().get("activation") for layer in layers)
    kernel_shapes = tuple(layer.get_weights()[0].shape for layer in layers)
    if activations != EXPECTED_ACTIVATIONS:
        raise RuntimeError(
            f"Unexpected activations: expected {EXPECTED_ACTIVATIONS}, found {activations}"
        )
    input_dimension = int(kernel_shapes[0][0]) if kernel_shapes else -1
    expected_kernel_shapes = (
        ((input_dimension, 10), *EXPECTED_KERNEL_SHAPES[1:])
        if input_dimension in ALLOWED_INPUT_DIMENSIONS
        else None
    )
    if expected_kernel_shapes is None or kernel_shapes != expected_kernel_shapes:
        raise RuntimeError(
            "Unexpected kernels: expected the official five-layer MLP with input "
            f"dimension in {ALLOWED_INPUT_DIMENSIONS}, found {kernel_shapes}"
        )

    payload: dict[str, np.ndarray] = {
        "num_layers": np.asarray(len(layers), dtype=np.int64),
        "input_dimension": np.asarray(input_dimension, dtype=np.int64),
        "source_saved_model_sha256": np.asarray(
            sha256(args.saved_model / "saved_model.pb")
        ),
    }
    for idx, layer in enumerate(layers):
        weights = layer.get_weights()
        kernel = np.asarray(weights[0], dtype=np.float32)
        bias = (
            np.asarray(weights[1], dtype=np.float32)
            if len(weights) == 2
            else np.zeros(kernel.shape[1], dtype=np.float32)
        )
        payload[f"kernel_{idx}"] = kernel
        payload[f"bias_{idx}"] = bias

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    rng = np.random.default_rng(20260816)
    features = rng.normal(size=(64, input_dimension)).astype(np.float32)
    expected = np.asarray(model(features, training=False), dtype=np.float32)
    actual = features
    for idx in range(len(layers)):
        actual = actual @ payload[f"kernel_{idx}"] + payload[f"bias_{idx}"]
        if idx < len(layers) - 1:
            actual = np.maximum(actual, np.float32(0.0))
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1e-4:
        args.output.unlink(missing_ok=True)
        raise RuntimeError(f"NumPy export verification failed: max error {max_error}")
    print(f"Exported {args.output}")
    print(f"SavedModel SHA256: {payload['source_saved_model_sha256'].item()}")
    print(f"TensorFlow/NumPy max absolute error: {max_error:.9g}")


if __name__ == "__main__":
    main()
