from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyMLCOPredictor:
    """Exact NumPy forward pass for the released five-layer ML-CO MLP.

    TensorFlow is needed only once to export the Keras weights. Keeping the
    benchmark process on NumPy avoids reserving a GPU and avoids introducing
    TensorFlow's pinned NumPy/protobuf versions into the existing environment.
    """

    EXPECTED_KERNEL_SHAPES = (
        (20, 10),
        (10, 10),
        (10, 10),
        (10, 10),
        (10, 1),
    )
    ALLOWED_INPUT_DIMENSIONS = (11, 20)

    @classmethod
    def expected_kernel_shapes(cls, input_dimension: int) -> tuple[tuple[int, int], ...]:
        if input_dimension not in cls.ALLOWED_INPUT_DIMENSIONS:
            raise ValueError(
                f"Unsupported ML-CO input dimension {input_dimension}; expected one of "
                f"{cls.ALLOWED_INPUT_DIMENSIONS}"
            )
        return ((input_dimension, 10), *cls.EXPECTED_KERNEL_SHAPES[1:])

    def __init__(self, weights_path: str | Path) -> None:
        self.weights_path = Path(weights_path).expanduser().resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Exported ML-CO weights not found: {self.weights_path}"
            )

        with np.load(self.weights_path, allow_pickle=False) as archive:
            num_layers = int(np.asarray(archive["num_layers"]).item())
            if num_layers != len(self.EXPECTED_KERNEL_SHAPES):
                raise ValueError(
                    f"Expected {len(self.EXPECTED_KERNEL_SHAPES)} dense layers, "
                    f"found {num_layers}"
                )
            self.kernels = tuple(
                np.asarray(archive[f"kernel_{idx}"], dtype=np.float32)
                for idx in range(num_layers)
            )
            self.biases = tuple(
                np.asarray(archive[f"bias_{idx}"], dtype=np.float32)
                for idx in range(num_layers)
            )
            source_hash = np.asarray(
                archive.get("source_saved_model_sha256", np.asarray("unknown"))
            ).item()
            self.source_saved_model_sha256 = str(source_hash)

        actual_shapes = tuple(kernel.shape for kernel in self.kernels)
        if not actual_shapes or len(actual_shapes[0]) != 2:
            raise ValueError(f"Invalid ML-CO kernels: {actual_shapes}")
        self.input_dimension = int(actual_shapes[0][0])
        expected_kernel_shapes = self.expected_kernel_shapes(self.input_dimension)
        if actual_shapes != expected_kernel_shapes:
            raise ValueError(
                "Unexpected ML-CO kernel shapes: "
                f"expected {expected_kernel_shapes}, found {actual_shapes}"
            )
        expected_bias_shapes = tuple((shape[1],) for shape in expected_kernel_shapes)
        actual_bias_shapes = tuple(bias.shape for bias in self.biases)
        if actual_bias_shapes != expected_bias_shapes:
            raise ValueError(
                "Unexpected ML-CO bias shapes: "
                f"expected {expected_bias_shapes}, found {actual_bias_shapes}"
            )

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_dimension:
            raise ValueError(
                f"Expected feature matrix (n, {self.input_dimension}), found {values.shape}"
            )
        for idx, (kernel, bias) in enumerate(zip(self.kernels, self.biases)):
            values = values @ kernel + bias
            if idx < len(self.kernels) - 1:
                values = np.maximum(values, np.float32(0.0))
        return values
