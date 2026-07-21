"""Keras ANN for fraud classification.

Architecture (default config)::

    Input(30)
      -> Dense(64, relu) -> BatchNorm -> Dropout(0.30)
      -> Dense(32, relu) -> BatchNorm -> Dropout(0.30)
      -> Dense(16, relu) -> BatchNorm -> Dropout(0.30)
      -> Dense(1, sigmoid)

The network is deliberately shallow: the input features are already PCA
components, so a wide-then-narrow funnel with dropout + L2 generalizes better
than a deep stack. The loss is binary cross-entropy; residual imbalance after
SMOTE is handled with class weights.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.optimizers import Adam

from ..config import CONFIG


def build_ann(input_dim: int) -> Model:
    """Construct and compile the ANN from ``config.yaml`` hyperparameters."""
    ann = CONFIG.model.ann

    inputs = layers.Input(shape=(input_dim,), name="transaction_features")
    x = inputs
    for i, units in enumerate(ann.hidden_units):
        x = layers.Dense(
            units,
            activation=ann.activation,
            kernel_regularizer=regularizers.l2(ann.l2),
            name=f"dense_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(ann.dropout, name=f"dropout_{i}")(x)

    outputs = layers.Dense(
        1, activation=ann.output_activation, name="fraud_probability"
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="ann_fraud_detector")
    model.compile(
        optimizer=Adam(learning_rate=ann.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="auprc", curve="PR"),
        ],
    )
    return model


def default_callbacks(monitor: str = "val_auprc") -> list:
    """EarlyStopping + LR reduction, tuned to maximize PR-AUC on validation."""
    ann = CONFIG.model.ann
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=ann.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="max",
            factor=0.5,
            patience=max(2, ann.early_stopping_patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
    ]
