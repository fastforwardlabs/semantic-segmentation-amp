from collections import defaultdict
from typing import Optional, Dict, List

from tqdm import tqdm
import numpy as np
import tensorflow as tf


def dice_coeff(y_true, y_pred, smooth=1e-6):

    # remove background channel from loss calculation
    y_true = y_true[:, :, :, 1:]
    y_pred = y_pred[:, :, :, 1:]

    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return score


def dice_loss(y_true, y_pred):
    # Dice similarity coefficient loss, brought to you by: https://github.com/nabsabraham/focal-tversky-unet
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def dice_coeff_old(y_true, y_pred, epsilon=1e-6):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    Args:
        y_true: b x X x Y x c One hot encoding of ground truth
        y_pred: b x X x Y x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors
    """
    # remove background channel from loss calculation
    y_true = y_true[:, :, :, 1:]
    y_pred = y_pred[:, :, :, 1:]

    axes = tuple(range(1, 3))
    numerator = 2.0 * tf.reduce_sum((y_pred * y_true), axis=axes)
    # denominator = tf.reduce_sum(y_pred + y_true, axis=axes)
    denominator = tf.reduce_sum(y_pred, axis=axes) + tf.reduce_sum(y_true, axis=axes)

    # return tf.reduce_mean((numerator + epsilon) / (denominator + epsilon))
    return tf.reduce_mean(numerator + epsilon, axis=-1) / tf.reduce_mean(
        denominator + epsilon, axis=-1
    )  # CORRECTED_AXIS
    # return tf.reduce_mean(numerator + epsilon) / tf.reduce_mean(
    #     denominator + epsilon
    # )  # CORRECTED


def dice_loss_old(y_true, y_pred):
    # Dice similarity coefficient loss, brought to you by: https://github.com/nabsabraham/focal-tversky-unet
    loss = 1 - dice_coeff_old(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(
        y_true, y_pred
    )
    return loss


def tversky_axis(y_true, y_pred, smooth=1e-6):
    # Focal Tversky loss, brought to you by:  https://github.com/nabsabraham/focal-tversky-unet

    # remove background channel from loss calculation
    y_true = y_true[:, :, :, 1:]
    y_pred = y_pred[:, :, :, 1:]

    axes = tuple(range(1, 4))  # compute over H x W x C, preserve batch dimension

    true_pos = tf.reduce_sum(y_true * y_pred, axis=axes)
    false_neg = tf.reduce_sum(y_true * (1 - y_pred), axis=axes)
    false_pos = tf.reduce_sum((1 - y_true) * y_pred, axis=axes)
    alpha = 0.7

    num = true_pos + smooth
    den = true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth

    return num / den


def tversky_loss_axis(y_true, y_pred):
    return 1 - tversky_axis(y_true, y_pred)


def tversky(y_true, y_pred, smooth=1e-6):
    # Focal Tversky loss, brought to you by:  https://github.com/nabsabraham/focal-tversky-unet

    # remove background channel from loss calculation
    y_true = y_true[:, :, :, 1:]
    y_pred = y_pred[:, :, :, 1:]

    y_true_pos = tf.keras.layers.Flatten()(y_true)
    y_pred_pos = tf.keras.layers.Flatten()(y_pred)
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7

    num = true_pos + smooth
    den = true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth

    return num / den


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


# def tversky(y_true, y_pred, smooth=1e-6):
#     # Focal Tversky loss, brought to you by:  https://github.com/nabsabraham/focal-tversky-unet

#     # remove background channel from loss calculation
#     y_true = y_true[:, :, :, 1:]
#     y_pred = y_pred[:, :, :, 1:]

#     y_true_pos = tf.keras.layers.Flatten()(y_true)
#     y_pred_pos = tf.keras.layers.Flatten()(y_pred)
#     true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
#     false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
#     false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
#     alpha = 0.7
#     return (true_pos + smooth) / (
#         true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
#     )


# def tversky_loss(y_true, y_pred):
#     return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return tf.keras.backend.pow((1 - pt_1), gamma)


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):  # type: ignore
    """TensorBoard callback with update_freq=N functionality."""

    def _implements_train_batch_hooks(self) -> bool:
        return super()._implements_train_batch_hooks() or isinstance(
            self.update_freq, int
        )

    def on_train_batch_end(
        self,
        batch: int,
        logs: Optional[Dict[str, float]] = None,
    ) -> None:
        super().on_train_batch_end(batch, logs)
        if batch % self.update_freq == 0 and logs is not None:
            with tf.summary.record_if(True), self._train_writer.as_default():
                for name, value in logs.items():
                    tf.summary.scalar("batch_" + name, value, step=self._train_step)


def dice_coeff_per_class(y_true, y_pred, smooth=1e-6):
    """
    Calculates per class dice similarity coefficient.

    Args:
        y_true (np.ndarray) - of shape (b, h, w, c)
        y_pred (np.ndarray) - of shape (b, h, w, c)

    Returns:
        metrics (dict) - dict of metrics where keys map to channels of input matrix


    """

    metrics = {}

    for class_idx in range(y_true.shape[-1]):

        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]

        y_true_pos = tf.keras.layers.Flatten()(y_true_class)
        y_pred_pos = tf.keras.layers.Flatten()(y_pred_class)

        true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
        false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
        false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)

        score = (2.0 * true_pos + smooth) / (
            2.0 * true_pos + false_pos + false_neg + smooth
        )

        metrics[class_idx] = score.numpy()

    return metrics


# def evaluate_per_class_dice(dataset: tf.data.Dataset, model) -> Dict:
#     """
#     Evaluates a model on a given dataset to calculate _per class_ dice similarity coefficient.

#     Args:
#         dataset - tf dataset upon which to run evaluation
#         model - trained UNet model to use for inference

#     Returns:
#         class_scores

#     """

#     class_scores = defaultdict(list)

#     for x, y_true in tqdm(dataset):
#         y_pred = model.predict(x)

#         batch_score = dice_coeff_per_class(y_true, y_pred)

#         for k, v in batch_score.items():
#             class_scores[k].append(v)

#     return {k: np.mean(v) for k, v in class_scores.items()}


def evaluate_per_class(dataset: tf.data.Dataset, model, metric) -> Dict:
    """
    Evaluates a model on a given dataset to calculate _per class_ dice similarity coefficient.

    Args:
        dataset - tf dataset upon which to run evaluation
        model - trained UNet model to use for inference

    Returns:
        class_scores

    """

    class_scores = defaultdict(list)

    for x, y_true in tqdm(dataset):
        y_pred = model.predict(x)

        batch_score = metric(y_true, y_pred)

        for k, v in batch_score.items():
            class_scores[k].append(v)

    return {k: np.mean(v) for k, v in class_scores.items()}


def tversky_per_class(y_true, y_pred, smooth=1e-6):

    metrics = {}

    for class_idx in range(y_true.shape[-1]):

        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]

        y_true_pos = tf.keras.layers.Flatten()(y_true_class)
        y_pred_pos = tf.keras.layers.Flatten()(y_pred_class)

        true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
        false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
        false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7

        num = true_pos + smooth
        den = true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth

        score = num / den

        metrics[class_idx] = score.numpy()
        # metrics[class_idx] = num.numpy(), den.numpy()

    return metrics
