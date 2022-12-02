import os
import json
from collections import defaultdict
from typing import Dict

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf

# METRICS & LOSSES


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice similarity coefficient

    Adapted from: https://github.com/nabsabraham/focal-tversky-unet

    """

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
    """
    Dice similarity coefficient loss

    """
    loss = 1 - dice_coef(y_true, y_pred)
    return loss


# UTILITIES


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


def evaluate_per_class(dataset: tf.data.Dataset, model, metric) -> Dict:
    """
    Evaluates a model on a given dataset to calculate _per class_ dice similarity coefficient.

    Args:
        dataset - tf dataset upon which to run evaluation
        model - trained UNet model to use for inference
        metric - either `tversky_per_class` or `dice_coeff_per_class`

    Returns:
        dict - average score per class

    """

    class_scores = defaultdict(list)

    for x, y_true in tqdm(dataset):
        y_pred = model.predict(x)

        batch_score = metric(y_true, y_pred)

        for k, v in batch_score.items():
            class_scores[k].append(v)

    return {k: float(np.mean(v)) for k, v in class_scores.items()}
    # return class_scores


def tversky_per_class(y_true, y_pred, smooth=1e-6):
    """
    Calculates the tversky score per class for a given batch of
    ground truths and predictions.

    Excludes background channel from calculations.

    Args:
        y_true (np.ndarray) - of shape (b, h, w, c)
        y_pred (np.ndarray) - of shape (b, h, w, c)

    Returns:
        metrics (dict)
    """

    metrics = {}

    for class_idx in range(1, y_true.shape[-1]):

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

    return metrics


def dice_coef_per_class(y_true, y_pred, smooth=1e-6):
    """
    Calculates the dice coefficient score per class for a given batch of
    ground truths and predictions.

    Excludes background channel from calculations.

    Args:
        y_true (np.ndarray) - of shape (b, h, w, c)
        y_pred (np.ndarray) - of shape (b, h, w, c)

    Returns:
        metrics (dict)
    """

    metrics = {}

    for class_idx in range(1, y_true.shape[-1]):

        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]

        y_true_f = tf.keras.layers.Flatten()(y_true_class)
        y_pred_f = tf.keras.layers.Flatten()(y_pred_class)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2.0 * intersection + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
        )

        metrics[class_idx] = score.numpy()

    return metrics


def collect_experiment_scores(log_dir):
    """
    Gather validation dice_coefficient score and loss for each training
    experiment in the provided log_dir, format as a pd.DataFrame, and return.

    For details on training experiments, see `/scripts/train_experiment.py`.

    Args:
        log_dir (str)

    Returns:
        pd.DataFrame

    """

    val_dice_score, val_loss = [], []
    experiments = os.listdir(log_dir)

    for exp in experiments:

        # gather training history
        with open(os.path.join(log_dir, exp, "model_history.json"), "r") as f:
            hist = json.load(f)

        # determine best dice score
        best_dice_epoch = np.argmax(hist["val_dice_coef"])
        best_dice = hist["val_dice_coef"][best_dice_epoch]

        # determine best loss
        best_loss_epoch = np.argmin(hist["val_loss"])
        best_loss = hist["val_loss"][best_loss_epoch]

        val_dice_score.append(best_dice)
        val_loss.append(best_loss)

    return pd.DataFrame(
        zip(experiments, val_dice_score, val_loss),
        columns=["experiment", "val_dice_score", "val_loss"],
    )
