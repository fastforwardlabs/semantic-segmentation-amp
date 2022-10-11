from typing import Optional, Dict

import tensorflow as tf


def dice_coeff(y_true, y_pred):
    smooth = 1.0
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


def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(
        y_true, y_pred
    )
    return loss


def tversky(y_true, y_pred, smooth=1e-6):
    # Focal Tversky loss, brought to you by:  https://github.com/nabsabraham/focal-tversky-unet
    y_true_pos = tf.keras.layers.Flatten()(y_true)
    y_pred_pos = tf.keras.layers.Flatten()(y_pred)
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (
        true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
    )


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


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
