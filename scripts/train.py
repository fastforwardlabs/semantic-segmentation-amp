import os
import datetime

import tensorflow as tf
from tensorflow.keras import layers

from src.model import unet_model
from src.dataset import SegmentationDataset
from src.data_pipeline import SegmentationDataPipeline

IMG_SHAPE = (256, 1600)
BATCH_SIZE = 8
EPOCHS = 10
ANNOTATIONS_PATH = "data/train.csv"
TRAIN_IMG_PATH = "data/train_images/"
LOG_DIR = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# def dice_coeff(y_true, y_pred, epsilon=1e-6):
#     """
#     Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
#     Assumes the `channels_last` format.

#     Args:
#         y_true: b x X x Y x c One hot encoding of ground truth
#         y_pred: b x X x Y x c Network output, must sum to 1 over c channel (such as after softmax)
#         epsilon: Used for numerical stability to avoid divide by zero errors
        
#     """
#     axes = tuple(range(1, 3))
#     numerator = 2.0 * tf.reduce_sum((y_pred * y_true), axis=axes)
#     denominator = tf.reduce_sum(y_pred + y_true, axis=axes)

#     return tf.reduce_mean((numerator + epsilon) / (denominator + epsilon))


# def dice_loss(y_true, y_pred, epsilon=1e-6):
#     return 1 - dice_coeff(y_true, y_pred, epsilon)

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = layers.Flatten()(y_true)
    y_pred_f = layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def main():

    # instantiate dataset and pipelne
    sd = SegmentationDataset(
        label_file=ANNOTATIONS_PATH,
        img_dir_path=TRAIN_IMG_PATH,
        img_shape=IMG_SHAPE,
    )

    # create train/test & x/y splits
    train_imgs, test_imgs = sd.get_train_test_split(test_size=0.2)
    
    # small sample
    # train_imgs = train_imgs[:8]
    # test_imgs = test_imgs[:8]
    
    X_train = sd.get_image_sequence(train_imgs)
    y_train = sd.get_label_sequence(train_imgs, label_type="preprocessed")
    X_test = sd.get_image_sequence(test_imgs)
    y_test = sd.get_label_sequence(test_imgs, label_type="preprocessed")

    # create dataset pipelines
    sdp = SegmentationDataPipeline(
        img_shape=IMG_SHAPE,
        label_type="preprocessed",
        pipeline_options={
            "map_parallel": None,
            "cache": False,
            "shuffle_buffer_size": False,
            "batch_size": BATCH_SIZE,
            "prefetch": False,
        },
    )
    
    train_dataset = sdp(X_train, y_train)
    test_dataset = sdp(X_test, y_test)
    
    # build model
    unet = unet_model(IMG_SHAPE)
    
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss=dice_loss,
        metrics=[dice_coeff]
    )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(LOG_DIR, "best_model.h5"), save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
    ]
    

    hist = unet.fit(
        train_dataset, epochs=EPOCHS, validation_data=test_dataset, callbacks=callbacks
    )


if __name__ == "__main__":
    main()
    