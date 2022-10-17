import os
import json
import datetime
import argparse
import shutil

import tensorflow as tf

from src.model import unet_model
from src.model_utils import (
    CustomTensorBoard,
    dice_coeff,
    dice_loss,
    bce_dice_loss,
    tversky,
    tversky_loss,
    focal_tversky_loss,
)
from src.dataset import SegmentationDataset
from src.data_pipeline import SegmentationDataPipeline

IMG_SHAPE = (256, 1600)
BATCH_SIZE = 8
ANNOTATIONS_PATH = "data/train.csv"
TRAIN_IMG_PATH = "data/train_images/"
LOSSES = {
    "dice_loss": dice_loss,
    "bce_dice_loss": bce_dice_loss,
    "tversky_loss": tversky_loss,
    "focal_tversky_loss": focal_tversky_loss,
}


def train_model(
    n_epochs: int,
    learning_rate: float,
    n_channels_bottleneck: int,
    loss_fn: str,
    sample_weights: bool,
    log_dir: str,
):

    # instantiate dataset and pipelne
    sd = SegmentationDataset(
        label_file=ANNOTATIONS_PATH,
        img_dir_path=TRAIN_IMG_PATH,
        img_shape=IMG_SHAPE,
    )

    # create train/test & x/y splits
    train_imgs, test_imgs = sd.get_train_test_split(test_size=0.1)

    # small sample
    train_imgs = train_imgs[:200]
    test_imgs = test_imgs[:50]

    X_train = sd.get_image_sequence(train_imgs)
    y_train = sd.get_label_sequence(train_imgs, label_type="preprocessed")
    X_test = sd.get_image_sequence(test_imgs)
    y_test = sd.get_label_sequence(test_imgs, label_type="preprocessed")

    # create dataset pipelines
    sdp = SegmentationDataPipeline(
        img_shape=IMG_SHAPE,
        label_type="preprocessed",
        pipeline_options={
            "map_parallel": None,  # off if None
            "cache": False,
            "shuffle_buffer_size": False,  # off if False
            "batch_size": BATCH_SIZE,
            "prefetch": False,  # off if False
        },
    )

    if sample_weights:
        print("Weighting each sample!")
        train_sample_weights = sd.get_sample_weight_sequence(train_imgs)
        train_dataset = sdp(
            X_train, y_train, is_train=True, sample_weights=train_sample_weights
        )
    else:
        train_dataset = sdp(X_train, y_train, is_train=True)

    test_dataset = sdp(X_test, y_test, is_train=False)

    # build model
    unet = unet_model(IMG_SHAPE, n_channels_bottleneck)

    unet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=LOSSES[loss_fn],
        metrics=[dice_coeff, tversky],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(log_dir, "best_model.h5"), save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, update_freq=100, histogram_freq=1, write_images=True
        ),
        # CustomTensorBoard(
        #     log_dir=log_dir, update_freq=100, histogram_freq=1, write_images=True
        # ),
    ]

    hist = unet.fit(
        x=train_dataset,
        epochs=n_epochs,
        validation_data=test_dataset,
        callbacks=callbacks,
    )

    return hist


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--n_epochs",
        type=int,
        default=5,
        help="Number of epochs to run training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--n_channels",
        type=int,
        default=512,
        help="Number of epochs to run training.",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="dice_loss",
        choices=["dice_loss", "bce_dice_loss", "tversky_loss", "focal_tversky_loss"],
    )
    parser.add_argument(
        "--sample_weights",
        type=bool,
        default=False,
        help="Whether to apply sample weighting across classes to balance samples.",
    )

    args = parser.parse_args()

    log_dir = f'logs/unet-epochs_{args.n_epochs}-lr_{args.lr}-channels_{args.n_channels}-loss_{args.loss_fn}-sw_{args.sample_weights}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    hist = train_model(
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        n_channels_bottleneck=args.n_channels,
        loss_fn=args.loss_fn,
        sample_weights=args.sample_weights,
        log_dir=log_dir,
    )

    # save out training metrics
    with open(os.path.join(log_dir, "model_history.json"), "w") as f:
        json.dump(hist.history, f)

    # save out model architecture and train script
    train_script_path = "scripts/train.py"
    model_architecture_path = "src/model.py"
    shutil.copyfile(train_script_path, os.path.join(log_dir, "train.py"))
    shutil.copyfile(model_architecture_path, os.path.join(log_dir, "model.py"))
