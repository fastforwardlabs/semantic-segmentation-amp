import os
import json
import datetime
import argparse
import shutil

from src.train2 import train_model
from src.model_utils import (
    tversky,
    tversky_loss,
    tversky_axis,
    # tversky_loss_axis,
    TverskyLossAxis,
)


from keras import backend as K
from keras.losses import binary_crossentropy


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2.0 * K.sum(intersection) + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )
    return 1.0 - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


IMG_SHAPE = (256, 1600)
BATCH_SIZE = 8
TEST_SIZE = 0.1
ANNOTATIONS_PATH = "data/train.csv"
TRAIN_IMG_PATH = "data/train_images/"
LOSSES = {
    "tversky_loss": tversky_loss,
    # "tversky_loss_axis": tversky_loss_axis,
    "tversky_loss_axis": TverskyLossAxis(),
    "bce_dice_loss": bce_dice_loss,
}
METRICS = {"tversky": tversky, "tversky_axis": tversky_axis, "dice_coef": dice_coef}


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
        choices=[
            "bce_dice_loss",
            "tversky_loss",
            "tversky_loss_axis",
        ],
    )
    parser.add_argument(
        "--sample_weights",
        action="store_true",
        help="Whether to apply sample weighting across classes to balance samples.",
    )
    parser.add_argument(
        "--small_sample",
        action="store_true",
        help="Whether to run with small sample size for dev purposes.",
    )
    parser.add_argument(
        "--sample_weight_strategy",
        type=str,
        default="ens",
        choices=[
            "ens",
            "ip",
        ],
    )
    parser.add_argument(
        "--sample_weight_ens_beta",
        type=float,
        default=0.999,
        help="Hyperparam for ENS sample weighting.",
    )
    parser.add_argument(
        "--resample_train_set",
        action="store_true",
        help="Whether apply resampling to training set as means to balance classes",
    )

    args = parser.parse_args()
    log_dir = f'logs_sigmoid_experiment/unet-epochs_{args.n_epochs}-lr_{args.lr}-channels_{args.n_channels}-loss_{args.loss_fn}-sw_{args.sample_weights}-strategy_{args.sample_weight_strategy if args.sample_weights else "NA"}-beta_{args.sample_weight_ens_beta if args.sample_weights else "NA"}-small_sample_{args.small_sample}-resample_train_set_{args.resample_train_set}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    kwargs = {
        "n_epochs": args.n_epochs,
        "learning_rate": args.lr,
        "n_channels_bottleneck": args.n_channels,
        "loss_fn": args.loss_fn,
        "sample_weights": args.sample_weights,
        "custom_log_dir": log_dir,
        "small_sample": args.small_sample,
        "img_shape": IMG_SHAPE,
        "batch_size": BATCH_SIZE,
        "test_size": TEST_SIZE,
        "annotations_path": ANNOTATIONS_PATH,
        "train_img_path": TRAIN_IMG_PATH,
        "losses": LOSSES,
        "metrics": METRICS,
        "resample_train_set": args.resample_train_set,
        "sample_weight_strategy": args.sample_weight_strategy,
        "sample_weight_ens_beta": args.sample_weight_ens_beta,
    }
    print(kwargs)

    hist = train_model(**kwargs)

    # save out training metrics
    with open(os.path.join(log_dir, "model_history.json"), "w") as f:
        json.dump(hist.history, f)

    # save out model architecture and train script
    train_script_path = "scripts/run_train.py"
    model_architecture_path = "src/model.py"
    shutil.copyfile(train_script_path, os.path.join(log_dir, "run_train.py"))
    shutil.copyfile(model_architecture_path, os.path.join(log_dir, "model.py"))
