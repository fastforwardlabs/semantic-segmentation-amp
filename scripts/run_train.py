# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2022
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import os
import json
import argparse
import shutil
import tensorflow as tf

from src.train import train_model
from src.model_utils import (
    dice_coef,
    evaluate_per_class,
    dice_coef_per_class,
)


IMG_SHAPE = (256, 1600)
BATCH_SIZE = 8
TEST_SIZE = 0.1
N_CHANNELS_OUT = 3
ANNOTATIONS_PATH = "data/train.csv"
TRAIN_IMG_PATH = "data/train_images/"
LOSSES = {
    "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy(),
}
METRICS = {"dice_coef": dice_coef}


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
            "categorical_crossentropy",
            "dice_loss",
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
        default="NA",
        choices=[
            "NA",
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
        "--oversample_train_set",
        action="store_true",
        help="Whether apply oversampling to training set as means to balance classes",
    )
    parser.add_argument(
        "--undersample_train_set",
        action="store_true",
        help="Whether apply undersampling to training set as means to balance classes",
    )

    args = parser.parse_args()

    log_dir = f'logs/unet-epochs_{args.n_epochs}-lr_{args.lr}-channels_{args.n_channels}-loss_{args.loss_fn}-small_sample_{args.small_sample}{"-sw_"+str(args.sample_weights) if args.sample_weights else ""}{"-strategy_"+(args.sample_weight_strategy) if args.sample_weights else ""}{"-beta_"+str(args.sample_weight_ens_beta) if args.sample_weights else ""}{"-oversample_train_set_"+str(args.oversample_train_set) if args.oversample_train_set else ""}{"-undersample_train_set_"+str(args.undersample_train_set) if args.undersample_train_set else ""}'

    kwargs = {
        "n_epochs": args.n_epochs,
        "learning_rate": args.lr,
        "n_channels_bottleneck": args.n_channels,
        "n_channels_out": N_CHANNELS_OUT,
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
        "oversample_train_set": args.oversample_train_set,
        "undersample_train_set": args.undersample_train_set,
        "sample_weight_strategy": args.sample_weight_strategy,
        "sample_weight_ens_beta": args.sample_weight_ens_beta,
    }
    print(kwargs)

    hist, test_dataset = train_model(**kwargs)
    hist.history["lr"] = [float(val) for val in hist.history["lr"]]

    # save out training metrics
    with open(os.path.join(log_dir, "model_history.json"), "w") as f:
        json.dump(hist.history, f)

    # save out model architecture and train script
    train_script_path = "scripts/run_train.py"
    model_architecture_path = "src/model.py"
    shutil.copyfile(train_script_path, os.path.join(log_dir, "run_train.py"))
    shutil.copyfile(model_architecture_path, os.path.join(log_dir, "model.py"))

    # evaluate per class score on test set
    model_dir = os.path.join(log_dir, "max_val_dice")
    model_name = os.listdir(model_dir)[-1]
    MODEL_PATH = os.path.join(model_dir, model_name)
    unet_model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects=(LOSSES | METRICS)
    )
    dice_per_class = evaluate_per_class(test_dataset, unet_model, dice_coef_per_class)

    with open(os.path.join(log_dir, "avg_dice_coeff_per_class.json"), "w") as f:
        json.dump(dice_per_class, f)
