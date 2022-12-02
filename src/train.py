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
import datetime

import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.model import unet_model
from src.dataset import SegmentationDataset
from src.data_pipeline import SegmentationDataPipeline


def train_model(
    n_epochs: int,
    learning_rate: float,
    n_channels_bottleneck: int,
    n_channels_out: int,
    loss_fn: str,
    test_size: float,
    sample_weights: bool,
    custom_log_dir: str,
    small_sample: bool,
    img_shape: tuple,
    batch_size: int,
    annotations_path: str,
    train_img_path: str,
    losses: dict,
    metrics: dict,
    oversample_train_set: bool,
    undersample_train_set: bool,
    sample_weight_strategy: str,
    sample_weight_ens_beta: float,
):
    """
    Main training function for Unet on semantic segmentation.

    Args:
        n_epochs (int)
        learning_rate (float)
        n_channels_bottleneck (int)
        n_channels_out (int)
        loss_fn (str)
        test_size (float)
        sample_weights (bool)
        log_dir (str)
        small_sample (bool)
        img_shape (tuple)
        batch_size (int)
        annotations_path (str)
        train_img_path (str)
        losses (dict)
        metrics (dict)

    Returns:
        tf.keras.callbacks.History

    """

    # instantiate dataset and pipelne
    sd = SegmentationDataset(
        test_size=test_size,
        label_file=annotations_path,
        img_dir_path=train_img_path,
        img_shape=img_shape,
        sample_weight_strategy=sample_weight_strategy,
        sample_weight_ens_beta=sample_weight_ens_beta,
    )

    # create train/test & x/y splits
    train_imgs = sd.train_imgs
    test_imgs = sd.test_imgs

    # apply resampling train images
    if oversample_train_set:
        print("Oversampling train set.")
        train_imgs = sd.resample_train_set(train_imgs, over_sample=True)
    elif undersample_train_set:
        print("Undersampling train set.")
        train_imgs = sd.resample_train_set(train_imgs, over_sample=False)

    # get stratified sample
    if small_sample:
        _, train_imgs = train_test_split(
            sd.imgid_to_classid_mapping[train_imgs],
            test_size=0.1,
            random_state=42,
            shuffle=True,
            stratify=sd.imgid_to_classid_mapping[train_imgs],
        )
        _, test_imgs = train_test_split(
            sd.imgid_to_classid_mapping[test_imgs],
            test_size=0.1,
            random_state=42,
            shuffle=True,
            stratify=sd.imgid_to_classid_mapping[test_imgs],
        )
        train_imgs = list(train_imgs.index)
        test_imgs = list(test_imgs.index)

    X_train = sd.get_image_sequence(train_imgs)
    y_train = sd.get_label_sequence(train_imgs, label_type="preprocessed")
    X_test = sd.get_image_sequence(test_imgs)
    y_test = sd.get_label_sequence(test_imgs, label_type="preprocessed")

    # create dataset pipelines
    sdp = SegmentationDataPipeline(
        img_shape=img_shape,
        label_type="preprocessed",
        pipeline_options={
            "map_parallel": None,  # off if None
            "cache": False,
            "shuffle_buffer_size": 25,  # off if False
            "batch_size": batch_size,
            "prefetch": False,  # off if False
        },
    )

    if sample_weights:
        print("Weighting each sample!")
        print(sd.class_weight_map)
        train_sample_weights = sd.get_sample_weight_sequence(train_imgs)
        train_dataset = sdp(
            X_train, y_train, is_train=True, sample_weights=train_sample_weights
        )
        assert len(X_train) == len(y_train) == len(train_sample_weights)
    else:
        train_dataset = sdp(X_train, y_train, is_train=True)

    test_dataset = sdp(X_test, y_test, is_train=False)

    # build model
    unet = unet_model(img_shape, n_channels_out, n_channels_bottleneck)
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses[loss_fn],
        metrics=list(metrics.values()),
    )

    if custom_log_dir is None:
        log_dir = f'logs/unet-epochs_{n_epochs}\
                    -lr_{learning_rate}\
                    -channels_{n_channels_bottleneck}\
                    -loss_{loss_fn}-sw_{sample_weights}\
                    -strategy_{sample_weight_strategy}\
                    -beta_{sample_weight_ens_beta}\
                    -small_sample_{small_sample}\
                    -{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    else:
        log_dir = custom_log_dir

    def lr_scheduler(epoch, lr, n_epochs=n_epochs):
        """
        Function for the schedule of learning rate.

        Schedule uses provided learning rate for first half of n_epochs, then
        exponentially decays for remainder of n_epochs.

        Args:
            epoch (int) - current epoch
            lr (float) - learning rate
            n_epochs (int) - total number of epochs

        Returns:
            learning rate
        """
        if epoch < (n_epochs / 2):
            return float(lr)
        else:
            return float(lr * tf.math.exp(-0.1))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                log_dir,
                "min_val_loss",
                "{epoch:02d}-{val_loss:.2f}-{val_dice_coef:.2f}-best_model.h5",
            ),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                log_dir,
                "max_val_dice",
                "{epoch:02d}-{val_loss:.2f}-{val_dice_coef:.2f}-best_model.h5",
            ),
            save_best_only=True,
            monitor="val_dice_coef",
            mode="max",
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, update_freq=100, histogram_freq=1, write_images=True
        ),
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
    ]

    hist = unet.fit(
        x=train_dataset,
        epochs=n_epochs,
        validation_data=test_dataset,
        callbacks=callbacks,
    )

    return hist, test_dataset
