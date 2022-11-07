import os
import datetime

import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.model2 import unet_model
from src.dataset import SegmentationDataset
from src.data_pipeline import SegmentationDataPipeline


def train_model(
    n_epochs: int,
    learning_rate: float,
    n_channels_bottleneck: int,
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
    resample_train_set: bool,
    sample_weight_strategy: str,
    sample_weight_ens_beta: float,
):
    """
    Main training function for Unet on semantic segmentation.

    Args:
        n_epochs (int)
        learning_rate (float)
        n_channels_bottleneck (int)
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
    if resample_train_set:
        print("Oversampling train set.")
        train_imgs = sd.oversample_train_set(train_imgs)

    # get stratified sample
    if small_sample:
        _, train_imgs = train_test_split(
            sd.imgid_to_classid_mapping[train_imgs],
            test_size=0.25,
            random_state=42,
            shuffle=True,
            stratify=sd.imgid_to_classid_mapping[train_imgs],
        )
        _, test_imgs = train_test_split(
            sd.imgid_to_classid_mapping[test_imgs],
            test_size=0.25,
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
            "shuffle_buffer_size": False,  # off if False
            "batch_size": batch_size,
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
    unet = unet_model(img_shape, n_channels_bottleneck)

    unet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses[loss_fn],
        metrics=list(metrics.values()),
    )

    if custom_log_dir is None:
        log_dir = f'logs/unet-epochs_{n_epochs}-lr_{learning_rate}-channels_{n_channels_bottleneck}-loss_{loss_fn}-sw_{sample_weights}-strategy_{sample_weight_strategy}-beta_{sample_weight_ens_beta}-small_sample_{small_sample}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    else:
        log_dir = custom_log_dir

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(log_dir, "best_model.h5"), save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, update_freq=100, histogram_freq=1, write_images=True
        ),
    ]

    hist = unet.fit(
        x=train_dataset,
        epochs=n_epochs,
        validation_data=test_dataset,
        callbacks=callbacks,
    )

    return hist
    # return unet, n_epochs, callbacks
