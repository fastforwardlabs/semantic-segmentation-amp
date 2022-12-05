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
from collections import defaultdict

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.dataset import SegmentationDataset
from src.data_pipeline import SegmentationDataPipeline
from src.data_utils import create_mask, get_dataset_path
from src.model_utils import (
    dice_coef,
)

BATCH_SIZE = 1
IMG_SHAPE = (256, 1600)
DATASET_DIR = get_dataset_path()
ANNOTATIONS_PATH = os.path.join(DATASET_DIR, "train.csv")
TRAIN_IMG_PATH = os.path.join(DATASET_DIR, "train_images")
MODEL_PATH = "model/best_model.h5"
LOSSES = {
    "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy(),
}
METRICS = {"dice_coef": dice_coef}
CLASS_MAP = {"Scratches": 3, "Patches": 4, "Scratches & Patches": -2}
USE_TEST_SET = True if DATASET_DIR == "/home/cdsw/data" else False


@st.cache
def get_dataset():
    """Construct dataset object"""
    return SegmentationDataset(
        label_file=ANNOTATIONS_PATH,
        img_dir_path=TRAIN_IMG_PATH,
        img_shape=IMG_SHAPE,
        drop_classes=True,
        test_size=0.1,
    )


@st.cache
def get_data_pipeline():
    """Construct data pipeline object"""
    return SegmentationDataPipeline(
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


def get_model():
    """Return trained model"""
    return tf.keras.models.load_model(MODEL_PATH, custom_objects=(LOSSES | METRICS))


def get_samples_by_class(class_idx, dataset, n_samples, test_set=True):
    """
    Get a batch of samples of particular class type.

    Args:
        class_idx (int)
        dataset (src.dataset.SegmentationDataset)
        n_samples (int)
        test_set (bool)

    Returns:
        X (List[str])
        y (List[str])

    """

    set_filter = dataset.test_imgs if test_set else dataset.train_imgs

    class_imgs = dataset.imgid_to_classid_mapping[set_filter][
        dataset.imgid_to_classid_mapping == class_idx
    ].index.tolist()

    X = dataset.get_image_sequence(class_imgs)
    y = dataset.get_label_sequence(class_imgs, label_type="preprocessed")

    return X[:n_samples], y[:n_samples]


def build_samples_queue(dataset, n_samples=100):
    """
    Collects n_samples for each class_idx to serve as a queue
    from which the UI will pull from.

    Args:
        dataset
        n_samples (int)

    Returns:
        Dict[Tuple[List]] - each value in the dict is a tuple of two lists (of paths) where common
        indicies indicates an image-to-annotation pair.
    """

    samples_dict = defaultdict(tuple)

    for class_idx in CLASS_MAP.values():
        x, y = get_samples_by_class(
            class_idx=class_idx,
            dataset=dataset,
            n_samples=n_samples,
            test_set=USE_TEST_SET,
        )
        samples_dict[class_idx] = x, y

    return samples_dict


def prepare_visual_assets(x, y, pipeline, model):
    """
    Uses the provided model, an image example (x), aground truth (y), and a
    data pipeline class to generate a mask prediction.
    """

    ds = pipeline([x], [y], is_train=False)
    sample = list(ds.take(1).as_numpy_iterator())[0]

    x = sample[0]
    y_true = sample[1]
    y_pred = model.predict(x)

    return x, y_true, y_pred


def toggle_show_masks():
    st.session_state["show_masks"] = not st.session_state["show_masks"]


def increment_index_tracker(defect_type):
    """
    Utility to keep track of the current index for each of the defect classes
    as the user cylces through them.

    Provided a defect_type, this function sets the index of the next example
    for that given defect_type and updates the session state counter.

    If we've reached the last sample in the class, restart at the beginning
    """

    current_idx = st.session_state["class_index_tracker"][CLASS_MAP[defect_type]]
    num_samples = len(st.session_state["samples_dict"][3][0])

    if current_idx == (num_samples - 1):
        st.session_state["class_index_tracker"][CLASS_MAP[defect_type]] = 0
    else:
        st.session_state["class_index_tracker"][CLASS_MAP[defect_type]] += 1


def get_next_example():
    """
    Generate the next example to visualize.

    This function is called when requesting another image based on user request
    or when defect type selector is altered. The function uses the `class_index_tracker`
    to identify the next example for the given defect class, and sets this example as the
    `current_x` and `current_y` held in session state.
    """

    defect_type = st.session_state["defect_type"]

    # increment defect index tracker
    increment_index_tracker(defect_type)
    next_defect_type_index = st.session_state["class_index_tracker"][
        CLASS_MAP[defect_type]
    ]

    # get and set next example as session state
    next_x = st.session_state.samples_dict[CLASS_MAP[defect_type]][0][
        next_defect_type_index
    ]
    next_y = st.session_state.samples_dict[CLASS_MAP[defect_type]][1][
        next_defect_type_index
    ]
    st.session_state["current_x"] = next_x
    st.session_state["current_y"] = next_y

    # toggle show masks
    if st.session_state["show_masks"]:
        toggle_show_masks()
    # st.session_state["show_masks"] = not st.session_state["show_masks"]


def save_channel_imgs(y, type="pred"):
    """
    Generates and saves segmentation masks for each channel in the given
    set of masks.

    Args:
        y (np.ndarray) - of shape 1xhxw,c
        type (str) = "pred" or "true"
    """

    y = create_mask(y, n_channels=3)

    for i in range(3):
        plt.imsave(
            f"apps/tmp/{type}/channel_{i}.png",
            y[0, :, :, i],
            vmin=0.0,
            vmax=1.0,
            cmap="cividis",
        )


def save_mask_overlay_image(x, y, shape=(256, 1600), n_channels=3, type="pred"):
    """
    Generates and saves an image with segmentation masks overlayed.

    Args:
        x (np.ndarray) - of shape 1xhxwxc
        y (np.ndarray) - of shape 1xhxwxc
        type (str) = "pred" or "true"
    """

    mask = create_mask(y, n_channels=n_channels)

    handles = []
    colors = [
        (0, 1, 0, 0.25),
        (0, 0, 1, 0.25),
    ]
    labels = ["Scratches", "Patches"]
    plt.figure(figsize=(16, 12))
    plt.axis("off")
    plt.imshow(x[0])

    # NOTE - we omit the first channel since its background
    for channel_idx in range(1, n_channels):

        emask = np.zeros(shape=shape + (4,))
        channel_mask = mask[0, :, :, channel_idx].astype(int)

        # get flat indicies of 1's from the channel mask
        active_flat_indicies = np.nonzero(channel_mask.ravel())

        # get coordinate arrays of flat indicies
        xs, ys = np.unravel_index(active_flat_indicies, shape=(256, 1600), order="C")

        emask[xs, ys, :] = colors[channel_idx - 1]

        plt.imshow(emask)
        handles.append(
            mpatches.Patch(
                color=colors[channel_idx - 1], label=f"{labels[channel_idx-1]}"
            )
        )

    plt.legend(handles=handles)
    plt.savefig(f"apps/tmp/{type}/overlay.png", bbox_inches="tight", pad_inches=0)
