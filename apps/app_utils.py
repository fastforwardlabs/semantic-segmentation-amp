from collections import defaultdict

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.model import unet_model
from src.dataset import SegmentationDataset
from src.data_pipeline import SegmentationDataPipeline
from src.data_utils import create_mask
from src.model_utils import (
    dice_coef,
    dice_loss,
    tversky,
    tversky_loss,
    tversky_axis,
    TverskyLossAxis,
)

BATCH_SIZE = 1
IMG_SHAPE = (256, 1600)
ANNOTATIONS_PATH = "data/train.csv"
TRAIN_IMG_PATH = "data/train_images/"
MODEL_PATH = "model/best_model.h5"
LOSSES = {
    "dice_loss": dice_loss,
    "tversky_loss": tversky_loss,
    "tversky_loss_axis": TverskyLossAxis(),
}
METRICS = {"dice_coef": dice_coef, "tversky": tversky, "tversky_axis": tversky_axis}
CLASS_MAP = {"Scratches": 3, "Patches": 4, "Both": -2}


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
            class_idx=class_idx, dataset=dataset, n_samples=n_samples
        )
        samples_dict[class_idx] = x, y

    return samples_dict


def prepare_visual_assets(x, y, pipeline, model):

    ds = pipeline([x], [y], is_train=False)
    sample = list(ds.take(1).as_numpy_iterator())[0]

    x = sample[0]
    y_true = sample[1]
    y_pred = model.predict(x)

    return x, y_true, y_pred


def show_masks():

    # toggle show masks
    st.session_state["show_masks"] = not st.session_state["show_masks"]


def increment_index_tracker(defect_type):
    """
    Utility to keep track of the current index for each of the defect classes
    as the user cylces through them.

    Provided a defect_type, this function returns the index of the next example
    for that given defect_type and updates the session state counter.
    """

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
    st.session_state["show_masks"] = not st.session_state["show_masks"]


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
