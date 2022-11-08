import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.utils.class_weight import compute_class_weight


def rle2mask(rle_string, img_size=(256, 1600), fill_color=(1, 0, 0, 0.25)):
    """
    Converts run-length encoded segmentation masks into RGBA image.

    Args:
        rle_string (str) - Run Length Encoding of segmentation mask
        img_size (tuple) - Image size of output mask overlay
        fill_color - RGBA color for segmentation mask pixels

    Returns:
        mask (np.ndarray) - segmentation mask of shape (img_size[0], img_size[1], 4)

    """
    mask = np.zeros(shape=img_size + (4,))
    nums = [int(x) for x in rle_string.split()]
    N = len(nums)
    for i in range(N // 2):
        n, length = nums[2 * i : 2 * i + 2]
        ns = [n - 1 + i for i in range(length)]
        ys, xs = np.unravel_index(ns, shape=img_size, order="F")
        mask[ys, xs, :] = fill_color
    return mask


def display_img_with_mask(df, image_id, display_classes=[]):
    """
    Displays an image with overlayed segmentation mask.

    Args:
        df (pd.DataFrame) - train.csv with RLE annotations as df
        display_classes (list) - which classes to include in image masks
        image_id (str)
    """
    display_classes = display_classes or [1, 2, 3, 4]
    fname = f"../data/train_images/{image_id}"
    img = plt.imread(fname)

    plt.figure(figsize=(16, 12))
    plt.imshow(img)

    colors = [
        (1, 0, 0, 0.25),
        (0, 1, 0, 0.25),
        (0, 0, 1, 0.25),
    ]

    df_mask = df.ImageId == image_id
    df_mask &= df.ClassId.isin(display_classes)
    handles, labels = [], []
    for color, row in zip(colors, df[df_mask].sort_values(by="ClassId").itertuples()):
        mask = rle2mask(row.EncodedPixels, fill_color=color)
        plt.imshow(mask)
        handles.append(mpatches.Patch(color=color, label=f"Defect {row.ClassId}"))
    plt.legend(handles=handles)


def plot_sample_batch(x, y_true, y_pred, softmax_output=True):
    """
    Plots a sample batch of data (images and masks) from a
    Tensorflow Dataset.

    Args:
        x - image batch of shape (b, h, w, 3)
        y_true - binary ground truth mask batch of shape (b, h, w, 8)
        y_pred - unet prediction batch of shape (b, h, w, 8)

    """
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("Sample Batch Visualized")

    x_batch = x
    y_batch = y_true
    n_channels = 4

    if softmax_output:
        y_pred = create_mask(y_pred)
        n_channels = 5

    batch_size = x_batch.shape[0]
    n_cols = 2
    n_rows = int(batch_size / n_cols)

    outer_grid = gridspec.GridSpec(n_rows, n_cols, wspace=0.1, hspace=0.2)

    for i in range(batch_size):
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            n_channels + 1, 2, subplot_spec=outer_grid[i], wspace=0.01, hspace=0.01
        )

        for j in range(n_channels + 1):
            if j == 0:
                ax = plt.Subplot(fig, inner_grid[j, 0])
                ax.imshow(x_batch[i], vmin=0.0, vmax=1.0)

                ax_true = plt.Subplot(fig, inner_grid[j, 1])
                ax_true.imshow(x_batch[i], vmin=0.0, vmax=1.0)

            else:
                ax = plt.Subplot(fig, inner_grid[j, 0])
                ax.imshow(y_batch[i][..., j - 1], vmin=0.0, vmax=1.0, cmap="cividis")

                ax_true = plt.Subplot(fig, inner_grid[j, 1])
                ax_true.imshow(
                    y_pred[i][..., j - 1], vmin=0.0, vmax=1.0, cmap="cividis"
                )

            if j == 0:
                ax.set_title(f"Ex {i}: Ground Truth Mask")
                ax_true.set_title(f"Ex {i}: Predicted Mask")

            ax.axis("off")
            ax_true.axis("off")

            fig.add_subplot(ax)
            fig.add_subplot(ax_true)

    plt.show()


def prepare_mask_label(label_element, img_height=256, img_width=1600, one_hot=True):
    """
    Prepares image annotation labels as matrix of binary mask channels.

    Initializes empty matrix of size n_classes (as inferred by shape of label_element).
    Converts each RLE label to binary mask, and inserts each of those masks in the appropriate matrix channel.

    Args:
        label_element (tf.Tensor: shape (2,n_classes), dtype=string)
        mask_height (int)
        mask_width (int)

    Returns:
        tf.Tensor (float64)
    """

    n_classes = len(label_element[1])
    mask = np.zeros((img_height, img_width, n_classes))

    for i in range(n_classes):
        # label = label_element[0][i].numpy()
        label = i

        try:
            rle = label_element[1][i].numpy()
        except AttributeError:
            rle = label_element[1][i]


        if rle != "-1":
            class_mask = rle2mask(
                rle,
                img_size=(img_height, img_width),
                fill_color=(1),
            )
            class_mask = class_mask[..., 0]  # take just one channel
            mask[..., int(label)] = class_mask

    # for numerical vector instead of one-hot matrix of labels
    if not one_hot:
        mask = np.concatenate([np.zeros((img_height, img_width, 1)), mask], axis=-1)
        mask = np.argmax(mask, axis=-1, keepdims=True)

    return mask


def create_mask(y_pred):
    """
    Creats a tensor of binary masks from the unet model softmax output.

    Args:
        y_pred (np.ndarray) - of shape (b, h, w, 5)

    """

    pred_mask = tf.argmax(y_pred, axis=-1)
    pred_mask = tf.one_hot(pred_mask, 5)

    return pred_mask.numpy()


def calculate_class_weight_map_ip(classes, imgid_to_classid_mapping):
    """
    Computes class based weight map that are inversely proportional (IP) to the class frequency.

    Args:
        classes (list) - list of class id's
        imgid_to_classid_mapping (pd.Series) - mappint of imgid to classid's


    Returns:
        dict
    """
    classes = np.sort(classes)

    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=imgid_to_classid_mapping
    )

    return dict(zip(classes, np.around(weights, 4)))


def calculate_class_weight_map_ens(beta, samples_per_cls, classes):
    """
    Computes class balanced weight map based on Effective Number of Samples (ENS).

    Class Balanced Loss: ((1-beta)/(1-beta^n))

    As described here: https://arxiv.org/pdf/1901.05555v1.pdf
    Adapted from: https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py

    Args:
        beta (float) - hyperparam for class balanced loss
        samples_per_cls (list) - list of size len(classes)
        classes (list) - list of unique class_ids

    Returns:
        dict mapping of class_id to weight

    """
    classes = np.sort(classes)

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(classes)

    return dict(zip(classes, np.around(weights, 4)))
