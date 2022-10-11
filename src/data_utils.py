import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

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
    
    
def plot_sample_batch(sample_batch):
    """
    Plots a sample batch of data (images and masks) from a
    Tensorflow Dataset.
    
    Args:
        sample_batch - batch returned from tf dataset iterator
            (e.g. sample_batch = list(tf_dataset.take(1).as_numpy_iterator()))
        
    """
    fig = plt.figure(figsize=(12, 18))
    fig.suptitle("Sample Batch Visualized")

    x_batch = sample_batch[0][0]
    y_batch = sample_batch[0][1]

    batch_size = x_batch.shape[0]
    n_cols = 2
    n_rows = int(batch_size / n_cols)

    outer_grid = gridspec.GridSpec(n_rows, n_cols, wspace=0)

    for i in range(batch_size):
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            6, 1, subplot_spec=outer_grid[i], wspace=0.1, hspace=0.1
        )

        for j in range(6):
            if j == 0:
                ax = plt.Subplot(fig, inner_grid[j])
                ax.imshow(x_batch[i], vmin=0., vmax=1.)

            else:
                ax = plt.Subplot(fig, inner_grid[j])
                ax.imshow(y_batch[i][..., j - 1],  vmin=0., vmax=1.)

            if j == 0:
                ax.set_title(f"Ex {i}")

            ax.axis("off")

            fig.add_subplot(ax)

    plt.show()