import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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