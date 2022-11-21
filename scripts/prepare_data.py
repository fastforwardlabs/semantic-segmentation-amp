import os
from src.dataset import SegmentationDataset

IMG_SHAPE = (256, 1600)
ANNOTATIONS_PATH = "data/train.csv"
TRAIN_IMG_PATH = "data/train_images/"

# instantiate dataset, then preprocess and save out mask annotations as .pngs
sd = SegmentationDataset(
    label_file=ANNOTATIONS_PATH, img_dir_path=TRAIN_IMG_PATH, img_shape=IMG_SHAPE, drop_classes=True, test_size=0.1
)
sd.preprocess_save_mask_labels()