import os
from src.dataset import SegmentationDataset

# check if API creds were provided
if os.getenv("KAGGLE_USERNAME") != "" and os.getenv("KAGGLE_API_KEY") != "":

    # try to download data
    res = os.system(
        "mkdir data && \
        kaggle competitions download -c severstal-steel-defect-detection && \
        mv severstal-steel-defect-detection.zip data/severstal-steel-defect-detection.zip && \
        unzip data/severstal-steel-defect-detection.zip -d ~/data"
    )
else:
    res = None

# check if download was successful and set dataset path accordingly
BASE_PATH = "/home/cdsw"
if res == 0 and os.path.exists(os.path.join(BASE_PATH, "data/train.csv")):
    os.environ["DATASET_DIR"] = os.path.join(BASE_PATH, "data")
    print("Successfully downloaded data from Kaggle!")
else:
    os.environ["DATASET_DIR"] = os.path.join(BASE_PATH, os.environ["DATASET_DIR"])
    os.system(
        f"tar -xvf /home/cdsw/sample_data.tar.gz && mv /home/cdsw/sample_data {os.environ['DATASET_DIR']}"
    )

    print("Could not download data from Kaggle, using sample data instead.")
    print("Response: ", res)


# instantiate dataset, then preprocess and save out mask annotations as .pngs
IMG_SHAPE = (256, 1600)
ANNOTATIONS_PATH = os.path.join(os.environ["DATASET_DIR"], "train.csv")
TRAIN_IMG_PATH = os.path.join(os.environ["DATASET_DIR"], "train_images")

sd = SegmentationDataset(
    label_file=ANNOTATIONS_PATH,
    img_dir_path=TRAIN_IMG_PATH,
    img_shape=IMG_SHAPE,
    drop_classes=True,
    test_size=0.1,
)
sd.preprocess_save_mask_labels()
