import os
from src.dataset import SegmentationDataset

# check if API creds were provided
if os.getenv("KAGGLE_USERNAME") is not "" and os.getenv("KAGGLE_API_KEY") is not "":

    # try to download data
    res = os.system(
        "mkdir data && \
        kaggle competitions download -c severstal-steel-defect-detection && \
        mv severstal-steel-defect-detection.zip data/severstal-steel-defect-detection.zip && \
        unzip data/severstal-steel-defect-detection.zip -d ~/data"
    )

# check if download was successful and set dataset path accordingly
if res == 0 and os.path.exists("/home/cdsw/data/train.csv"):
    os.environ["DATASET_DIR"] = "/home/cdsw/data/"
    print("Successfully downloaded data from Kaggle!")
else:
    os.environ["DATASET_DIR"] = "/home/cdsw/sample_data/"
    os.system("tar -xvf /home/cdsw/sample_data.tar.gz")
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
