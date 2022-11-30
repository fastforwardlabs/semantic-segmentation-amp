import os
import shutil

from src.dataset import SegmentationDataset
from src.model_utils import collect_experiment_scores

BASE_PATH = "/home/cdsw"
LOG_DIR = os.path.join(BASE_PATH, "logs")


def prepare_data():
    """
    This script is intended to prepare data and model assets for the AMP. Specifically,
    the actions taken are:
        1. Check if Kaggle API credentials were provided, download full dataset if so
        2. If not, use local sample. Set environment variable accordingly
        3. Save out preprocessed segmentation mask labels for the dataset examples
        4. Copy a trained model from ~/logs to ~/model directory to be used by Streamlit app

    """

    # 1. check if API creds were provided
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

    # 2. check if download was successful and set dataset path accordingly
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

    # 3. instantiate dataset, then preprocess and save out mask annotations as .pngs
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

    # 4. get saved model from ~/logs and copy to ~/model directory
    experiment_scores = collect_experiment_scores(log_dir=LOG_DIR)
    best_dice_exp_idx = experiment_scores[
        "val_dice_score"
    ].idxmax()  # get best experiment
    best_dice_exp = experiment_scores.iloc[best_dice_exp_idx].experiment
    model_name = os.listdir(os.path.join(LOG_DIR, best_dice_exp, "max_val_dice"))[
        -1
    ]  # get best model (last saved checkpoint) from experiment

    # copy model to ~/model directory
    MODEL_PATH = os.path.join(LOG_DIR, best_dice_exp, "max_val_dice", model_name)
    NEW_MODEL_PATH = os.path.join(BASE_PATH, "model", "best_model.h5")
    os.makedirs(os.path.dirname(NEW_MODEL_PATH))
    shutil.copy(MODEL_PATH, NEW_MODEL_PATH)


if __name__ == "__main__":
    prepare_data()
