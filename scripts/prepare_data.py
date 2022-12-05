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
import shutil

from src.dataset import SegmentationDataset
from src.model_utils import collect_experiment_scores
from src.data_utils import set_dataset_path, get_dataset_path

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
        5. Create temp directory in support of Streamlit app UI

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
        set_dataset_path(path=os.path.join(BASE_PATH, "data"))
        print("Successfully downloaded data from Kaggle!")
    else:
        set_dataset_path(path=os.path.join(BASE_PATH, "sample_data"))
        os.system(
            f"tar -xvf /home/cdsw/sample_data.tar.gz && mv /home/cdsw/sample_data {os.path.join(BASE_PATH, "sample_data")}"
        )

        print("Could not download data from Kaggle, using sample data instead.")
        print("Response: ", res)

    # 3. instantiate dataset, then preprocess and save out mask annotations as .pngs
    DATASET_DIR = get_dataset_path()
    IMG_SHAPE = (256, 1600)
    ANNOTATIONS_PATH = os.path.join(DATASET_DIR, "train.csv")
    TRAIN_IMG_PATH = os.path.join(DATASET_DIR, "train_images")

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

    # 5. create tmp directory for streamlit app
    for mask_type in ["pred", "true"]:
        os.makedirs(os.path.join(BASE_PATH, f"apps/tmp/{mask_type}"))


if __name__ == "__main__":
    prepare_data()
