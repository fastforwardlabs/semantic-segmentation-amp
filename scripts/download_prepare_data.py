# NOTE - In order to download the dataset, you must create a Kaggle account and save an API token to {API_CREDS_PATH}

import os

from src.dataset import SegmentationDataset

API_CREDS_PATH = "~/.kaggle/kaggle.json"

!mkdir data
!kaggle competitions download -c severstal-steel-defect-detection
!mv severstal-steel-defect-detection.zip data/severstal-steel-defect-detection.zip
!unzip severstal-steel-defect-detection.zip

