!pip3 install -r requirements.txt

# NOTE - In order to download the dataset, you must create a Kaggle account and save an API token to {API_CREDS_PATH}
!mkdir data
!kaggle competitions download -c severstal-steel-defect-detection
!mv severstal-steel-defect-detection.zip data/severstal-steel-defect-detection.zip
!unzip data/severstal-steel-defect-detection.zip -d ~/data