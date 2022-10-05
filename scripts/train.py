import os
import datetime

import tensorflow as tf
from tensorflow.keras import layers

from src.model import unet_model
from src.dataset import SegmentationDataset
from src.data_pipeline import SegmentationDataPipeline

IMG_SHAPE = (256, 1600)
BATCH_SIZE = 8
EPOCHS = 20
ANNOTATIONS_PATH = "data/train.csv"
TRAIN_IMG_PATH = "data/train_images/"
LOG_DIR = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def normalize(image, mask):
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0
    return image, mask

def main():

    # instantiate dataset and pipelne
    sd = SegmentationDataset(
        label_file=ANNOTATIONS_PATH,
        img_dir_path=TRAIN_IMG_PATH,
        img_shape=IMG_SHAPE,
    )

    # create train/test & x/y splits
    train_imgs, test_imgs = sd.get_train_test_split(test_size=0.2)
    X_train, y_train = sd.get_image_sequence(train_imgs), sd.get_label_sequence(
        train_imgs, label_type="preprocessed"
    )
    X_test, y_test = sd.get_image_sequence(test_imgs), sd.get_label_sequence(
        test_imgs, label_type="preprocessed"
    )

    # create dataset pipelines
    sdp = SegmentationDataPipeline(
        img_shape=IMG_SHAPE,
        label_type="preprocessed",
        pipeline_options={
            "map_parallel": None,
            "cache": False,
            "shuffle_buffer_size": False,
            "batch_size": BATCH_SIZE,
            "prefetch": False,
        },
    )
    train_dataset = sdp(X_train, y_train)
    test_dataset = sdp(X_test, y_test)
    
    train_dataset = train_dataset.map(normalize)
    test_dataset = test_dataset.map(normalize)
    
    
    # build model
    unet = unet_model(IMG_SHAPE)
    
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(LOG_DIR, "best_model.h5"), save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
    ]
    

    hist = unet.fit(
        train_dataset, epochs=EPOCHS, validation_data=test_dataset, callbacks=callbacks
    )


if __name__ == "__main__":
    main()
    