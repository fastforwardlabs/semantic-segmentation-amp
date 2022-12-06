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

import numpy as np
import tensorflow as tf

from src.data_utils import prepare_mask_label, add_background_channel


class SegmentationDataPipeline:
    """
    Callable utility class for creating TensorFlow data pipelines
    from SegementationDataset sequences with optional pipeline optimization settings.

    Specifically, the user should specify image shape, whether the labels are "preprocessed"
    or "inline", and which `tf.data` pipeline optimizations to apply upon class instantiation.

    For more details on pipeline optimizations, see:
         https://www.tensorflow.org/guide/data_performance

    Args:
        img_shape (tuple)
        label_type (str) - "preprocessed" or "inline"
        pipeline_options (dict)

    """

    def __init__(
        self,
        img_shape,
        label_type,
        pipeline_options={
            "map_parallel": tf.data.AUTOTUNE,  # off if None
            "cache": True,
            "shuffle_buffer_size": 25,  # off if False
            "batch_size": 8,
            "prefetch": tf.data.AUTOTUNE,  # off if False
        },
    ):
        self.img_height, self.img_width = img_shape
        self.label_type = label_type
        self.pipeline_options = pipeline_options

    def __call__(self, img_seq, label_seq, is_train=True, sample_weights=None):
        """
        Apply `tf.data` transformations to the provided img_seq and label_seq to
        return a `tf.Dataset` that can be used for model training.

        For more info on usage, see `/notebooks/Data_Pipeline_Walkthrough.ipynb`.

        Args:
            img_seq (list) - list of sequence elements
            label_seq (list) - list of label elements
            is_train (bool, optional) - if training pipeline, this applies horizontal
                 flips as form of data augmentation
            sample_weights (list) - list of sample weights

        Returns:
            tf.data.Dataset
        """

        img_ds = (
            tf.data.Dataset.from_tensor_slices(img_seq)
            .map(
                self.load_image,
                num_parallel_calls=self.pipeline_options["map_parallel"],
            )
            .map(
                self.normalize, num_parallel_calls=self.pipeline_options["map_parallel"]
            )
        )

        if self.label_type == "inline":
            label_ds = (
                tf.data.Dataset.from_tensor_slices(label_seq)
                .map(
                    self.tf_prepare_mask_label,
                    num_parallel_calls=self.pipeline_options["map_parallel"],
                )
                .map(
                    self.tf_add_background_channel,
                    num_parallel_calls=self.pipeline_options["map_parallel"],
                )
                .map(
                    self.normalize,
                    num_parallel_calls=self.pipeline_options["map_parallel"],
                )
            )

        elif self.label_type == "preprocessed":
            label_ds = (
                tf.data.Dataset.from_tensor_slices(label_seq)
                .map(
                    self.load_image,
                    num_parallel_calls=self.pipeline_options["map_parallel"],
                )
                .map(
                    self.normalize,
                    num_parallel_calls=self.pipeline_options["map_parallel"],
                )
            )

        if sample_weights is not None:
            sample_weight_ds = tf.data.Dataset.from_tensor_slices(sample_weights)

            zip_ds = tf.data.Dataset.zip((img_ds, label_ds, sample_weight_ds))

        else:
            zip_ds = tf.data.Dataset.zip((img_ds, label_ds))

        if is_train:
            print("Augmenting")
            zip_ds = zip_ds.map(
                self.augment, num_parallel_calls=self.pipeline_options["map_parallel"]
            )

        if self.pipeline_options["cache"]:
            print("Caching")
            zip_ds = zip_ds.cache()

        if self.pipeline_options["shuffle_buffer_size"]:
            print("Shuffling")
            zip_ds = zip_ds.shuffle(
                self.pipeline_options["shuffle_buffer_size"], seed=42
            )

        if self.pipeline_options["batch_size"]:
            print("Batching")
            zip_ds = zip_ds.batch(self.pipeline_options["batch_size"])

        if self.pipeline_options["prefetch"]:
            print("Prefetching")
            zip_ds = zip_ds.prefetch(self.pipeline_options["prefetch"])

        return zip_ds

    def load_image(self, img_path):
        """
        Loads an image from provided path and returns as tf.Tensor
        """

        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img)
        img = tf.image.resize(img, (self.img_height, self.img_width))
        img = tf.image.convert_image_dtype(img, tf.float32)

        return img

    def tf_add_background_channel(self, mask):
        """
        Adds a background channel to a segmentation mask tensor
        of defects. This is needed for the lost function to softmax
        correctly.
        """

        mask = tf.py_function(
            func=add_background_channel,
            inp=[mask],
            Tout=[tf.float32],
        )

        return mask[0]

    def tf_prepare_mask_label(self, label_element):
        """
        A `tf.py_function` wrapper for prepare_mask_label
        logic. Required when using non-TF operations.

        """

        mask = tf.py_function(
            func=prepare_mask_label,
            inp=[label_element, self.img_height, self.img_width],
            Tout=[tf.float32],
        )

        return mask[0]

    def normalize(self, image):
        """
        Normalize pixel values between 0 and 1
        """
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def augment(self, image, mask, sample_weight=None):
        """
        Randomly apply horizontal flips as form of
        data augmentation at train time
        """

        if tf.random.uniform(()) > 0.5:
            # Random flipping of the image and mask
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        if sample_weight is not None:
            return image, mask, sample_weight
        else:
            return image, mask
