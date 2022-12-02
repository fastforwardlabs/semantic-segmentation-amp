import numpy as np
import tensorflow as tf

from src.data_utils import prepare_mask_label, add_background_channel


class SegmentationDataPipeline:
    """
    Callable utility class for creating TensorFlow data pipelines
    from SegementationDataset sequences with optional pipeline optimization settings.

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

        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img)
        img = tf.image.resize(img, (self.img_height, self.img_width))
        img = tf.image.convert_image_dtype(img, tf.float32)

        return img

    def load_mask_label(self, mask_label_path):
        mask = np.load(mask_label_path.numpy().decode())
        return mask

    def tf_load_mask_label(self, mask_label_path):

        mask = tf.py_function(
            func=self.load_mask_label,
            inp=[mask_label_path],
            Tout=[tf.float32],
        )

        return mask[0]

    def tf_add_background_channel(self, mask):

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
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def augment(self, image, mask, sample_weight=None):

        if tf.random.uniform(()) > 0.5:
            # Random flipping of the image and mask
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        if sample_weight is not None:
            return image, mask, sample_weight
        else:
            return image, mask
