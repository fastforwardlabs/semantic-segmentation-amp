import numpy as np
import tensorflow as tf

from src.data_utils import rle2mask


class SegmentationDataPipeline:
    """
    Callable utility class for creating TensorFlow data pipelines
    from SegementationDataset sequences.
    
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
            "shuffle_buffer_size": 500,   # off if False
            "batch_size": 8,
            "prefetch": tf.data.AUTOTUNE, # off if False
        },
    ):
        self.img_height, self.img_width = img_shape
        self.label_type = label_type
        self.pipeline_options = pipeline_options

    def __call__(self, img_seq, label_seq):
        print("img:", type(img_seq), len(img_seq))
        print("label:", type(label_seq), len(label_seq))

        img_ds = tf.data.Dataset.from_tensor_slices(img_seq).map(
            self.prepare_image, num_parallel_calls=self.pipeline_options["map_parallel"]
        )
        
        if self.label_type == "inline":
            label_ds = tf.data.Dataset.from_tensor_slices(label_seq).map(
                self.tf_prepare_mask_label, num_parallel_calls=self.pipeline_options["map_parallel"]
            )
            
        elif self.label_type == "preprocessed":
            label_ds = tf.data.Dataset.from_tensor_slices(label_seq).map(
                self.prepare_image, num_parallel_calls=self.pipeline_options["map_parallel"]
            )
            
            
        zip_ds = tf.data.Dataset.zip((img_ds, label_ds))

        if self.pipeline_options["cache"]:
            print("Caching")
            zip_ds = zip_ds.cache()
            
        if self.pipeline_options["shuffle_buffer_size"]:
            print("Shuffling")
            zip_ds = zip_ds.shuffle(self.pipeline_options["shuffle_buffer_size"], seed=42)
            
        if self.pipeline_options["batch_size"]:
            print("Batching")
            zip_ds = zip_ds.batch(self.pipeline_options["batch_size"])

        if self.pipeline_options["prefetch"]:
            print("Prefetching")
            zip_ds = zip_ds.prefetch(self.pipeline_options["prefetch"])

        return zip_ds

    def prepare_image(self, img_path):
        """
        Loads and preprocesses image given a path.

        Args:
            img_path (str)
        """

        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img)
        img = tf.image.resize(img, (self.img_height, self.img_width))
        img = tf.image.convert_image_dtype(img, tf.float32)

        return img

    def prepare_mask_label(self, label_element):
        """
        Prepares image annotation labels as matrix of binary mask channels.

        Initializes empty matrix of desired size. Converts each RLE label to
        binary mask, and inserts each of those masks in the appropriate matrix channel.

        Args:
            label_element (tf.Tensor: shape (2,5), dtype=string)
            mask_height (int)
            mask_width (int)

        Returns:
            tf.Tensor (float64)
        """

        mask = np.zeros((self.img_height, self.img_width, 4))

        for i in range(label_element.shape[1]):
            label = label_element[0][i].numpy()
            rle = label_element[1][i].numpy()

            if rle != "-1":
                class_mask = rle2mask(
                    rle,
                    img_size=(self.img_height, self.img_width),
                    fill_color=(1),
                )
                class_mask = class_mask[..., 0]  # take just one channel
                mask[..., int(label) - 1] = class_mask

        return mask

    def tf_prepare_mask_label(self, label_element):
        """
        A `tf.py_function` wrapper for prepare_mask_label
        logic. Required when using non-TF operations.

        """

        mask = tf.py_function(
            func=self.prepare_mask_label,
            inp=[label_element],
            Tout=[tf.float64],
        )
        
        return mask[0]
