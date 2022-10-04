import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SegmentationDataset:
    """
    Dataset utility class.

    Provides functionality for preprocessing images and segmentation masks
    into format combatiple with `tf.data`.

    """

    def __init__(self, label_file, img_dir_path, img_shape):
        self.label_file = label_file
        self.img_dir_path = img_dir_path
        self.img_height, self.img_width = img_shape

        self.label_dir_path = self._set_label_path()
        self.df = self._prepare_dataset()

        if len(os.listdir(self.label_dir_path)) == 0:
            self._preprocess_save_mask_labels()

    def _set_label_path(self):

        label_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(self.img_dir_path)), "mask_labels"
        )
        os.makedirs(label_dir_path, exist_ok=True)

        return label_dir_path

    def _prepare_dataset(self):
        df = self._load_dataset(self.label_file, self.img_dir_path)
        df = self._normalize_df(df)
        return df

    def _load_dataset(self, label_file, img_dir_path):
        """
        Prepares a dataframe with all images and corresponding masks.

        Since the provided annotations dataset (csv) excludes non-defective examples,
        this function merges all image data into one dataframe. Non-defective examples
        recieve an EncodedPixels value of -1.

        Args:
            annotations_path (str) - path to annotations csv file
            img_path (str) - path to corresponding images directory

        Returns:
            pd.DataFrame
        """
        df = pd.read_csv(label_file)
        img_paths = os.listdir(img_dir_path)

        non_defect_imgs = list(set(img_paths) - set(df.ImageId))
        non_defect_imgs_df = pd.DataFrame(
            {"ImageId": non_defect_imgs, "ClassId": 0, "EncodedPixels": -1}
        )

        df = pd.concat([df, non_defect_imgs_df]).reset_index(drop=True)

        return df

    def _normalize_df(self, df):
        """
        Inserts dummy records so that each image_id has a row for each class_id.

        By normalizing the dataframe in this way, we have a consistent shape for each
        label to load as a tensor in tf.data pipeline.
        """

        dummy_records = []

        for img_id in df.ImageId.unique().tolist():

            tmp = df[df.ImageId == img_id]
            for class_id in [0, 1, 2, 3, 4]:
                if class_id not in tmp.ClassId.tolist():
                    dummy_record = (tmp.ImageId.values[0], class_id, -1)
                    dummy_records.append(dummy_record)

        dummy_df = pd.DataFrame(dummy_records, columns=df.columns)
        return (
            pd.concat([df, dummy_df])
            .sort_values(by=["ImageId", "ClassId"])
            .reset_index(drop=True)
        )

    def _preprocess_save_mask_labels(self):
        """
        Prepreocesses segmentation masks for each image and saves them as .png files locally for easier loading
        into tf.data pipeline.

        """
        img_ids = self.df.ImageId.unique().tolist()
        label_seq = self.get_label_sequence(img_ids)

        for img_id, label_element in zip(img_ids, label_seq):
            mask_tensor = self.prepare_mask_label(label_element)
            tf.keras.utils.save_img(
                path=os.path.join(self.label_dir_path, f"{img_id[:-4]}.png"),
                x=mask_tensor,
            )

    def get_train_test_split(self, test_size=0.2):
        """
        Splits all images into train and test set.

        Split is made to stratify across image classes where non-defective
        and mulitclass images are considered their own class.

        Args:
            test_size (float)

        Returns:
            Tuple[List] of image_ids

        """
        # get unique defective images, assign class (-2 if multiclass)
        unique_image_defectclass = (
            self.df[self.df.EncodedPixels != -1]
            .groupby("ImageId")[["ImageId", "ClassId"]]
            .apply(lambda x: x.iloc[0]["ClassId"] if len(x) == 1 else -2)
        )

        # get unique non-defective images, assign class of -1
        unique_image_nondefectclass = (
            self.df.groupby("ImageId")["EncodedPixels"]
            .apply(lambda x: -1 if not any([type(val) == str for val in x]) else np.nan)
            .dropna()
            .astype(int)
        )

        unique_image_classes = pd.concat(
            [unique_image_defectclass, unique_image_nondefectclass]
        )

        train_imgs, test_imgs = train_test_split(
            unique_image_classes,
            test_size=test_size,
            random_state=42,
            shuffle=True,
            stratify=unique_image_classes,
        )

        return train_imgs.index.tolist(), test_imgs.index.tolist()

    def get_image_sequence(self, img_ids):
        """
        Formats image paths for each image in img_ids into a sequence (List) that can be used
        to create tf.data.Dataset.
        """
        return [os.path.join(self.img_dir_path, img_id) for img_id in img_ids]

    def get_label_sequence(self, img_ids, label_type):
        """
        Formats label annotations for each image into a sequence taht can be used to create
        a tf.data.Dataset. 
        
            If label_type is "inline", sequence consistes of annotations (ClassID, EncodedPixels)
            for each image.
        
            If label_type is "preprocessed", sequence consists of a list of image paths.
    
        
        Args:
            img_ids (list)
            label_type (str) - "preprocessed" or "inline"

        """
        
        if label_type == "inline":
            label_dict = {
                img_id: table.to_dict("list")
                for img_id, table in self.df.groupby("ImageId")[
                    ["ClassId", "EncodedPixels"]
                ]
            }

            label_elements = []
            img_paths = [os.path.join(self.img_dir_path, img_id) for img_id in img_ids]

            for img_path in img_paths:
                img_id = img_path.split("/")[-1]
                labels = label_dict[img_id]
                element = (
                    [str(x) for x in labels["ClassId"]],
                    [str(x) for x in labels["EncodedPixels"]],
                )
                label_elements.append(element)

            return label_elements
        
        elif label_type == "preprocessed":
            
            return [os.path.join(self.label_dir_path, f"{img_id[:-4]}.png") for img_id in img_ids]

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

        for i in range(len(label_element[1])):
            label = label_element[0][i]
            rle = label_element[1][i]

            if rle != "-1":
                class_mask = rle2mask(
                    rle,
                    img_size=(self.img_height, self.img_width),
                    fill_color=(1),
                )
                class_mask = class_mask[..., 0]  # take just one channel
                mask[..., int(label) - 1] = class_mask

        return mask

    