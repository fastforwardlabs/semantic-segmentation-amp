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

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.data_utils import (
    prepare_mask_label,
    add_background_channel,
    calculate_class_weight_map_ip,
    calculate_class_weight_map_ens,
)


class SegmentationDataset:
    """
    Dataset utility class.

    Provides functionality for loading/preprocessing images and segmentation masks
    into format combatiple with `tf.data`, as well as functionality for general modeling strategies
    like train/test splits and sampling strategies.

    Attributes:
        test_size (float) - fractional indication of held out test set size
        label_file (str) - path to train.csv file provided with dataset
        img_dir_path (str) - path to /train_images directory provided with dataset
        img_shape (str) - height x width of working images
        drop_classes (bool) - flag that toggels working with subset of image classes (excludes classes 1 & 2)
        sample_weight_strategy (str) - specify strategy for calculating sample weights (ens or ip). See `_build_class_weight_map()`.
        sample_weight_ens_beta (float) - see `_build_class_weight_map()` for details

    """

    def __init__(
        self,
        test_size,
        label_file,
        img_dir_path,
        img_shape,
        drop_classes=True,
        sample_weight_strategy="ens",
        sample_weight_ens_beta=0.99,
    ):
        self.test_size = test_size
        self.label_file = label_file
        self.img_dir_path = img_dir_path
        self.img_height, self.img_width = img_shape
        self.drop_classes = drop_classes

        self.label_dir_path = self._set_label_path()
        self.df = self._prepare_dataset()
        self.imgid_to_classid_mapping = self._imgid_to_classid_mapping()
        self.class_weight_map = self._build_class_weight_map(
            strategy=sample_weight_strategy, beta=sample_weight_ens_beta
        )
        self.train_imgs, self.test_imgs = self.get_train_test_split(
            test_size=self.test_size
        )

    def _set_label_path(self):

        label_dir_path = os.path.join(os.path.dirname(self.img_dir_path), "mask_labels")
        os.makedirs(label_dir_path, exist_ok=True)

        return label_dir_path

    def _prepare_dataset(self):
        df = self._load_dataset(self.label_file, self.img_dir_path)
        df = self._normalize_df(df)
        if self.drop_classes:
            df = self._drop_imgs_by_class(df, class_ids_to_drop=[0, 1, 2])
        return df

    def _load_dataset(self, label_file, img_dir_path, use_non_defective=False):
        """
        Prepares a dataframe with all images and corresponding masks.

        Since the provided annotations dataset (csv) excludes non-defective examples,
        this function merges all image data into one dataframe. Non-defective examples
        recieve an EncodedPixels value of -1.

        Args:
            annotations_path (str) - path to annotations csv file
            img_path (str) - path to corresponding images directory
            use_non_defective (bool) - whether to include non-defective images

        Returns:
            pd.DataFrame
        """
        df = pd.read_csv(label_file)
        img_paths = os.listdir(img_dir_path)

        if use_non_defective:
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

    def _drop_imgs_by_class(self, df, class_ids_to_drop):
        """
        Utility to remove all data for specified class_id's.

        This function will drop all images from the provided dataframe that correspond
        to a defect of type class_ids_to_drop.

        Args:
            df (pd.DataFrame)
            class_ids_to_drop (list)

        Returns:
            pd.DataFrame

        """

        if len(class_ids_to_drop) == 0:
            raise Exception("Must specify which classes to drop.")

        img_ids_to_drop = []
        for img_id, table in df.groupby("ImageId")[["ClassId", "EncodedPixels"]]:

            table_dict = table.to_dict("list")

            for class_id in class_ids_to_drop:
                if table_dict["EncodedPixels"][class_id] != -1:
                    img_ids_to_drop.append(img_id)

        df = df[~df.ImageId.isin(img_ids_to_drop)]

        return df[~df.ClassId.isin(class_ids_to_drop)].reset_index(drop=True)

    def preprocess_save_mask_labels(self):
        """
        Prepreocesses segmentation masks for each image and saves them as .png files locally for easier loading
        into tf.data pipeline.

        Masks are saved with 3 channels.

        """
        img_ids = self.df.ImageId.unique().tolist()
        label_seq = self.get_label_sequence(img_ids, label_type="inline")

        if len(os.listdir(self.label_dir_path)) == 0:
            print(
                f"Preprocessing RLE mask labels and saving out as .png files to {self.label_dir_path}"
            )

            for i, (img_id, label_element) in enumerate(zip(img_ids, label_seq)):
                mask_tensor = prepare_mask_label(
                    label_element, self.img_height, self.img_width
                )
                mask_tensor = add_background_channel(mask_tensor)
                tf.keras.utils.save_img(
                    path=os.path.join(self.label_dir_path, f"{img_id[:-4]}.png"),
                    x=mask_tensor,
                )

        else:
            print("Segmentation masks have already been preprocessed and saved")

    def _imgid_to_classid_mapping(self):
        """
        Returns a pd.Series mapping img_id's to a designated class_id.

        Class ID's include the four defect types (1,2,3,4), as well as ID's for
        multiclass (-2) and non-defective (-1).

        Returns:
            pd.Series

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

        return pd.concat([unique_image_defectclass, unique_image_nondefectclass])

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

        train_imgs, test_imgs = train_test_split(
            self.imgid_to_classid_mapping,
            test_size=test_size,
            random_state=42,
            shuffle=True,
            stratify=self.imgid_to_classid_mapping,
        )

        return train_imgs.index.tolist(), test_imgs.index.tolist()

    def _build_class_weight_map(self, strategy="ens", beta=0.999):
        """
        Builds a class weighting estimate for each class to help balance
        the imbalanced samples from the dataset.

        Supports two strategies:
            Effective Number of Samples (ens)
            Inversely Proportional (ip)

        Args:
            strategy (str) - "ens" or "ip"
            beta (float) - hyperparam for ENS strategy

        Returns:
            dict
        """
        classes = (
            self.imgid_to_classid_mapping.value_counts().sort_index().index.tolist()
        )
        samples_per_class = (
            self.imgid_to_classid_mapping.value_counts().sort_index().tolist()
        )

        if strategy == "ens":
            return calculate_class_weight_map_ens(
                beta=beta, samples_per_cls=samples_per_class, classes=classes
            )
        elif strategy == "ip":
            return calculate_class_weight_map_ip(
                classes=classes, imgid_to_classid_mapping=self.imgid_to_classid_mapping
            )
        elif strategy == "NA":
            pass
        else:
            raise Exception("Must specify valid strategy: 'ens' or 'ip'.")

    def resample_train_set(self, train_imgs, over_sample=True):
        """
        Apply naive resampling (under or over sampling) to provided list of train-set image id's.

        This function under/over samples for all classes other than multiclass, then combines
        and shuffles all image_ids before returning.

        """

        print(
            "Old Class Distribution: \n",
            self.imgid_to_classid_mapping[train_imgs]
            .value_counts(normalize=False)
            .sort_index(),
        )

        rs = (
            RandomOverSampler(random_state=42)
            if over_sample
            else RandomUnderSampler(random_state=42)
        )

        # remove multi-class examples from oversampling
        img_class_ids = self.imgid_to_classid_mapping[train_imgs]
        multiclass = img_class_ids[img_class_ids == -2]
        img_class_ids = img_class_ids[img_class_ids != -2]

        img_ids = img_class_ids.index.to_numpy().reshape(-1, 1)
        class_ids = img_class_ids.to_numpy()
        resampled_img_ids, resampled_class_ids = rs.fit_resample(img_ids, class_ids)

        # combine and shuffle multi-class samples
        resampled = pd.Series(
            index=np.squeeze(resampled_img_ids), data=resampled_class_ids
        )
        combined = pd.concat((resampled, multiclass)).sample(frac=1, random_state=42)

        print(
            "New Class Distribution: \n",
            combined.value_counts(normalize=False).sort_index(),
        )

        return list(combined.index)

    def get_image_sequence(self, img_ids):
        """
        Formats image paths for each image in img_ids into a sequence (List) that can be used
        to create tf.data.Dataset.
        """
        return [os.path.join(self.img_dir_path, img_id) for img_id in img_ids]

    def get_label_sequence(self, img_ids, label_type):
        """
        Formats label annotations for each image into a sequence that can be used to create
        a tf.data.Dataset. Only defect annotations are included in the sequence, meanining class 0
        is dropped.

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
                    [str(x) for i, x in enumerate(labels["ClassId"])],
                    [str(x) for i, x in enumerate(labels["EncodedPixels"])],
                )
                label_elements.append(element)

            return label_elements

        elif label_type == "preprocessed":

            return [
                os.path.join(self.label_dir_path, f"{img_id[:-4]}.png")
                for img_id in img_ids
            ]

    def get_sample_weight_sequence(self, img_ids):
        """
        Formats class weightings for each image in img_ids into a sequence (List) that can be used
        to create tf.data.Dataset.

        """
        return np.expand_dims(
            [
                self.class_weight_map[id]
                for id in self.imgid_to_classid_mapping[img_ids].values
            ],
            axis=-1,
        )
