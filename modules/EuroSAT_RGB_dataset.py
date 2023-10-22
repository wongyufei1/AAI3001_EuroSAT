# https://www.scaler.com/topics/pytorch/how-to-split-a-torch-dataset/
# https://discuss.pytorch.org/t/apply-different-transform-data-augmentation-to-train-and-validation/63580

import os
import csv

import torch
import rasterio
import numpy as np
import glob
from PIL import Image
from torch.utils.data import Dataset


class EuroSatRgbDataset(Dataset):
    def __init__(self, data, indices_to_labels, transform=None):
        self.data = data
        self.indices_to_labels = indices_to_labels
        self.transform = transform

    def __getitem__(self, idx):
        # load image and standardize to RGB
        img = Image.open(self.data[idx]["img_path"]).convert("RGB")


        # transform image
        if self.transform:
            img = self.transform(img)

        # encode label
        label = encode_labels([self.data[idx]["label"]],
                              self.indices_to_labels)

        # get image and encoded labels
        return img, label

    def __len__(self):
        return len(self.data)


# EuroSatRgbDataset for multi-label requirements
class EuroSatRgbDatasetMultiLabel(EuroSatRgbDataset):
    def __init__(self, data, indices_to_labels, transform=None):
        super().__init__(data, indices_to_labels, transform)

    def __getitem__(self, idx):
        # load image and standardize to RGB
        img = Image.open(self.data[idx]["img_path"]).convert("RGB")

        # transform image
        if self.transform:
            img = self.transform(img)

        label_before_encoding = [(self.data[idx]["label"])]

        # extra requirements for labelling
        if self.data[idx]["label"] == "AnnualCrop" or self.data[idx]["label"] == "PermanentCrop":
            label_before_encoding = ['AnnualCrop', 'PermanentCrop']
        elif self.data[idx]["label"] == "Forest":
            label_before_encoding.append("HerbaceousVegetation")

        # encode label
        label = encode_labels(label_before_encoding, self.indices_to_labels)

        # get image and encoded labels
        return img, label


class EuroSatHyperSpectralDataset(EuroSatRgbDataset):
    def __init__(self, data, indices_to_labels, transform=None):
        super().__init__(data, indices_to_labels, transform)

    def __getitem__(self, idx):
        bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        with rasterio.open(self.data[idx]["img_path"]) as dataset:
            img = dataset.read(bands).astype(np.float64)  # (13, 64, 64)

        img = torch.from_numpy(img)

        # # transform image
        if self.transform:
            img = self.transform(img)

        # encode label
        label = encode_labels([self.data[idx]["label"]],
                              self.indices_to_labels)
        # get image and encoded labels
        return img, label


def extract_bands(img_path):
    bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    with rasterio.open(img_path) as dataset:
        img = dataset.read(bands)

    return [Image.fromarray(img[i]) for i in range(len(bands))]


def encode_labels(labels, indices_to_labels):
    """
    One-hot encode string label/s of a sample.
    Should work for both multiclass and multilabel samples.

    :param labels: all truth label/s of a sample
    :param indices_to_labels: indices to labels mapping
    :return: one-hot encoded label/s of a sample
    """
    # initialise all label indices to 0
    encoded_labels = torch.zeros(len(indices_to_labels))

    # update label indices to 1 for present labels
    for label in labels:
        if label in indices_to_labels:
            idx = indices_to_labels.index(label)
            encoded_labels[idx] = 1
        else:
            raise ValueError(f"Label \"{label}\" not within available labels.")

    return encoded_labels


def decode_labels(predictions, indices_to_labels):
    """
    Decode one-hot predictions into string label/s of a sample.
    Should work for both multiclass and multilabel samples.

    :param predictions: one-hot encoded predictions of a sample
    :param indices_to_labels: indices to labels mapping
    :return: all truth label/s strings of a sample
    """

    # get all indices where prediction is 1
    indices = [i for i, x in enumerate(predictions) if x == 1]

    # convert indices to class labels
    labels = [indices_to_labels[i] for i in indices]

    return labels


def load_data(root):
    data = []

    # load image files and their respective labels into memory
    for label in os.listdir(root):
        label_dir = os.path.join(root, label)

        for img in os.listdir(label_dir):
            img_path = os.path.join(root, label, img)
            data.append({"img_path": img_path, "label": label})

    return data


def load_split_data(root):
    rgb_path = f"{root}/EuroSAT"
    hyperspectral_path = f"{root}/EuroSATallBands"

    # Paths to predefined splits
    paths = {
        'rgb_train':
            {
                "root": rgb_path,
                "path": f"{root}/EuroSAT/train.csv",
                "split": []
             },
        'rgb_val':
            {
                "root": rgb_path,
                "path": f"{root}/EuroSAT/validation.csv",
                "split": []
            },
        'rgb_test':
            {
                "root": rgb_path,
                "path": f"{root}/EuroSAT/test.csv",
                "split": []
            },
        'hyperspectral_train':
            {
                "root": hyperspectral_path,
                "path": f"{root}/EuroSATallBands/train.csv",
                "split": []
            },
        'hyperspectral_val':
            {
                "root": hyperspectral_path,
                "path": f"{root}/EuroSATallBands/validation.csv",
                "split": []
            },
        'hyperspectral_test':
            {
                "root": hyperspectral_path,
                "path": f"{root}/EuroSATallBands/test.csv",
                "split": []
            }

    }

    for key, path in paths.items():
        file = open(path["path"])
        csv_reader = csv.reader(file, delimiter=',')

        i = 0

        for row in csv_reader:
            # Skip the first row
            if "ClassName" in row:
                pass
            else:
                if "rgb" in key:
                    img_path = os.path.join(path["root"], row[1])
                    paths[key]["split"].append(
                        {"img_path": img_path, "label": row[3]})
                else:
                    img_path = os.path.join(path["root"], row[0])
                    paths[key]["split"].append(
                        {"img_path": img_path, "label": row[2]})

    return paths['rgb_train']['split'], paths['rgb_val']['split'], paths['rgb_test']['split'], paths['hyperspectral_train']['split'], paths['hyperspectral_val']['split'], paths['hyperspectral_test']['split']


def load_data_hyperspectral(root):
    rgb_data = []
    hyperspectral_data = []

    rgb_path = f"{root}/EuroSAT"
    hyperspectral_path = f"{root}/EuroSATallBands"

    # Load RGB image files paths and their respective labels into memory
    for label in os.listdir(rgb_path):

        # Check if the file contains ".DS", "csv and "json"
        if ".DS" in label or ".csv" in label or ".json" in label:
            pass
        else:
            label_dir = os.path.join(rgb_path, label)

            for img in os.listdir(label_dir):
                img_path = os.path.join(rgb_path, label, img)
                rgb_data.append({"img_path": img_path, "label": label})

    # Load Hyperspectral file paths and their respective labels into memory
    for label in os.listdir(hyperspectral_path):
        # Check if the file contains ".DS", "csv and "json"
        if ".DS" in label or ".csv" in label or ".json" in label:
            pass
        else:
            label_dir = os.path.join(hyperspectral_path, label)

            for img in os.listdir(label_dir):
                img_path = os.path.join(hyperspectral_path, label, img)
                hyperspectral_data.append(
                    {"img_path": img_path, "label": label})

    return rgb_data, hyperspectral_data





