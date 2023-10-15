# https://www.scaler.com/topics/pytorch/how-to-split-a-torch-dataset/
# https://discuss.pytorch.org/t/apply-different-transform-data-augmentation-to-train-and-validation/63580

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split


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