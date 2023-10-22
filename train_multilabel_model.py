import os

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import models, transforms as T
from torchvision.models import EfficientNet_B0_Weights

from modules.EuroSAT_RGB_dataset import EuroSatRgbDatasetMultiLabel, load_data
from modules.EuroSAT_RGB_model import EuroSatRgbModelMultiLabel


def run(data_path):
    # make results reproducible
    np.random.seed(0)
    torch.manual_seed(0)

    # define class mapping
    indices_to_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation",
                         "Highway", "Industrial", "Pasture",
                         "PermanentCrop", "Residential", "River", "SeaLake"]

    # define configurations
    batch_size = 64
    epochs = 10
    loss = torch.nn.BCEWithLogitsLoss()  # Loss function for multi-label classification
    lrates = [0.01, 0.001]
    n_classes = len(indices_to_labels)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    save_dir = "rgb_multilabel_save_files"

    # define transforms, only need 1 transform for multi-label classification
    data_transforms = {
        "flip": {
            "train": T.Compose([
                T.Resize(64),
                T.RandomCrop(60),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])]),  # mean and std for each channel
            "valid": T.Compose([
                T.Resize(64),
                T.CenterCrop(60),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])]),  # mean and std for each channel
        }
    }

    """
        Split dataset into train, validation and test splits and load into batches
    """
    print("Loading dataset...")
    data = load_data(data_path)
    print(f"Total size: {len(data)}\n")

    # # Randomize the data and shuffle it so we can use a smaller sample for now
    # np.random.shuffle(data)
    # data = data[:1000]

    # split data
    print("Splitting dataset...")
    train_split, valid_split, test_split = random_split(data, (0.7, 0.15, 0.15))

    # save splits
    with open(os.path.join(save_dir, "rgb_multilabel_datasets_split.txt"), "w+") as file:
        file.write(",".join([str(idx) for idx in train_split.indices]) + "\n")
        file.write(",".join([str(idx) for idx in valid_split.indices]) + "\n")
        file.write(",".join([str(idx) for idx in test_split.indices]))

    # load data into datasets
    datasets = {
        "train": EuroSatRgbDatasetMultiLabel(train_split, indices_to_labels),
        "valid": EuroSatRgbDatasetMultiLabel(valid_split, indices_to_labels)
    }
    print("Number of samples:")
    print(f"Train: {len(datasets['train'])}")
    print(f"Validation: {len(datasets['valid'])}")

    datasets["train"].transform = data_transforms["flip"]["train"]
    datasets["valid"].transform = data_transforms["flip"]["valid"]

    # load data into dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "valid": torch.utils.data.DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False)
    }

    best_model = {"model": None, "param": None, "epoch": None, "measure": None, "weights": None}

    # loop through learning rates
    for lr in lrates:
        print(f"\nTraining model... (transform: flip, lr: {lr})")

        # clear gpu cache
        torch.cuda.empty_cache() if device == 'cuda' else None

        # load model
        euro_sat_rgb_model_multi_label = EuroSatRgbModelMultiLabel(model=models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT),
                                                                 device=device,
                                                                 n_classes=n_classes,
                                                                 criterion=loss,
                                                                 lr=lr,
                                                                 epochs=epochs)

        # fit model
        best_epoch, best_measure, best_weights = euro_sat_rgb_model_multi_label.fit(dataloaders["train"], dataloaders["valid"])

        # init best param, best measure and best weights, replace if new measure is better
        if best_model["measure"] is None or best_measure > best_model["measure"]:
            best_model["model"] = euro_sat_rgb_model_multi_label
            best_model["param"] = lr
            best_model["epoch"] = best_epoch
            best_model["measure"] = best_measure
            best_model["weights"] = best_weights

    # save best model
    torch.save(best_model["weights"], os.path.join(save_dir, "rgb_multilabel_model.pt"))

    with open(os.path.join(save_dir, "rgb_multilabel_params.txt"), "w+") as file:
        file.write("transform,parameter,epoch\n")
        file.write(",".join(["flip", str(best_model["param"]), str(best_model["epoch"])]))

    # save losses
    with open(os.path.join(save_dir, "rgb_multilabel_train_losses.txt"), "w+") as file:
        file.write(",".join(map(str, best_model["model"].train_losses)))

    with open(os.path.join(save_dir, "rgb_multilabel_valid_losses.txt"), "w+") as file:
        file.write(",".join(map(str, best_model["model"].valid_losses)))


if __name__ == "__main__":
    run("data/EuroSAT_RGB")
