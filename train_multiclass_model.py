import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import models, transforms as T
from torchvision.models import EfficientNet_B0_Weights

from modules.EuroSAT_RGB_dataset import EuroSatRgbDataset, load_data
from modules.EuroSAT_RGB_model import EuroSatRgbModel


def train_model():
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
    loss = torch.nn.CrossEntropyLoss()
    lrates = [0.1, 0.01, 0.001]
    n_classes = len(indices_to_labels)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # define transforms
    data_transforms = {
        "train": T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.3458, 0.3816, 0.4091], [0.2022, 0.1354, 0.1136])
        ]),
        "valid": T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.3458, 0.3816, 0.4091], [0.2022, 0.1354, 0.1136])
        ])
    }

    """
        Split dataset into train, validation and test splits and load into batches
    """
    print("Loading dataset...")
    data = load_data("EuroSAT_RGB")
    print(f"Total size: {len(data)}\n")

    # split data
    print("Splitting dataset...")
    train_split, valid_split, test_split = random_split(data, (0.7, 0.15, 0.15))

    # load data into datasets
    datasets = {
        "train": EuroSatRgbDataset(train_split, indices_to_labels, data_transforms["train"]),
        "valid": EuroSatRgbDataset(valid_split, indices_to_labels, data_transforms["valid"])
    }
    print("Number of samples:")
    print(f"Train: {len(datasets['train'])} Validation: {len(datasets['valid'])}")

    # load data into batches
    dataloaders = {
        "train": torch.utils.data.DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "valid": torch.utils.data.DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False)
    }
    print("Number of batches:")
    print(f"Train: {len(dataloaders['train'])} Validation: {len(dataloaders['valid'])}")

    """
        Calculate mean and standard deviation for normalising image dataset.
        
        Mean = [0.3458, 0.3816, 0.4091]
        Std = [0.2022, 0.1354, 0.1136]
    """
    # imgs = torch.stack([img for img, label in datasets["train"]])
    #
    # mean = torch.mean(imgs, dim=(0, 2, 3))
    # std = torch.std(imgs, dim=(0, 2, 3))
    # print(mean)
    # print(std)

    best_model = {"model": None, "param": None, "epoch": None, "measure": None, "weights": None}

    for lr in lrates:
        print(f"\nTraining model... (lr: {lr})")

        # clear gpu cache
        torch.cuda.empty_cache() if device == "cuda" else None

        euro_sat_rgb_model = EuroSatRgbModel(model=models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT),
                                             device=device,
                                             n_classes=n_classes,
                                             criterion=loss,
                                             lr=lr,
                                             epochs=epochs)

        best_epoch, best_measure, best_weights = euro_sat_rgb_model.fit(dataloaders["train"], dataloaders["valid"])

        # init best param, best measure and best weights, replace if new measure is better
        if best_model["measure"] is None or best_measure > best_model["measure"]:
            best_model["model"] = euro_sat_rgb_model
            best_model["param"] = lr
            best_model["epoch"] = best_epoch
            best_model["measure"] = best_measure
            best_model["weights"] = best_weights

        # save weights
        torch.save(best_model["weights"], "rgb_multiclass_model.pt")

    # save losses
    with open("rgb_multiclass_train_losses.txt", "w+") as file:
        file.write(",".join(map(str, best_model["model"].train_losses)))

    with open("rgb_multiclass_valid_losses.txt", "w+") as file:
        file.write(",".join(map(str, best_model["model"].valid_losses)))


if __name__ == "__main__":
    train_model()
