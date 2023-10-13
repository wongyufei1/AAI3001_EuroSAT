import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchvision import models, transforms as T
from torchvision.models import EfficientNet_B0_Weights

from modules.EuroSAT_RGB_dataset import EuroSatRgbDataset, load_data
from modules.EuroSAT_RGB_evaluator import EuroSatRgbEvaluator
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
    lrates = [0.01, 0.001]
    n_classes = len(indices_to_labels)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # define transforms
    data_transforms = {
        "flip": {
            "train": T.Compose([
                T.Resize(64),
                T.RandomCrop(60),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])]),
            "valid": T.Compose([
                T.Resize(64),
                T.CenterCrop(60),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])]),
        },
        "rotate": {
            "train": T.Compose([
                T.Resize(64),
                T.RandomCrop(60),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])]),
            "valid": T.Compose([
                T.Resize(64),
                T.CenterCrop(60),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])]),
        },
        "contrast": {
            "train": T.Compose([
                T.Resize(64),
                T.RandomCrop(60),
                T.RandomAutocontrast(),
                T.ToTensor(),
                T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])]),
            "valid": T.Compose([
                T.Resize(64),
                T.CenterCrop(60),
                T.RandomAutocontrast(),
                T.ToTensor(),
                T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])]),
        }
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

    # save splits
    with open("rgb_multiclass_datasets_split.txt", "w+") as file:
        file.write(",".join([str(idx) for idx in train_split.indices]) + "\n")
        file.write(",".join([str(idx) for idx in valid_split.indices]) + "\n")
        file.write(",".join([str(idx) for idx in test_split.indices]))

    # load data into datasets
    datasets = {
        "train": EuroSatRgbDataset(train_split, indices_to_labels),
        "valid": EuroSatRgbDataset(valid_split, indices_to_labels)
    }
    print("Number of samples:")
    print(f"Train: {len(datasets['train'])}")
    print(f"Validation: {len(datasets['valid'])}")

    """
        Calculate mean and standard deviation for normalising image dataset.
        
        Resize: 256 Crop: 224
        Mean = [0.3458, 0.3816, 0.4091]
        Std = [0.2022, 0.1354, 0.1136]
        
        Resize: 64 Crop: 60
        Mean = [0.3450, 0.3809, 0.4084]
        Std = [0.2038, 0.1370, 0.1152]
    """
    # imgs = torch.stack([img for img, label in datasets["train"]])
    #
    # mean = torch.mean(imgs, dim=(0, 2, 3))
    # std = torch.std(imgs, dim=(0, 2, 3))
    # print(mean)
    # print(std)
    # exit()

    transform_best_models = {
        "flip": {"model": None, "param": None, "epoch": None, "measure": None, "weights": None},
        "rotate": {"model": None, "param": None, "epoch": None, "measure": None, "weights": None},
        "contrast": {"model": None, "param": None, "epoch": None, "measure": None, "weights": None}
    }

    for t_label, transform in data_transforms.items():
        # update transform
        datasets["train"].transform = transform["train"]
        datasets["valid"].transform = transform["valid"]

        # load data into batches
        dataloaders = {
            "train": torch.utils.data.DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
            "valid": torch.utils.data.DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False)
        }

        for lr in lrates:
            print(f"\nTraining model... (transform: {t_label}, lr: {lr})")

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
            if transform_best_models[t_label]["measure"] is None or best_measure > transform_best_models[t_label]["measure"]:
                transform_best_models[t_label]["model"] = euro_sat_rgb_model
                transform_best_models[t_label]["param"] = lr
                transform_best_models[t_label]["epoch"] = best_epoch
                transform_best_models[t_label]["measure"] = best_measure
                transform_best_models[t_label]["weights"] = best_weights

    """
        Evaluate selected models and get best model and data augmentation
    """
    euro_sat_rgb_evaluator = EuroSatRgbEvaluator(indices_to_labels)
    best_model = {"model": None, "param": None, "epoch": None, "weights": None, "transform": None,
                  "accuracy": None, "precision": None}

    for t_label, t_model in transform_best_models.items():
        print(f"\nEvaluating models... (transform: {t_label})")

        # update transform and load data into batches
        datasets["valid"].transform = data_transforms[t_label]["valid"]
        valid_loader = torch.utils.data.DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False)

        # predict validation data
        logits, labels = t_model["model"].predict_batches(valid_loader)
        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(probs, dim=1)

        # one-hot encode predictions
        one_hot_preds = torch.zeros(probs.shape)
        for idx, pred in enumerate(one_hot_preds):
            one_hot_preds[idx, preds[idx]] = 1.

        # calculate each class's average precision measure
        mean_avg_precision, classes_avg_precision = euro_sat_rgb_evaluator.avg_precision_by_class(one_hot_preds,
                                                                                                  labels)
        # calculate each class's accuracy score
        mean_accuracy, classes_accuracy = euro_sat_rgb_evaluator.accuracy_by_class(one_hot_preds, labels)

        print(f"\n{'Class':<25} {'Average Precision':<25} {'Accuracy':<15}\n")
        for idx, avg_precision in enumerate(classes_avg_precision):
            print(f"{indices_to_labels[idx]:<25} {avg_precision:<25} {classes_accuracy[idx]:<15}")
        print(f"\n{'Mean':<25} {mean_avg_precision:<25} {mean_accuracy:<15}")

        # pick best model and transform
        if best_model["accuracy"] is None or mean_accuracy > best_model["accuracy"]:
            best_model["model"] = t_model["model"]
            best_model["param"] = t_model["param"]
            best_model["epoch"] = t_model["epoch"]
            best_model["weights"] = t_model["weights"]
            best_model["transform"] = t_label
            best_model["accuracy"] = mean_accuracy
            best_model["precision"] = mean_avg_precision

    """
        Save best model, parameters and losses
    """
    print("\nSaving best model...")
    print(f"Transform: {best_model['transform']}")
    print(f"Learning Rate: {best_model['param']}")
    print(f"Epoch: {best_model['epoch']}")

    # save weights
    torch.save(best_model["weights"], "rgb_multiclass_model.pt")

    with open("rgb_multiclass_params.txt", "w+") as file:
        file.write("transform,parameter,epoch\n")
        file.write(",".join([best_model["transform"], str(best_model["param"]), str(best_model["epoch"])]))

    with open("rgb_multiclass_train_losses.txt", "w+") as file:
        file.write(",".join(map(str, best_model["model"].train_losses)))

    with open("rgb_multiclass_valid_losses.txt", "w+") as file:
        file.write(",".join(map(str, best_model["model"].valid_losses)))


if __name__ == "__main__":
    train_model()
