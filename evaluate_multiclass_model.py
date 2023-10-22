import os

import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchvision import models, transforms as T
from torchvision.models import EfficientNet_B0_Weights

from modules.EuroSAT_RGB_dataset import load_data, EuroSatRgbDataset, decode_labels
from modules.EuroSAT_RGB_evaluator import EuroSatRgbEvaluator
from modules.EuroSAT_RGB_model import EuroSatRgbModel


def run(data_path):
    """
        Setup for evaluation.
    """
    # make results reproducible
    np.random.seed(0)
    torch.manual_seed(0)

    # define class mapping for all classes
    indices_to_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation",
                         "Highway", "Industrial", "Pasture",
                         "PermanentCrop", "Residential", "River", "SeaLake"]

    # define model configurations
    batch_size = 64
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    n_classes = len(indices_to_labels)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    save_dir = "rgb_multiclass_save_files"
    weights = torch.load(os.path.join(save_dir, "rgb_multiclass_model.pt"), map_location=device)

    # define transform
    data_transform = T.Compose([
        T.Resize(64),
        T.CenterCrop(60),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])
    ])

    """
        Split dataset into train, validation and test splits and load into batches.
    """
    print("Loading validation and test dataset...")
    data = load_data(data_path)

    with open(os.path.join(save_dir, "rgb_multiclass_datasets_split.txt")) as file:
        dataset_splits = file.readlines()
        valid_split = [data[int(idx)] for idx in dataset_splits[1].strip().split(",")]
        test_split = [data[int(idx)] for idx in dataset_splits[2].strip().split(",")]

    # load data into datasets
    datasets = {
        "valid": EuroSatRgbDataset(valid_split, indices_to_labels, data_transform),
        "test": EuroSatRgbDataset(test_split, indices_to_labels, data_transform)
    }
    print("Number of samples:")
    print(f"Validation: {len(datasets['valid'])}")
    print(f"Test: {len(datasets['test'])}")

    # load data into batches with updated transforms
    dataloaders = {
        "valid": torch.utils.data.DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False),
        "test": torch.utils.data.DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)
    }

    """
        Initialise model and evaluator.
    """
    with open(os.path.join(save_dir, "rgb_multiclass_params.txt")) as file:
        lines = file.readlines()
        transform, lr, epoch = lines[1].strip().split(",")

    print("\nLoading best model...")
    print(f"Transform: {transform}")
    print(f"Learning Rate: {lr}")
    print(f"Epoch: {epoch}")

    euro_sat_rgb_model = EuroSatRgbModel(model=model,
                                         device=device,
                                         n_classes=n_classes,
                                         weights=weights)

    """
        Evaluate model on validation and test datasets.
    """
    euro_sat_rgb_evaluator = EuroSatRgbEvaluator(indices_to_labels)

    print(f"\nEvaluating best model...")
    for dl_name, dataloader in dataloaders.items():
        # predict validation data
        logits, labels = euro_sat_rgb_model.predict_batches(dataloader)
        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(probs, dim=1)

        # one-hot encode predictions
        one_hot_preds = torch.zeros(probs.shape)
        for idx, pred in enumerate(one_hot_preds):
            one_hot_preds[idx, preds[idx]] = 1.

        # save labels and predictions to file
        with open(os.path.join(save_dir, f"rgb_multiclass_{dl_name}_predictions.txt"), "w+") as file:
            file.write("img_path;labels;predictions\n")

            for idx, pred in enumerate(one_hot_preds):
                label = ",".join(decode_labels(labels[idx], indices_to_labels))
                decoded_pred = ",".join(decode_labels(pred, indices_to_labels))
                if dl_name == "valid":
                    img_file = os.path.split(valid_split[idx]["img_path"])[1]
                else:
                    img_file = os.path.split(test_split[idx]["img_path"])[1]

                file.write(";".join([img_file, label, decoded_pred]) + "\n")

        # calculate each class's average precision measure
        mean_avg_precision, classes_avg_precision = euro_sat_rgb_evaluator.avg_precision_by_class(probs.to("cpu"), labels)

        # calculate each class's accuracy score
        mean_accuracy, classes_accuracy = euro_sat_rgb_evaluator.accuracy_by_class(one_hot_preds, labels)

        print(f"\nEvaluation Results on {dl_name} dataset:")
        print(f"{'Class':<25} {'Average Precision':<25} {'Accuracy':<15}\n")
        for idx, avg_precision in enumerate(classes_avg_precision):
            print(f"{indices_to_labels[idx]:<25} {avg_precision:<25} {classes_accuracy[idx]:<15}")
        print(f"\n{'Mean':<25} {mean_avg_precision:<25} {mean_accuracy:<15}")

    """
        Plot training and validation losses from saved files
    """
    # read in losses data
    with open(os.path.join(save_dir, "rgb_multiclass_train_losses.txt")) as file:
        train_losses = [float(loss) for loss in file.readline().strip().split(",")]
    with open(os.path.join(save_dir, "rgb_multiclass_valid_losses.txt")) as file:
        valid_losses = [float(loss) for loss in file.readline().strip().split(",")]

    figure, ax = plt.subplots(1, 1, figsize=(10, 7))

    # plot train and validation losses
    sns.lineplot(x=range(len(train_losses)),
                 y=train_losses,
                 ax=ax, label="Train loss")
    sns.lineplot(x=range(len(valid_losses)),
                 y=valid_losses,
                 ax=ax, label="Valid loss")

    ax.set_title("Best Model's Average Losses Over Epochs")

    plt.show()


if __name__ == "__main__":
    run("data/EuroSAT_RGB")
