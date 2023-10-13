import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchvision import models, transforms as T
from torchvision.models import EfficientNet_B0_Weights

from modules.EuroSAT_RGB_dataset import load_data, EuroSatRgbDataset
from modules.EuroSAT_RGB_evaluator import EuroSatRgbEvaluator
from modules.EuroSAT_RGB_model import EuroSatRgbModel


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
weights = torch.load("rgb_multiclass_model.pt")
n_classes = len(indices_to_labels)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# define transform
data_transform = T.Compose([
    T.Resize(64),
    T.CenterCrop(60),
    T.ToTensor(),
    T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])
])
"""
    Split dataset into train, validation and test splits and load into batches.
"""
print("Loading dataset...")
data = load_data("EuroSAT_RGB")
print(f"Total size: {len(data)}\n")

# split data
print("Splitting dataset...")
train_split, valid_split, test_split = random_split(data, (0.7, 0.15, 0.15))

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
with open("rgb_multiclass_params.txt") as file:
    transform, lr, epoch = file.readline().strip().split(",")

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

    # calculate each class's average precision measure
    mean_avg_precision, classes_avg_precision = euro_sat_rgb_evaluator.avg_precision_by_class(one_hot_preds, labels)

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
with open("rgb_multiclass_train_losses.txt") as file:
    train_losses = [float(loss) for loss in file.readline().strip().split(",")]
with open("rgb_multiclass_valid_losses.txt") as file:
    valid_losses = [float(loss) for loss in file.readline().strip().split(",")]

figure, axes = plt.subplots(1, 2, figsize=(13, 5))

# plot train and validation losses
sns.lineplot(x=range(len(train_losses)),
             y=train_losses,
             ax=axes[0])
sns.lineplot(x=range(len(valid_losses)),
             y=valid_losses,
             ax=axes[1])

axes[0].set_title("Best Model's Average Train Losses Over Epochs")
axes[1].set_title("Best Model's Average Validation Losses Over Epochs")

plt.show()
