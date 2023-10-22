import os

import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchvision import models, transforms as T
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights
# Import resnet18
from torchvision.models import resnet18

from modules.EuroSAT_RGB_dataset import load_data, EuroSatRgbDataset, decode_labels, load_split_data, EuroSatHyperSpectralDataset
from modules.EuroSAT_RGB_evaluator import EuroSatRgbEvaluator
from modules.EuroSAT_RGB_model import EuroSatRgbModel, RGB_HyperSpectral_Model, RGBEfficientNetModel, HyperspectralModel


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
save_dir = "hyperspectral_multiclass_save_files"
rgb_weights = torch.load(os.path.join(save_dir, "rgb_combined_model.pt"), map_location=device)
hyper_weights = torch.load(os.path.join(save_dir, "hyper_combined_model.pt"), map_location=device)
ROOT_DIR = "../EuroSAT_multispectral" # Change this to the correct path

# Calculate mean and std for each channel
hyper_mean = [1353.9527587890625, 1115.400390625, 1033.31103515625, 934.7520751953125, 1180.4534912109375, 1964.8804931640625, 2326.806884765625, 2254.523193359375, 723.2532348632812, 13.145559310913086, 1780.3253173828125, 1097.9527587890625, 2543.12353515625]
hyper_std = [243.3072967529297, 330.1734619140625, 395.2242126464844, 592.9466552734375, 574.9304809570312, 885.2379150390625, 1113.62060546875, 1142.745849609375, 404.9068298339844, 9.187087059020996, 1026.2681884765625, 764.8196411132812, 1267.559814453125]

# define transform
rgb_data_transform = T.Compose([
    T.Resize(64),
    T.CenterCrop(60),
    T.ToTensor(),
    T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])
])

hyper_data_transform = T.Compose([
    T.Resize(64),
    T.Normalize(hyper_mean, hyper_std)
])


"""
    Load data based on existing splits
"""
print("Loading validation and test dataset...")
rgb_train, rgb_val, rgb_test, hyper_train, hyper_val, hyper_test = load_split_data(ROOT_DIR)

# Change the Hyper Test and Hyper Val to be the same size as the RGB Test and RGB Val, shuffle first?
hyper_test = hyper_test[:len(rgb_test)]
hyper_val = hyper_val[:len(rgb_val)]

# load data into datasets
datasets = {
	"rgb": {
		"test": EuroSatRgbDataset(rgb_test, indices_to_labels),
		"valid": EuroSatRgbDataset(rgb_val, indices_to_labels),
	},
	"hyper": {
		"test": EuroSatHyperSpectralDataset(hyper_test, indices_to_labels),
		"valid": EuroSatHyperSpectralDataset(hyper_val, indices_to_labels),
	}

}


print("Number of samples:")
print(f"RGB Test: {len(rgb_test)}")
print(f"RGB Valid: {len(rgb_val)}")
print(f"Hyper Test: {len(hyper_test)}")
print(f"Hyper Valid: {len(hyper_val)}")


datasets['rgb']['test'].transform = rgb_data_transform
datasets['rgb']['valid'].transform = rgb_data_transform
datasets['hyper']['test'].transform = hyper_data_transform
datasets['hyper']['valid'].transform = hyper_data_transform

# load data into batches with updated transforms
dataloaders = {
    "test":
        {
            "rgb": torch.utils.data.DataLoader(datasets['rgb']['test'], batch_size=batch_size, shuffle=False),
            "hyper": torch.utils.data.DataLoader(datasets['hyper']['test'], batch_size=batch_size, shuffle=False)
        },
    "valid":
        {
            "rgb": torch.utils.data.DataLoader(datasets['rgb']['valid'], batch_size=batch_size, shuffle=False),
            "hyper": torch.utils.data.DataLoader(datasets['hyper']['valid'], batch_size=batch_size, shuffle=False)
        }
}

"""
    Initialise model and evaluator.
"""
with open(os.path.join(save_dir, "hyper_multiclass_params.txt")) as file:
    lines = file.readlines()
    transform, lr, epoch = lines[1].strip().split(",")

print("\nLoading best model...")
print(f"Transform: {transform}")
print(f"Learning Rate: {lr}")
print(f"Epoch: {epoch}")


euro_rgb_model = RGBEfficientNetModel(model=models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT),
								   device=device,
								   n_classes=n_classes)


euro_hyper_model = HyperspectralModel(model=resnet18(weights=ResNet18_Weights.DEFAULT),
									 device=device,
									 n_classes=n_classes)


euro_combined_model = RGB_HyperSpectral_Model(rgb_model=euro_rgb_model.get_model(),
                                              hyper_model=euro_hyper_model.get_model(),
                                              device=device,
                                              n_classes=n_classes,
                                              rgb_weights=rgb_weights,
                                              hyper_weights=hyper_weights)


"""
    Evaluate model on validation and test datasets.
"""
euro_sat_combined_evaluator = EuroSatRgbEvaluator(indices_to_labels)

print(f"\nEvaluating best model...")
for dl_name, dataloader in dataloaders.items():
    # predict validation data
    logits, labels = euro_combined_model.predict_batches(dataloader['rgb'], dataloader['hyper'])
    probs = torch.softmax(logits, dim=1)
    _, preds = torch.max(probs, dim=1)

    # one-hot encode predictions
    one_hot_preds = torch.zeros(probs.shape)
    for idx, pred in enumerate(one_hot_preds):
        one_hot_preds[idx, preds[idx]] = 1.

    # save labels and predictions to file
    with open(os.path.join(save_dir, f"hyper_multiclass_{dl_name}_predictions.txt"), "w+") as file:
        file.write("img_path;labels;predictions\n")

        for idx, pred in enumerate(one_hot_preds):
            label = ",".join(decode_labels(labels[idx], indices_to_labels))
            decoded_pred = ",".join(decode_labels(pred, indices_to_labels))
            if dl_name == "valid":
                img_file = os.path.split(rgb_val[idx]["img_path"])[1]
            else:
                img_file = os.path.split(rgb_test[idx]["img_path"])[1]

            file.write(";".join([img_file, label, decoded_pred]) + "\n")

    # calculate each class's average precision measure
    mean_avg_precision, classes_avg_precision = euro_sat_combined_evaluator.avg_precision_by_class(probs.to("cpu"), labels)

    # calculate each class's accuracy score
    mean_accuracy, classes_accuracy = euro_sat_combined_evaluator.accuracy_by_class(one_hot_preds, labels)

    print(f"\nEvaluation Results on {dl_name} dataset:")
    print(f"{'Class':<25} {'Average Precision':<25} {'Accuracy':<15}\n")
    for idx, avg_precision in enumerate(classes_avg_precision):
        print(f"{indices_to_labels[idx]:<25} {avg_precision:<25} {classes_accuracy[idx]:<15}")
    print(f"\n{'Mean':<25} {mean_avg_precision:<25} {mean_accuracy:<15}")

"""
    Plot training and validation losses from saved files
"""
# read in losses data
with open(os.path.join(save_dir, "hyper_multiclass_train_losses.txt")) as file:
    train_losses = [float(loss) for loss in file.readline().strip().split(",")]
with open(os.path.join(save_dir, "hyper_multiclass_valid_losses.txt")) as file:
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
