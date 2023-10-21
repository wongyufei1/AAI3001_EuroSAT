# Import Statements
import os

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import models, transforms as T
from torchvision.models import EfficientNet_B0_Weights

# Import resnet18
from torchvision.models import resnet18

from modules.EuroSAT_RGB_dataset import EuroSatRgbDataset, EuroSatHyperSpectralDataset, load_data, load_data_hyperspectral, load_split_data, extract_bands
from modules.EuroSAT_RGB_evaluator import EuroSatRgbEvaluator
from modules.EuroSAT_RGB_model import EuroSatRgbModel, HyperspectralModel

# Make results reproducible
np.random.seed(0)
torch.manual_seed(0)

# Define class mapping
indices_to_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation",
                     "Highway", "Industrial", "Pasture",
                     "PermanentCrop", "Residential", "River", "SeaLake"]

# Define configurations (Same for RGB and Tiff)
batch_size = 64
epochs = 10
loss = torch.nn.CrossEntropyLoss()
lrates = [0.001] # TODO: Add 0.01 later
n_classes = len(indices_to_labels)
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
save_dir = "hyperspectral_multiclass_save_files"

# Define transform for RGB and Tiff, normalise will have to be based on the calculate mean and SD
data_transforms = {
    "rgb": {
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
    },
	"hyper": {
		"train": T.Compose([
			T.Resize(64),
			T.ToTensor(),
			T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])]),
		"valid": T.Compose([
			T.Resize(64),
			T.ToTensor(),
			T.Normalize([0.3450, 0.3809, 0.4084], [0.2038, 0.1370, 0.1152])]),
	}
}

# Split dataset into train, validation and test splits and load them into batchs for RGB
print("Loading dataset...")
rgb_train, rgb_val, rgb_test, hyper_train, hyper_val, hyper_test = load_split_data("../EuroSAT_multispectral")

# Load data for TIFF, split them into 4 different data sets and do splits for train, val and test
datasets_rgb = {
	"train": EuroSatRgbDataset(rgb_train, indices_to_labels),
	"valid": EuroSatRgbDataset(rgb_val, indices_to_labels),
}

datasets = {
	"rgb": {
		"train": EuroSatRgbDataset(rgb_train, indices_to_labels),
		"valid": EuroSatRgbDataset(rgb_val, indices_to_labels),
	},
	"hyper": {
		"train": EuroSatHyperSpectralDataset(hyper_train, indices_to_labels),
		"valid": EuroSatHyperSpectralDataset(hyper_val, indices_to_labels),
	}
}

datasets['rgb']['train'].transform = data_transforms['rgb']['train']
datasets['rgb']['valid'].transform = data_transforms['rgb']['valid']
datasets['hyper']['train'].transform = data_transforms['hyper']['train']
datasets['hyper']['valid'].transform = data_transforms['hyper']['valid']

# Load the data into dataloaders
dataloaders = {
	"rgb": {
		"train": torch.utils.data.DataLoader(datasets['rgb']['train'], batch_size=batch_size, shuffle=True),
		"valid": torch.utils.data.DataLoader(datasets['rgb']['valid'], batch_size=batch_size, shuffle=True),
	},
	"hyper": {
		"train": torch.utils.data.DataLoader(datasets['hyper']['train'], batch_size=batch_size, shuffle=True),
		"valid": torch.utils.data.DataLoader(datasets['hyper']['valid'], batch_size=batch_size, shuffle=True),
	}
}

# Train Custom Model that concatenates all the model
# clear gpu cache
torch.cuda.empty_cache() if device == 'cuda' else None

# Load model
euro_hyper_model = HyperspectralModel(model=resnet18(pretrained=True),
									 device=device,
									 n_classes=n_classes,
									 criterion=loss,
									 lr=lrates[0],
									 epochs=epochs)

# Fit model
best_epoch, best_measure, best_weights = euro_hyper_model.fit(dataloaders['hyper']["train"], dataloaders['hyper']["valid"])



"""
Model to train
EfficientNet for RGB, last layer don't classify, return feature map
EfficientNet for Tiff as well but need train on all 13 channels.
Change the first and last layer to fit the dataset

Concatenate the feature maps from both models and pass it through a linear layer to classify

"""

# Train the model

# Save the model
