# Import Statements
import os

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import models, transforms as T
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights

# Import resnet18
from torchvision.models import resnet18

from modules.EuroSAT_RGB_dataset import EuroSatRgbDataset, EuroSatHyperSpectralDataset, load_split_data
from modules.EuroSAT_RGB_model import HyperspectralModel, RGBEfficientNetModel, RGB_HyperSpectral_Model


def run(data_path):
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

	# Calculated mean and SD for HyperSpectral data, not sure if this is correct
	hyper_mean = [1353.9527587890625, 1115.400390625, 1033.31103515625, 934.7520751953125, 1180.4534912109375, 1964.8804931640625, 2326.806884765625, 2254.523193359375, 723.2532348632812, 13.145559310913086, 1780.3253173828125, 1097.9527587890625, 2543.12353515625]
	hyper_std = [243.3072967529297, 330.1734619140625, 395.2242126464844, 592.9466552734375, 574.9304809570312, 885.2379150390625, 1113.62060546875, 1142.745849609375, 404.9068298339844, 9.187087059020996, 1026.2681884765625, 764.8196411132812, 1267.559814453125]

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
				T.Normalize(hyper_mean, hyper_std)]),
			"valid": T.Compose([
				T.Resize(64),
				T.Normalize(hyper_mean, hyper_std)]),
		}
	}

	# Load data based on existing splits
	print("Loading dataset...")
	rgb_train, rgb_val, rgb_test, hyper_train, hyper_val, hyper_test = load_split_data(data_path)

	print("Number of samples:")
	print("RGB Train: ", len(rgb_train))
	print("RGB Val: ", len(rgb_val))
	print("RGB Test: ", len(rgb_test))
	print("Hyper Train: ", len(hyper_train))
	print("Hyper Val: ", len(hyper_val))
	print("Hyper Test: ", len(hyper_test))

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

	best_model = {"model": None, "rgb_model": None, "hyper_model": None, "param": None, "epoch": None, "measure": None, "rgb_weights": None, "hyper_weights": None}

	# Train Custom Model that concatenates all the model
	# clear gpu cache
	torch.cuda.empty_cache() if device == 'cuda' else None

	# Load model
	euro_rgb_model = RGBEfficientNetModel(model=models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT),
									   device=device,
									   n_classes=n_classes,
									   criterion=loss,
									   lr=lrates[0],
									   epochs=epochs)

	euro_hyper_model = HyperspectralModel(model=resnet18(weights=ResNet18_Weights.DEFAULT),
										 device=device,
										 n_classes=n_classes,
										 criterion=loss,
										 lr=lrates[0],
										 epochs=epochs)

	# Combined model
	euro_combined_model = RGB_HyperSpectral_Model(rgb_model=euro_rgb_model.get_model(), hyper_model=euro_hyper_model.get_model(), device=device, n_classes=n_classes, criterion=loss, lr=lrates[0], epochs=epochs)

	# Fit model
	best_epoch, best_measure, rgb_best_weights, hyper_best_weights = euro_combined_model.fit(dataloaders['rgb']["train"], dataloaders['hyper']["train"], dataloaders['rgb']["valid"], dataloaders['hyper']["valid"])

	# Save model
	best_model["model"] = euro_combined_model
	best_model["rgb_model"] = euro_combined_model.get_rgb_model()
	best_model["hyper_model"] = euro_combined_model.get_hyper_model()
	best_model["param"] = lrates[0]
	best_model["epoch"] = best_epoch
	best_model["measure"] = best_measure
	best_model["rgb_weights"] = rgb_best_weights
	best_model["hyper_weights"] = hyper_best_weights

	# If Directory hyperspectral_multiclass_save_files does not exist, create it
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Save model
	torch.save(best_model['rgb_weights'], os.path.join(save_dir, "rgb_combined_model.pt"))
	torch.save(best_model['hyper_weights'], os.path.join(save_dir, "hyper_combined_model.pt"))

	with open(os.path.join(save_dir, "hyper_multiclass_params.txt"), "w+") as file:
		file.write("transform,parameter,epoch\n")
		file.write(",".join(["flip", str(best_model["param"]), str(best_model["epoch"])]))

	# save losses
	with open(os.path.join(save_dir, "hyper_multiclass_train_losses.txt"), "w+") as file:
		file.write(",".join(map(str, best_model["model"].train_losses)))

	with open(os.path.join(save_dir, "hyper_multiclass_valid_losses.txt"), "w+") as file:
		file.write(",".join(map(str, best_model["model"].valid_losses)))


if __name__ == "__main__":
	run("data/EuroSAT_multispectral")