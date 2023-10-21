import torch
import torch.nn as nn
import torchvision.models as models

# Assuming you have two models that are sliced at their second-to-last layers
model_rgb = models.vgg16(pretrained=True)
model_rgb = torch.nn.Sequential(*(list(model_rgb.children())[:-1]))  # Sliced at the second-to-last layer

model_hyperspectral = models.resnet18(pretrained=True)
model_hyperspectral = torch.nn.Sequential(*(list(model_hyperspectral.children())[:-1]))  # Sliced at the second-to-last layer

# Example input images
input_image_rgb = torch.randn(1, 3, 224, 224)  # Example with a 224x224 RGB image
input_image_hyperspectral = torch.randn(1, 3, 224, 224)  # Example with a 224x224 hyperspectral image

# Pass the input images through the sliced models to get the features
features_rgb = model_rgb(input_image_rgb)
features_hyperspectral = model_hyperspectral(input_image_hyperspectral)

# Flatten the feature maps
features_rgb = torch.flatten(features_rgb, 1)
features_hyperspectral = torch.flatten(features_hyperspectral, 1)

# Concatenate the features
concatenated_features = torch.cat((features_rgb, features_hyperspectral), dim=1)

# Define the fully connected layers for prediction
num_features_rgb = features_rgb.size(1)
num_features_hyperspectral = features_hyperspectral.size(1)
num_classes = 10  # Adjust num_classes according to your problem

linear_layer = nn.Linear(num_features_rgb + num_features_hyperspectral, num_classes)

# Pass the concatenated features through the linear layer
output = linear_layer(concatenated_features)

# Get softmax predictions
softmax_predictions = nn.functional.softmax(output, dim=1)

print(softmax_predictions)

# Apply any further activation or loss function based on your specific task


# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
#
# path = "../EuroSAT_multispectral/EuroSATallBands/AnnualCrop/AnnualCrop_2750.tif"
# path_rgb = "../EuroSAT_multispectral/EuroSAT/AnnualCrop/AnnualCrop_2750.jpg"
# bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#
# img_rgb = Image.open(path_rgb).convert("RGB")
# print(type(img_rgb))
#
# print(img_rgb.size)
#
#
# with rasterio.open(path) as dataset:
# 	img = dataset.read(bands)
#
# 	print(img.shape)
#
# 	# Convert img[0] into a PIL image
# 	img_0 = Image.fromarray(img[0])
#
# 	# Plot the image
# 	plt.imshow(img_0)
# 	plt.show()
#
#
# # Compare types of img and img_rgb
# print(type(img_0))
# print(type(img_rgb))

"""

# Group all the 1st bands together, 2nd bands together etc.
for element in hyper_train:
	img_path = element['img_path']
	imgs = extract_bands(img_path)

	band_list = []

	for index, img in enumerate(imgs):
		datasets_hyper['train'][f"band{index + 1}"].append({
			'img': img,
			'label': element['label']
		})

for element in hyper_val:
	img_path = element['img_path']
	imgs = extract_bands(img_path)

	band_list = []

	for index, img in enumerate(imgs):
		datasets_hyper['valid'][f"band{index + 1}"].append({
			'img': img,
			'label': element['label']
		})

# Convert the list of dicts to a list of EuroSatHyperSpectralDataset
for key, value in datasets_hyper.items():
	for band, data in value.items():
		datasets_hyper[key][band] = EuroSatHyperSpectralDataset(data, indices_to_labels)

print("Number of RGB samples:")
print(f"Train: {len(datasets_rgb['train'])}")
print(f"Valid: {len(datasets_rgb['valid'])}")

print("Number of Hyper samples:")
for key, value in datasets_hyper.items():
	for band, data in value.items():
		print(f"{band}: {len(data)}")

datasets_rgb["train"].transform = data_transforms["rgb"]["train"]
datasets_rgb["valid"].transform = data_transforms["rgb"]["valid"]

for key, value in datasets_hyper.items():
	for band, data in value.items():
		datasets_hyper[key][band].transform = data_transforms["hyper"][key]


"""