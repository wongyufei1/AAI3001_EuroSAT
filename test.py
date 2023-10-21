import os
import csv
# import rasterio
# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
# import seaborn as sn
# import pandas as pd
# from IPython.display import clear_output
# import torch
# from torch.nn import Conv2d, Linear
# import torchvision.models as models
# from torchsummary import summary
# import torch.optim as optim
# import torch.utils.data as data
# from torch.utils.data import DataLoader

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

path = "../EuroSAT_multispectral/EuroSAT/train.csv"

file = open(path)

csvreader = csv.reader(file)

i = 0

for row in csvreader:
	print(row)
	i += 1

	if i == 2:
		exit()


# import rasterio
# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
# import seaborn as sn
# import pandas as pd
# from IPython.display import clear_output
# import torch
# from torch.nn import Conv2d, Linear
# import torchvision.models as models
# from torchsummary import summary
# import torch.optim as optim
# import torch.utils.data as data
# from torch.utils.data import DataLoader

# import torch
# from torchvision.models import resnet50, efficientnet_b0
# from torchsummary import summary
# from torchvision.models import EfficientNet_B0_Weights

# # Load the pre-trained EfficientNet model
# model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)


datapath = "../EuroSAT_multispectral/EuroSATallBands/AnnualCrop"

# indices_to_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation",
#                      "Highway", "Industrial", "Pasture",
#                      "PermanentCrop", "Residential", "River", "SeaLake"]


for label in indices_to_labels:
    print(label)
    label_path = f"{datapath}/{label}"
    path_to_tiffs = glob.glob(os.path.join(label_path, "*.tif"))

    for path in path_to_tiffs[0]:
        tiff_path = path

        # Open the TIFF File in read only mode using Rasterio

        with rasterio.open(tiff_path, "r") as src:
            # Note that band numbering in rasterio starts at 1, not 0
            band_nums = [2, 6, 8, 11]

            # Read the data from the specified bands (2, 3, 4, 8) and store it in a variable (we will only work with the four 10m channels)
            tempdata = src.read((2, 3, 4, 8))

            # Print the shape of the data array
            print("Shape is ", tempdata.shape)

    # Read the data from the specified bands (2, 3, 4, 8) and store it in a variable (we will only work with the four 10m channels)


# # Find all the TIFF files in the directory and its subdirectories
# path_to_tiffs = glob.glob(os.path.join(
#     datapath, "**", "*.tif"), recursive=True)

# # Choose the first TIFF file as an example
# tiff_file = path_to_tiffs[0]

# # Open the TIFF file in read-only mode using Rasterio
# with rasterio.open(tiff_file, "r") as src:

#     band_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#     # Read the data from the specified bands (2, 3, 4, 8) and store it in a variable (we will only work with the four 10m channels)
#     data = src.read(band_nums)

#     # Assuming data is the array with shape (4, 64, 64)
#     channel_1 = np.expand_dims(data[0], axis=0)  # Shape (1, 64, 64)
#     channel_2 = np.expand_dims(data[1], axis=0)  # Shape (1, 64, 64)
#     channel_3 = np.expand_dims(data[2], axis=0)  # Shape (1, 64, 64)
#     channel_4 = np.expand_dims(data[3], axis=0)  # Shape (1, 64, 64)

#     # Verify the shapes of the channels
#     print("Shape of channel 1:", channel_1.shape)
#     print("Shape of channel 2:", channel_2.shape)
#     print("Shape of channel 3:", channel_3.shape)
#     print("Shape of channel 4:", channel_4.shape)

#     # Print the shape of the data array
#     print("Shape is ", data.shape)

#     print("Band 1 Shaoe is ", data[0].shape)

#     plt.imshow(data[0])
#     plt.title("Band {}".format(4))

#     plt.show()


# for i, band in enumerate(data, start=1):
#     # Get the band number from the source file
#     with rasterio.open(tiff_file, "r") as src:
#         band_num = src.indexes[band_nums[i-1]-1]

#     # Plot the band and set the title to the band number
#     plt.imshow(band)
#     plt.title("Band {}".format(band_num))

#     # Show the plot in the notebook
#     plt.show()


