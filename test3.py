from functools import partial
import torch
import torchvision.models as models

# Define a new class that inherits from the pre-trained model
class CustomResNet(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(CustomResNet, self).__init__()
        self.pretrained_model = pretrained_model
        # Add more custom layers if needed

    def forward(self, x):
        # Custom forward pass logic
        out_list = []
        for i in range(self.input_channels):
            out = self.pretrained_model.conv_layers[i](x[:, i].unsqueeze(1))
            out = F.relu(out)
            out = self.pretrained_model.pool(out)  # You may use pooling or other operations here
            out_list.append(out)

        # Concatenate the features
        concatenated = torch.cat(out_list, dim=1)

        # Process the concatenated features further
        processed_output = self.pretrained_model.fc_layers(concatenated)

        return processed_output

# Load the pre-trained ResNet model
pretrained_resnet = models.resnet18(pretrained=True)

# Use functools.partial to overwrite the forward function
custom_forward_resnet = partial(CustomResNet, pretrained_model=pretrained_resnet)

# Create an instance of the custom model with the modified forward function
custom_model = custom_forward_resnet()