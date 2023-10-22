import torch
from tqdm import tqdm
import torch.nn as nn


class EuroSatRgbModel:
    def __init__(self, model, device, n_classes, weights=None, criterion=None, lr=None, epochs=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.config_model(n_classes, weights)
        if lr is not None:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9)
        else:
            self.optimizer = None

        self.train_losses = []
        self.valid_losses = []
        self.valid_accuracies = []

    # load model with custom number of classes
    def config_model(self, out_classes, weights):
        """
        Change the last layer of the neural network to fit the number of predictable classes.
        Load trained weights if available. (for evaluating)

        :param out_classes: number of possible classes to be predicted
        :param weights: trained weights of the model
        :return: nil
        """
        in_feats = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_feats, out_classes)

        if weights is not None:
            self.model.load_state_dict(weights)

        self.model.to(self.device)

    def train(self, dataloader):
        # set to training mode
        self.model.train()

        epoch_losses = []

        datasize = 0
        avg_loss = 0

        for batch in tqdm(dataloader):
            # load inputs and labels to device
            inputs = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            # predict with model and calculate loss
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)

            # calculate gradient and back propagation
            self.optimizer.zero_grad()  # reset accumulated gradients
            loss.backward()  # compute new gradients
            self.optimizer.step()  # apply new gradients to change model parameters

            # calculate running avg loss
            avg_loss = (avg_loss * datasize + loss) / \
                (datasize + inputs.shape[0])
            epoch_losses.append(float(avg_loss))

            # update data size
            datasize += inputs.shape[0]

        return avg_loss, epoch_losses

    def evaluate(self, dataloader):
        # set to eval mode
        self.model.eval()

        # do not record computations for computing the gradient
        with torch.no_grad():
            epoch_losses = []
            epoch_accuracies = []

            datasize = 0
            avg_accuracy = 0
            avg_loss = 0

            for batch in tqdm(dataloader):
                # load inputs and labels to device
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                # predict with model
                outputs = self.model(inputs)

                # compute some losses over time
                loss = self.criterion(outputs, labels)
                avg_loss = (avg_loss * datasize + loss) / \
                    (datasize + inputs.shape[0])
                epoch_losses.append(float(avg_loss))

                # compute some accuracies over time
                _, preds = torch.max(torch.softmax(outputs, 1), 1)
                _, labels = torch.max(labels, 1)

                accuracy = torch.sum(preds == labels)
                avg_accuracy = (avg_accuracy * datasize +
                                accuracy) / (datasize + inputs.shape[0])
                epoch_accuracies.append(float(avg_accuracy))

                # update data size
                datasize += inputs.shape[0]

        return avg_loss, avg_accuracy, epoch_losses, epoch_accuracies

    def fit(self, train_loader, val_loader):
        best_measure = -1
        best_epoch = -1

        if self.epochs is None or self.criterion is None or self.optimizer is None:
            raise ValueError(
                "Missing parameters \"epochs/criterion/optimizer\"")

        for epoch in range(self.epochs):
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 10)

            # train and evaluate model performance
            train_loss, _ = self.train(train_loader)
            valid_loss, measure, _, _ = self.evaluate(val_loader)
            print(f"\nTrain Loss: {train_loss}")
            print(f"Valid Loss: {valid_loss}")
            print(f'Measure: {measure.item()}')

            # save metrics
            self.train_losses.append(float(train_loss))
            self.valid_losses.append(float(valid_loss))
            self.valid_accuracies.append(float(measure))

            # update best performing epoch and save model weights
            if measure > best_measure:
                print(f'Updating best measure: {best_measure} -> {measure}')
                best_epoch = epoch
                best_weights = self.model.state_dict()
                best_measure = measure

        return best_epoch, best_measure, best_weights

    def predict_batches(self, dataloader):
        # set to eval mode
        self.model.eval()

        # do not record computations for computing the gradient
        with torch.no_grad():
            logits = None
            labels = None

            for batch in tqdm(dataloader):
                # load inputs to device
                inputs = batch[0].to(self.device)

                outputs = self.model(inputs)

                # save predictions and labels
                if logits is None:
                    logits = outputs
                    labels = batch[1]
                else:
                    logits = torch.cat((logits, outputs), dim=0)
                    labels = torch.cat((labels, batch[1]), dim=0)

        return logits, labels


# EuroSatRgbModel for multi-label
class EuroSatRgbModelMultiLabel(EuroSatRgbModel):
    def __init__(self, model, device, n_classes, weights=None, criterion=None, lr=None, epochs=None):
        super().__init__(model, device, n_classes, weights, criterion, lr, epochs)

    # change the evaluation function to do sigmoid
    def evaluate(self, dataloader):
        # set to eval mode
        self.model.eval()

        # do not record computations for computing the gradient
        with torch.no_grad():
            epoch_losses = []
            epoch_accuracies = []

            datasize = 0
            avg_accuracy = 0
            avg_loss = 0

            for batch in tqdm(dataloader):
                # load inputs and labels to device
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                # predict with model
                outputs = self.model(inputs)

                # compute some losses over time
                loss = self.criterion(outputs, labels)
                avg_loss = (avg_loss * datasize + loss) / \
                    (datasize + inputs.shape[0])
                epoch_losses.append(float(avg_loss))

                # compute some accuracies over time (For multi-label), using sigmoid
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                accuracy = sum([(label == preds[idx]).all(dim=0)
                               for idx, label in enumerate(labels)])

                avg_accuracy = (avg_accuracy * datasize +
                                accuracy) / (datasize + inputs.shape[0])
                epoch_accuracies.append(float(avg_accuracy))

                # update data size
                datasize += inputs.shape[0]

        return avg_loss, avg_accuracy, epoch_losses, epoch_accuracies


# Custom hyperspectral model that has custom forward function
class HyperspectralModel(nn.Module):
    def __init__(self, model, device, n_classes, weights=None, criterion=None, lr=None, epochs=None):
        super(HyperspectralModel, self).__init__()

        self.model = model
        # Layers to be modified
        self.pooling = nn.AvgPool2d(2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(13 * 3 * 3, n_classes) # Change if needed
        self.conv1 = nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.device = device
        self.n_classes = n_classes
        self.weights = weights
        self.criterion = criterion
        self.lr = lr
        if lr is not None:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9)
        else:
            self.optimizer = None
        self.epochs = epochs

        self.config_model(n_classes, weights)

        self.train_losses = []
        self.valid_losses = []
        self.valid_accuracies = []

    def get_model(self):
        return self.model

    def config_model(self, out_classes, weights):
        # Change the first layer of resnet to accept 13 channels
        self.model.conv1 = self.conv1

        # Change the last fully connected layer to output 10 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, out_classes)

        if weights is not None:
            self.model.load_state_dict(weights)

        self.model.to(self.device)

    def forward(self, x):
        # x is a batch of images
        print("Forwarding")
        features = []
        for i in range(x.shape[1]): # Iterate over channels
            out = self.model(x[:, i, :, :].unsqueeze(1).expand(-1, 3, -1, -1))
            features.append(out)

        # Concatenate features
        concatenated = torch.cat(features, dim=1)
        relu = self.relu(concatenated)
        pooled = self.pooling(relu)

        return pooled

    def train(self, dataloader):
        # set to train mode
        self.model.train()

        epoch_losses = []

        datasize = 0
        avg_loss = 0

        # Print out the model first layer
        print('Model Conv1')
        print(self.model.conv1)

        for batch in tqdm(dataloader):
            # load inputs and labels to device
            inputs = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            # predict with model and calculate loss
            outputs = self.model(inputs.float())

            loss = self.criterion(outputs, labels)

            # calculate gradient and back propagation
            self.optimizer.zero_grad()  # reset accumulated gradients
            loss.backward()  # compute new gradients
            self.optimizer.step()  # apply new gradients to change model parameters

            # calculate running avg loss
            avg_loss = (avg_loss * datasize + loss) / \
                       (datasize + inputs.shape[0])
            epoch_losses.append(float(avg_loss))

            # update data size
            datasize += inputs.shape[0]

        return avg_loss, epoch_losses

    def evaluate(self, dataloader):
        # set to eval mode
        self.model.eval()

        # do not record computations for computing the gradient
        with torch.no_grad():
            epoch_losses = []
            epoch_accuracies = []

            datasize = 0
            avg_accuracy = 0
            avg_loss = 0

            for batch in tqdm(dataloader):
                # load inputs and labels to device
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                # predict with model
                outputs = self.model(inputs.float())

                # compute some losses over time
                loss = self.criterion(outputs, labels)

                print("Outputs")
                print(outputs.shape)

                print("Labels")
                print(labels.shape)

                avg_loss = (avg_loss * datasize + loss) / \
                    (datasize + inputs.shape[0])
                epoch_losses.append(float(avg_loss))

                # compute some accuracies over time
                accuracy = torch.sum(torch.argmax(outputs, dim=1) == labels.argmax(dim=1))
                avg_accuracy = (avg_accuracy * datasize +
                                accuracy) / (datasize + inputs.shape[0])
                epoch_accuracies.append(float(avg_accuracy))

                # update data size
                datasize += inputs.shape[0]

        return avg_loss, avg_accuracy, epoch_losses, epoch_accuracies

    def fit(self, train_dataloader, valid_dataloader):
        best_measure = -1
        best_epoch = -1

        if self.epochs is None or self.criterion is None or self.optimizer is None:
            raise ValueError(
                "Missing parameters \"epochs/criterion/optimizer\"")

        for epoch in range(self.epochs):
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 10)

            # train and evaluate model performance
            train_loss, _ = self.train(train_dataloader)
            valid_loss, measure, _, _ = self.evaluate(valid_dataloader)
            print(f"\nTrain Loss: {train_loss}")
            print(f"Valid Loss: {valid_loss}")
            print(f'Measure: {measure.item()}')

            # save metrics
            self.train_losses.append(float(train_loss))
            self.valid_losses.append(float(valid_loss))
            self.valid_accuracies.append(float(measure))

            # update best performing epoch and save model weights
            if measure > best_measure:
                print(f'Updating best measure: {best_measure} -> {measure}')
                best_epoch = epoch
                best_weights = self.model.state_dict()
                best_measure = measure

        return best_epoch, best_measure, best_weights

    def predict_batches(self, dataloader):
        # set to eval mode
        self.model.eval()

        # do not record computations for computing the gradient
        with torch.no_grad():
            logits = None
            labels = None

            for batch in tqdm(dataloader):
                # load inputs to device
                inputs = batch[0].to(self.device)

                outputs = self.model(inputs)

                # save predictions and labels
                if logits is None:
                    logits = outputs
                    labels = batch[1]
                else:
                    logits = torch.cat((logits, outputs), dim=0)
                    labels = torch.cat((labels, batch[1]), dim=0)

        return logits, labels


# Custom RGB Efficient Net model with customer forward
class RGBEfficientNetModel(nn.Module):
    def __init__(self, model, device, n_classes, weights=None, criterion=None, lr=None, epochs=None):
        super(RGBEfficientNetModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.device = device
        self.config_model(n_classes, weights)
        if lr is not None:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9)
        else:
            self.optimizer = None

        self.train_losses = []
        self.valid_losses = []
        self.valid_accuracies = []

        # load model with custom number of classes

    def forward(self, x):
        # Use the second last layer as the feature map
        features = self.efficient_net_model.extract_features(x)
        return features

    def get_model(self):
        return self.model

    def config_model(self, out_classes, weights):
        """
		Change the last layer of the neural network to fit the number of predictable classes.
		Load trained weights if available. (for evaluating)

		:param out_classes: number of possible classes to be predicted
		:param weights: trained weights of the model
		:return: nil
		"""
        in_feats = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_feats, out_classes)

        if weights is not None:
            self.model.load_state_dict(weights)

        self.model.to(self.device)



# Custom model that thats RGB Model and HyperSpectral Model and concatenates them
class RGB_HyperSpectral_Model(nn.Module):
    def __init__(self, rgb_model, hyper_model, device, n_classes, rgb_weights=None, hyper_weights=None, criterion=None, lr=0.001, epochs=10):
        super(RGB_HyperSpectral_Model, self).__init__()
        self.rgb_model = rgb_model
        self.hyper_model = hyper_model

        self.rgb_weights = rgb_weights
        self.hyper_weights = hyper_weights

        self.device = device
        self.n_classes = n_classes
        self.rgb_criterion = criterion
        self.hyper_criterion = criterion
        self.lr = lr
        self.epochs = epochs
        if lr is not None:
            self.rgb_model_optimizer = torch.optim.SGD(
                self.rgb_model.parameters(), lr=lr, momentum=0.9)
            self.hyper_model_optimizer = torch.optim.SGD(
                self.hyper_model.parameters(), lr=lr, momentum=0.9)
        else:
            self.rgb_model_optimizer = None
            self.hyper_model_optimizer = None

        self.config_model(rgb_weights, hyper_weights)

        self.train_losses = []
        self.valid_losses = []
        self.valid_accuracies = []

    def get_rgb_model(self):
        return self.rgb_model

    def get_hyper_model(self):
        return self.hyper_model

    def config_model(self, rgb_weights, hyper_weights):
        if rgb_weights is not None:
            self.rgb_model.load_state_dict(rgb_weights)

        if hyper_weights is not None:
            self.hyper_model.load_state_dict(hyper_weights)

    def forward(self, rgb_x, hyper_x):
        # Maybe should have used this
        # Forward pass through both models
        rgb_out = self.rgb_model(rgb_x)
        hyper_out = self.hyper_model(hyper_x)

        concatenated = torch.cat((rgb_out, hyper_out), dim=1)

        linear_layer = nn.Linear(concatenated.size(1), self.n_classes)
        output = linear_layer(concatenated.view(concatenated.size(0), -1))

        return output


    def train(self, rgb_dataloader, hyper_dataloader):
        # Set to train mode
        self.rgb_model.train()
        self.hyper_model.train()

        epoch_losses = []

        rgb_datasize, hyper_datasize = 0, 0
        avg_loss = 0

        for rgb_batch, hyper_batch in tqdm(zip(rgb_dataloader, hyper_dataloader)):
            # load inputs and labels to device
            rgb_inputs = rgb_batch[0].to(self.device)
            rgb_labels = rgb_batch[1].to(self.device)

            hyper_inputs = hyper_batch[0].to(self.device)
            hyper_labels = hyper_batch[1].to(self.device)

            # predict with models
            rgb_outputs = self.rgb_model(rgb_inputs.to(self.device))
            hyper_outputs = self.hyper_model(hyper_inputs.float().to(self.device))

            rgb_loss = self.rgb_criterion(rgb_outputs, rgb_labels)
            hyper_loss = self.hyper_criterion(hyper_outputs, hyper_labels)

            # calculate gradient and back propagation
            self.rgb_model_optimizer.zero_grad()  # reset accumulated gradients
            rgb_loss.backward()  # compute new gradients
            self.rgb_model_optimizer.step()  # apply new gradients to change model parameters

            self.hyper_model_optimizer.zero_grad()  # reset accumulated gradients
            hyper_loss.backward()  # compute new gradients
            self.hyper_model_optimizer.step()  # apply new gradients to change model parameters

            # Calculate running average loss for both sets of inputs
            # COULD VERY MUCH BE WRONG
            rgb_avg_loss = (avg_loss * rgb_datasize + rgb_loss) / (rgb_datasize + rgb_inputs.shape[0])
            hyper_avg_loss = (avg_loss * hyper_datasize + hyper_loss) / (hyper_datasize + hyper_inputs.shape[0])

            avg_loss = (rgb_avg_loss + hyper_avg_loss) / 2

            epoch_losses.append(float(avg_loss))

            # Update data size
            rgb_datasize += rgb_inputs.shape[0]
            hyper_datasize += hyper_inputs.shape[0]

        return avg_loss, epoch_losses

    def evaluate(self, rgb_dataloader, hyper_dataloader):
        # Set to eval mode
        self.rgb_model.eval()
        self.hyper_model.eval()

        # do not record computations for computing the gradient
        with torch.no_grad():
            epoch_losses = []
            epoch_accuracies = []

            rgb_datasize, hyper_datasize = 0, 0
            avg_accuracy = 0
            avg_loss = 0

            for rgb_batch, hyper_batch in tqdm(zip(rgb_dataloader, hyper_dataloader)):
                # load inputs and labels to device
                rgb_inputs = rgb_batch[0].to(self.device)
                rgb_labels = rgb_batch[1].to(self.device)

                hyper_inputs = hyper_batch[0].to(self.device)
                hyper_labels = hyper_batch[1].to(self.device)

                # predict with models
                rgb_outputs = self.rgb_model(rgb_inputs.to(self.device))
                hyper_outputs = self.hyper_model(hyper_inputs.float().to(self.device))

                rgb_loss = self.rgb_criterion(rgb_outputs, rgb_labels)
                hyper_loss = self.hyper_criterion(hyper_outputs, hyper_labels)

                rgb_avg_loss = (avg_loss * rgb_datasize + rgb_loss) / (rgb_datasize + rgb_inputs.shape[0])
                hyper_avg_loss = (avg_loss * hyper_datasize + hyper_loss) / (hyper_datasize + hyper_inputs.shape[0])

                avg_loss = (rgb_avg_loss + hyper_avg_loss) / 2

                epoch_losses.append(float(avg_loss))

                # compute some accuracies over time
                rgb_accuracy = torch.sum(torch.argmax(rgb_outputs, dim=1) == rgb_labels.argmax(dim=1))
                hyper_accuracy = torch.sum(torch.argmax(hyper_outputs, dim=1) == hyper_labels.argmax(dim=1))

                rgb_avg_accuracy = (avg_accuracy * rgb_datasize +
                                rgb_accuracy) / (rgb_datasize + rgb_inputs.shape[0] + hyper_inputs.shape[0])

                hyper_avg_accuracy = (avg_accuracy * hyper_datasize +
                                hyper_accuracy) / (hyper_datasize + rgb_inputs.shape[0] + hyper_inputs.shape[0])

                avg_accuracy = (rgb_avg_accuracy + hyper_avg_accuracy) / 2

                epoch_accuracies.append(float(avg_accuracy))

                # update data size
                rgb_datasize += rgb_inputs.shape[0]
                hyper_datasize += hyper_inputs.shape[0]

        return avg_loss, avg_accuracy, epoch_losses, epoch_accuracies

    def fit(self, rgb_train_dataloader, hyper_train_dataloader, rgb_val_dataloader, hyper_val_dataloader):
        best_measure = -1
        best_epoch = -1

        if self.epochs is None or self.rgb_criterion is None or self.hyper_criterion is None or \
                self.rgb_model_optimizer is None or self.hyper_model_optimizer is None:
            raise ValueError(
                "Missing parameters \"epochs/criterion/optimizer\"")

        for epoch in range(self.epochs):
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 10)

            # train and evaluate model performance
            train_loss, _ = self.train(rgb_train_dataloader, hyper_train_dataloader)
            valid_loss, measure, _, _ = self.evaluate(rgb_val_dataloader, hyper_val_dataloader)
            print(f"\nTrain Loss: {train_loss}")
            print(f"Valid Loss: {valid_loss}")
            print(f'Measure: {measure.item()}')

            # save metrics
            self.train_losses.append(float(train_loss))
            self.valid_losses.append(float(valid_loss))
            self.valid_accuracies.append(float(measure))

            # update best performing epoch and save model weights
            if measure > best_measure:
                print(f'Updating best measure: {best_measure} -> {measure}')
                best_epoch = epoch
                rgb_best_weights = self.rgb_model.state_dict()
                hyper_best_weights = self.hyper_model.state_dict()
                best_measure = measure

        return best_epoch, best_measure, rgb_best_weights, hyper_best_weights


    def predict_batches(self, rgb_dataloader, hyper_dataloader):
        # Set to eval mode
        self.rgb_model.eval()
        self.hyper_model.eval()

        # do not record computations for computing the gradient
        with torch.no_grad():
            logits = None
            labels = None

            for rgb_batch, hyper_batch in tqdm(zip(rgb_dataloader, hyper_dataloader)):

                # load inputs and labels to device
                rgb_inputs = rgb_batch[0].to(self.device)
                rgb_labels = rgb_batch[1].to(self.device)

                hyper_inputs = hyper_batch[0].to(self.device)
                hyper_labels = hyper_batch[1].to(self.device)

                # predict with models
                rgb_outputs = self.rgb_model(rgb_inputs.to(self.device))
                hyper_outputs = self.hyper_model(hyper_inputs.float().to(self.device))



                # Pass through relu and pooled and linear layer for hyper
                hyper_relu = nn.ReLU()
                hyper_pooled = nn.AvgPool2d(2)

                hyper_outputs = hyper_relu(hyper_outputs)

                # hyper_outputs = hyper_pooled(hyper_outputs)

                # Maybe flatten
                features_rgb = torch.flatten(rgb_outputs, start_dim=1)
                features_hyper = torch.flatten(hyper_outputs, start_dim=1)

                # concatenate outputs
                concatenated = torch.cat((features_rgb, features_hyper), dim=1)

                # Pass through linear layer
                linear_layer = nn.Linear(concatenated.size(1), self.n_classes)
                outputs = linear_layer(concatenated.view(concatenated.size(0), -1)).to(self.device)

                # concatenate labels TODO: Check if this is correct
                labels_batch = torch.cat((rgb_labels, hyper_labels), dim=1)

                # concatenate logits
                if logits is None:
                    logits = outputs
                    labels = rgb_batch[1]
                else:
                    logits = torch.cat((logits, outputs), dim=0)
                    labels = torch.cat((labels, rgb_batch[1]), dim=0)

        return logits, labels





