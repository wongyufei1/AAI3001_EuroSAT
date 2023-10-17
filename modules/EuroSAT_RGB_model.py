import torch
from tqdm import tqdm


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
                avg_loss = (avg_loss * datasize + loss) / (datasize + inputs.shape[0])
                epoch_losses.append(float(avg_loss))

                # compute some accuracies over time
                _, preds = torch.max(torch.softmax(outputs, 1), 1)
                _, labels = torch.max(labels, 1)

                accuracy = torch.sum(preds == labels)
                avg_accuracy = (avg_accuracy * datasize + accuracy) / (datasize + inputs.shape[0])
                epoch_accuracies.append(float(avg_accuracy))

                # update data size
                datasize += inputs.shape[0]

        return avg_loss, avg_accuracy, epoch_losses, epoch_accuracies

    def fit(self, train_loader, val_loader):
        best_measure = -1
        best_epoch = -1

        if self.epochs is None or self.criterion is None or self.optimizer is None:
            raise ValueError("Missing parameters \"epochs/criterion/optimizer\"")

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
                # accuracy = torch.sum(preds == labels) # May not be correct
                # accuracy = (preds == labels).sum()

                # accuracy = (preds == labels).sum() / labels.size(1)
                accuracy = sum([(label == preds[idx]).all(dim=0) for idx, label in enumerate(labels)])
                # accuracy = (preds == labels).sum() / (labels.size(0) * labels.size(1))

                avg_accuracy = (avg_accuracy * datasize +
                                accuracy) / (datasize + inputs.shape[0])
                epoch_accuracies.append(float(avg_accuracy))

                # update data size
                datasize += inputs.shape[0]

        return avg_loss, avg_accuracy, epoch_losses, epoch_accuracies
