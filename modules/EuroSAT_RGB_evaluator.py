import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score


class EuroSatRgbEvaluator:
    def __init__(self, indices_to_labels):
        self.indices_to_labels = indices_to_labels
        self.n_classes = len(indices_to_labels)

    # calculate average precision score for each class
    def avg_precision_by_class(self, preds, labels):
        classes_avg_precision = None

        # loop through each class
        for i in range(self.n_classes):
            # calculate score
            avg_precision = average_precision_score(labels[:, i], preds[:, i])

            # append to array
            if classes_avg_precision is None:
                classes_avg_precision = np.array(avg_precision)
            else:
                classes_avg_precision = np.append(classes_avg_precision, avg_precision)

        mean_avg_precision = np.mean(classes_avg_precision)

        return mean_avg_precision, classes_avg_precision

    # calculate accuracy score for each class
    def accuracy_by_class(self, preds, labels):
        classes_accuracy = None

        # loop through each class
        for i in range(self.n_classes):
            # calculate score
            accuracy = accuracy_score(labels[:, i], preds[:, i])

            # append to array
            if classes_accuracy is None:
                classes_accuracy = np.array(accuracy)
            else:
                classes_accuracy = np.append(classes_accuracy, accuracy)

        mean_accuracy = np.mean(classes_accuracy)

        return mean_accuracy, classes_accuracy
