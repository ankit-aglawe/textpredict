"""
Custom implementation of common evaluation metrics: accuracy score, precision, recall, F1 score, and support.
Inspired by scikit-learn's implementation.

References:
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
"""


def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score.

    Args:
        y_true (list or np.array): True labels.
        y_pred (list or np.array): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    total = len(y_true)
    return correct / total


def precision_recall_fscore_support(y_true, y_pred, average="weighted"):
    """
    Calculate precision, recall, and F1 score.

    Args:
        y_true (list or np.array): True labels.
        y_pred (list or np.array): Predicted labels.
        average (str, optional): Type of averaging to perform on the data.
                                 Options are 'micro', 'macro', 'weighted', and None.
                                 Defaults to 'weighted'.

    Returns:
        tuple: (precision, recall, f1, support)
    """
    unique_labels = set(y_true).union(set(y_pred))
    precision = {}
    recall = {}
    f1 = {}
    support = {}

    for label in unique_labels:
        tp = sum((y_t == label and y_p == label) for y_t, y_p in zip(y_true, y_pred))
        fp = sum((y_t != label and y_p == label) for y_t, y_p in zip(y_true, y_pred))
        fn = sum((y_t == label and y_p != label) for y_t, y_p in zip(y_true, y_pred))

        support[label] = tp + fn

        # Calculate precision
        precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Calculate recall
        recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Calculate F1 score
        f1[label] = (
            (2 * precision[label] * recall[label] / (precision[label] + recall[label]))
            if (precision[label] + recall[label]) > 0
            else 0.0
        )

    if average == "micro":
        tp = sum((y_t == y_p) for y_t, y_p in zip(y_true, y_pred))
        fp = sum((y_t != y_p) for y_t, y_p in zip(y_true, y_pred))
        fn = fp
        precision_micro = tp / (tp + fp)
        recall_micro = tp / (tp + fn)
        f1_micro = (
            2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
        )
        return precision_micro, recall_micro, f1_micro, sum(support.values())

    elif average == "macro":
        precision_macro = sum(precision.values()) / len(unique_labels)
        recall_macro = sum(recall.values()) / len(unique_labels)
        f1_macro = sum(f1.values()) / len(unique_labels)
        return precision_macro, recall_macro, f1_macro, sum(support.values())

    elif average == "weighted":
        total_support = sum(support.values())
        precision_weighted = (
            sum(precision[label] * support[label] for label in unique_labels)
            / total_support
        )
        recall_weighted = (
            sum(recall[label] * support[label] for label in unique_labels)
            / total_support
        )
        f1_weighted = (
            sum(f1[label] * support[label] for label in unique_labels) / total_support
        )
        return precision_weighted, recall_weighted, f1_weighted, total_support

    else:
        precision_list = [precision[label] for label in unique_labels]
        recall_list = [recall[label] for label in unique_labels]
        f1_list = [f1[label] for label in unique_labels]
        support_list = [support[label] for label in unique_labels]
        return precision_list, recall_list, f1_list, support_list
