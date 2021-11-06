from typing import List, Optional, Union
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import numpy as np


MODELS: List[Union[LogisticRegression, DecisionTreeClassifier]] = [
    LogisticRegression(),
    DecisionTreeClassifier(max_depth=1),
    DecisionTreeClassifier(max_depth=3),
    DecisionTreeClassifier(max_depth=5),
    DecisionTreeClassifier(max_depth=7),
]

DATAFILE = "data/seeds_dataset.csv"


def show_metrics(y_train: np.ndarray,
                 y_test: np.ndarray,
                 y_preds_train: List[np.ndarray],
                 y_preds_test: List[np.ndarray],
                 titles: Optional[List[str]] = None):
    n_models = len(y_preds_train)

    size = (6.4 * n_models, 4.8 * 2)
    fig = plt.figure(figsize=size)

    if titles is None:
        titles = ("" for _ in range(n_models))

    for i, (y_pred_train, y_pred_test, title) in \
            enumerate(zip(y_preds_train, y_preds_test, titles)):

        ax_train = fig.add_subplot(3, n_models, i + 1)
        ax_test = fig.add_subplot(3, n_models, i + 1 + n_models)
        ax_train.set_title(title)

        for y_pred, y, ax, ylabel, xtickslables in (zip(
            [y_pred_train, y_pred_test],
            [y_train, y_test],
            [ax_train, ax_test],
            ["Train set", "Test set"],
            [False, True],
        )):
            matrix = confusion_matrix(y, y_pred)
            yticklabels = i == 0
            heatmap(matrix, annot=True, ax=ax,
                    cbar=False, yticklabels=yticklabels,
                    xticklabels=xtickslables)

            ax.set_ylabel(ylabel if i == 0 else None)

        avg = "binary" if len(matrix) == 2 else "macro"

        kwargs = {"zero_division": 0, "average": avg}

        y_zipped = zip([y_train, y_test], [y_pred_train, y_pred_test])
        accuracy = [accuracy_score(y, y_pred) for y, y_pred in y_zipped]
        y_zipped = zip([y_train, y_test], [y_pred_train, y_pred_test])
        precision = [precision_score(y, y_pred, **kwargs)
                     for y, y_pred in y_zipped]
        y_zipped = zip([y_train, y_test], [y_pred_train, y_pred_test])
        recall = [recall_score(y, y_pred, **kwargs) for y, y_pred in y_zipped]
        y_zipped = zip([y_train, y_test], [y_pred_train, y_pred_test])
        f1 = [f1_score(y, y_pred, **kwargs) for y, y_pred in y_zipped]

        metrics = [accuracy, precision, recall, f1]
        metric_labels = ["acc.", "prec.", "recall", "f1"]

        ax_bar = fig.add_subplot(3, n_models, i + 1 + 2*n_models)
        width = 1/3
        x = np.arange(len(metrics))
        train_metrics = list(zip(*metrics))[0]
        test_metrics = list(zip(*metrics))[1]
        ax_bar.bar(x, train_metrics, width=width, label="Train set")
        ax_bar.bar(x + width, test_metrics, width=width, label="Test set")
        ax_bar.set_xticks(x + width/2)
        ax_bar.set_xticklabels(metric_labels)
        ax_bar.legend()
        ax_bar.set_ylim(0, 1)

    fig.tight_layout(w_pad=0)
    plt.subplots_adjust(wspace=0.15, left=0.03)
    plt.show()


def normalize(matrix: np.ndarray):
    result = np.array(matrix)
    for i, col in enumerate(matrix.T):
        result[:, i] /= max(col)
    return result


if __name__ == "__main__":

    data = pd.read_csv(DATAFILE)
    X = normalize(data.drop('class', axis=1).values)
    y = data['class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

    y_preds_test = []
    y_preds_train = []
    for model in MODELS:
        model.fit(X_train, y_train)
        y_preds_train.append(model.predict(X_train))
        y_preds_test.append(model.predict(X_test))

    show_metrics(y_train, y_test, y_preds_train, y_preds_test,
                 ["Logistic Regression",
                  "Tree (max_depth=1)",
                  "Tree (max_depth=3)",
                  "Tree (max_depth=5)",
                  "Tree (max_depth=7)", ])
