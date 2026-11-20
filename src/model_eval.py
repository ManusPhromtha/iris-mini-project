import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_prep import get_iris_code_df
from model_eval_iris import get_iris_pair_df
from model_train import is_vector_similar


def model_alone(vectors, labels, loss_fn, threshold=0.5):
    pair_comb = torch.combinations(torch.arange(len(vectors)), 2)
    pair_prob = torch.empty(pair_comb.size()[0], dtype=float)
    pair_true = torch.empty(pair_comb.size()[0], dtype=float)
    for idx in tqdm(torch.arange(pair_comb.size()[0])):
        a, b = pair_comb[idx]
        pair_prob[idx] = torch.clamp(is_vector_similar(vectors[a], vectors[b]), min=0, max=1)
        pair_true[idx] = labels[a].eq(labels[b])

    loss = loss_fn(pair_prob, pair_true)
    accuracy = ((pair_prob.ge(threshold) == pair_true).sum() / len(pair_comb)) * 100

    print(f"Average Loss: {loss}, accuracy: {accuracy}")

    return pair_prob.detach().numpy(), pair_true.detach().numpy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Model Evaluation Script",
        description="Evaluate the model with test data from previous stage",
    )
    parser.add_argument("-v", "--vector-path", help="Path to save model vector")
    parser.add_argument("-t", "--threshold", default=0.5, help="Prediction threshold cutoff")
    args = parser.parse_args()
    vector_path = str(args.vector_path)
    threshold = float(args.threshold)

    print(f"Set threshold {threshold}")

    from sklearn import metrics

    iris_code_df = get_iris_code_df()
    iris_code_df = iris_code_df[~iris_code_df["Output_Metadata"].isna() & iris_code_df["Output_Error"].isna()]
    iris_code_df["id"] = iris_code_df["ImagePath"].apply(lambda path: path.split("/")[-1].split(".")[0])
    iris_code_df = iris_code_df.set_index(["id"])

    print("----- Model Alone -----")
    vector_df = pd.read_feather(vector_path)
    vector_df = vector_df[vector_df.index.isin(iris_code_df.index)]

    loss_fn = torch.nn.BCELoss()
    y_prob, y_true = model_alone(
        torch.tensor(vector_df["vectors"].to_list()), torch.tensor(vector_df["label"].to_list()), loss_fn
    )
    y_pred = np.greater_equal(y_prob, threshold)

    print("Confusion matrix")
    print(metrics.confusion_matrix(y_true, y_pred, labels=[True, False]))
    print("Classification report")
    print(metrics.classification_report(y_true, y_pred, labels=[True, False]))
    print(f"ROC/AUC score: {metrics.roc_auc_score(y_true, y_prob):.4f}")

    # print("----- Iris Alone -----")
    iris_pair_df = get_iris_pair_df(iris_code_df.index.to_list())
    iris_pair_df = iris_pair_df[~iris_pair_df["hamming_distance"].isna()]

    y_dist = iris_pair_df["hamming_distance"]
    y_true = iris_pair_df["label"]
    y_pred = y_dist < 0.35

    print("Confusion matrix")
    print(metrics.confusion_matrix(y_true, y_pred, labels=[True, False]))
    print("Classification report")
    print(metrics.classification_report(y_true, y_pred, labels=[True, False]))
