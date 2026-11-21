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


# TODO: create a evaluation function for iris + CNN model
def iris_and_model(
    iris_pair_df: pd.DataFrame, vector_dict: dict, model_threshold=0.5, min_threshold=0.33, max_threshold=0.37
):
    indecisive_df = iris_pair_df[
        (min_threshold <= iris_pair_df["hamming_distance"]) & (iris_pair_df["hamming_distance"] <= max_threshold)
    ]
    decisive_df = iris_pair_df[~iris_pair_df.index.isin(indecisive_df.index)]
    y_pred = list(decisive_df["hamming_distance"] < min_threshold)
    y_true = list(decisive_df["label"])

    tqdm.pandas()
    for _, row in tqdm(indecisive_df.iterrows(), total=indecisive_df.shape[0]):
        prob = is_vector_similar(vector_dict[row["a_id"]], vector_dict[row["b_id"]])
        y_pred.append(prob.item() >= model_threshold)
        y_true.append(row["label"])

    return y_pred, y_true


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Model Evaluation Script",
        description="Evaluate the model with test data from previous stage",
    )
    parser.add_argument("-v", "--vector-path", help="Path to save model vector")
    parser.add_argument("-mt", "--model-threshold", default=0.5, help="Model prediction threshold (0 to 1)")
    parser.add_argument(
        "-ht", "--hamming-threshold", default=0.35, help="Hamming distance prediction threshold (around 0.3 to 0.4)"
    )
    parser.add_argument("-lt", "--lower-threshold", default=0.33, help="Min hamming distance prediction threshold")
    parser.add_argument("-gt", "--greater-threshold", default=0.37, help="Max hamming distance prediction threshold")

    args = parser.parse_args()
    vector_path = str(args.vector_path)
    model_threshold = float(args.model_threshold)
    hamming_threshold = float(args.hamming_threshold)
    min_threshold = float(args.lower_threshold)
    max_threshold = float(args.greater_threshold)

    from sklearn import metrics

    iris_code_df = get_iris_code_df()
    iris_code_df = iris_code_df[~iris_code_df["Output_Metadata"].isna() & iris_code_df["Output_Error"].isna()]
    iris_code_df["id"] = iris_code_df["ImagePath"].apply(lambda path: path.split("/")[-1].split(".")[0])
    iris_code_df = iris_code_df.set_index(["id"])

    print("----- Model Alone -----")
    print(f"Set model threshold {model_threshold}")

    vector_df = pd.read_feather(vector_path)
    vector_df = vector_df[vector_df.index.isin(iris_code_df.index)]

    loss_fn = torch.nn.BCELoss()
    y_prob, y_true = model_alone(
        torch.tensor(vector_df["vectors"].to_list()),
        torch.tensor(vector_df["label"].to_list()),
        loss_fn,
        threshold=model_threshold,
    )
    y_pred = np.greater_equal(y_prob, model_threshold)

    print("Confusion matrix")
    print(metrics.confusion_matrix(y_true, y_pred, labels=[True, False]))
    print("Classification report")
    print(metrics.classification_report(y_true, y_pred, labels=[True, False]))
    print(f"ROC/AUC score: {metrics.roc_auc_score(y_true, y_prob):.4f}")

    print("----- Iris Alone -----")
    print(f"Set hamming distance threshold {hamming_threshold}")

    iris_pair_df = get_iris_pair_df(iris_code_df.index.to_list())
    iris_pair_df = iris_pair_df[~iris_pair_df["hamming_distance"].isna()]

    y_dist = iris_pair_df["hamming_distance"]
    y_true = iris_pair_df["label"]
    y_pred = y_dist < hamming_threshold

    print("Confusion matrix")
    print(metrics.confusion_matrix(y_true, y_pred, labels=[True, False]))
    print("Classification report")
    print(metrics.classification_report(y_true, y_pred, labels=[True, False]))

    print("----- Iris + Model -----")
    print(f"Set min|max distance threshold {min_threshold}|{max_threshold}")

    y_pred, y_true = iris_and_model(
        iris_pair_df,
        vector_df["vectors"].apply(lambda vector: torch.tensor(list(vector))).to_dict(),
        model_threshold=model_threshold,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )

    print("Confusion matrix")
    print(metrics.confusion_matrix(y_true, y_pred, labels=[True, False]))
    print("Classification report")
    print(metrics.classification_report(y_true, y_pred, labels=[True, False]))
