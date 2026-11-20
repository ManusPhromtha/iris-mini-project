import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_prep import get_iris_code_df
from model_prep import get_data

PAIR_PATH = ".temp/iris_pair.feather"


def create_iris_pair_df(ids: list[str]) -> pd.DataFrame:
    n = len(ids)
    total_combination = int(n * (n - 1) / 2)

    a_ids = np.full(shape=(total_combination), fill_value="S0000L00")
    b_ids = np.full(shape=(total_combination), fill_value="S0000L00")
    labels = np.full(shape=(total_combination), fill_value=False)
    idx = 0
    for row in range(n):
        for col in range(row + 1, n):
            a_ids[idx] = ids[row]
            b_ids[idx] = ids[col]
            labels[idx] = ids[row][1:6] == ids[col][1:6]
            idx += 1

    pair_df = pd.DataFrame.from_dict({"a_id": a_ids, "b_id": b_ids, "label": labels})
    pair_df["hamming_distance"] = None
    pair_df.to_feather(PAIR_PATH)
    return pair_df


def get_iris_pair_df(ids: list[str]) -> pd.DataFrame:
    return pd.read_feather(PAIR_PATH) if os.path.exists(PAIR_PATH) else create_iris_pair_df(ids)


def infer_iris_pair(iris_dict: dict, ids: list[str], count: int = -1):
    import iris

    iris_pair_df = get_iris_pair_df(ids)
    print(f"Remaining rows: {iris_pair_df['hamming_distance'].isna().sum()}")

    missing_df = iris_pair_df[iris_pair_df["hamming_distance"].isna()]
    if count > 0:
        missing_df = missing_df.head(count)

    matcher = iris.SimpleHammingDistanceMatcher()

    tqdm.pandas(desc="Calculate iris code hamming distance")
    missing_df["hamming_distance"] = missing_df.progress_apply(
        lambda row: matcher.run(
            iris_dict[row["a_id"]],
            iris_dict[row["b_id"]],
        ),
        axis=1,
    )

    iris_pair_df = pd.concat([missing_df, iris_pair_df[~iris_pair_df.index.isin(missing_df.index)]])
    iris_pair_df.to_feather(PAIR_PATH)

    print(f"Remaining rows: {iris_pair_df['hamming_distance'].isna().sum()}")

    return iris_pair_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="Iris Code Hamming Distance Script")
    parser.add_argument("-n", "--count", default=1000, help="Number of rows to calculate distance")
    args = parser.parse_args()
    count = int(args.count)

    iris_code_df = get_iris_code_df()
    iris_code_df = iris_code_df[~iris_code_df["Output_Metadata"].isna() & iris_code_df["Output_Error"].isna()]
    iris_code_df["id"] = iris_code_df["ImagePath"].apply(lambda path: path.split("/")[-1].split(".")[0])
    iris_code_df = iris_code_df.set_index(["id"])

    _, _, X_test = get_data()
    ids = []
    for _, group in X_test:
        for path in group["ImagePath"]:
            ids.append(path.split("/")[-1].split(".")[0])
    iris_code_df = iris_code_df[iris_code_df.index.isin(ids)]
    print(infer_iris_pair(iris_code_df["Output_Code"].to_dict(), iris_code_df.index.to_list(), count=count).head(2))
