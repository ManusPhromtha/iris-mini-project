import os

import iris
import numpy as np
import pandas as pd

import filepath


class FileNotFoundException(Exception):
    def __init__(self, message):
        super().__init__(message)


# ---- Iris DataFrame ----
def create_iris_df() -> pd.DataFrame:
    df = pd.read_csv(filepath.KAGGLE_IRIS_CSV_PATH, index_col=0)

    def fix_path(path: str):
        arr = path.split("/")
        arr = arr[4:]
        arr.insert(0, "CASIA-Iris-Thousand")
        return os.path.join(filepath.KAGGLE_DIR, "/".join(arr))

    df["ImagePath"] = df["ImagePath"].apply(fix_path)
    df.to_feather(filepath.IRIS_DF_PATH)

    return df


def read_iris_df() -> pd.DataFrame:
    return pd.read_feather(filepath.IRIS_DF_PATH)


def get_iris_df() -> pd.DataFrame:
    return read_iris_df() if os.path.exists(filepath.IRIS_DF_PATH) else create_iris_df()


# ---- Iris Code DataFrame ----
def create_iris_code_df() -> pd.DataFrame:
    df = get_iris_df()

    df["Eye_side"] = np.where(df["Label"] == "R", "right", "left")
    df["Output_Error"] = None
    df["Output_Code"] = None
    df["Output_Metadata"] = None

    save_iris_code_df(df)
    return df


def save_iris_code_df(df: pd.DataFrame):
    df = df.copy()
    df["Output_Code"] = df["Output_Code"].apply(lambda code: code.serialize() if code is not None else None)
    df.to_feather(filepath.IRIS_CODE_DF_PATH)


def read_iris_code_df() -> pd.DataFrame:
    df = pd.read_feather(filepath.IRIS_CODE_DF_PATH)
    df["Output_Code"] = df["Output_Code"].apply(
        lambda code: iris.IrisTemplate.deserialize(code) if code is not None else None
    )
    return df


def get_iris_code_df() -> pd.DataFrame:
    return read_iris_code_df() if os.path.exists(filepath.IRIS_CODE_DF_PATH) else create_iris_code_df()


def infer_iris_code(iris_df: pd.DataFrame, count: int) -> pd.DataFrame:
    missing_df = iris_df[iris_df["Output_Metadata"].isna()].head(count)

    if missing_df.shape[0] > 0:
        import cv2
        from tqdm import tqdm

        tqdm.pandas(desc="Inferring iris code")

        iris_pipeline = iris.IRISPipeline()

        def run_pipeline(image_path: str, eye_side: str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            output = iris_pipeline(iris.IRImage(img_data=image, eye_side=eye_side))
            if output["error"] is not None:
                return output["error"], None, output["metadata"]

            return None, output["iris_template"], output["metadata"]

        missing_df[["Output_Error", "Output_Code", "Output_Metadata"]] = missing_df.progress_apply(
            lambda row: run_pipeline(row["ImagePath"], row["Eye_side"]),
            axis=1,
            result_type="expand",
        )

        iris_df = pd.concat([missing_df, iris_df[np.invert(iris_df.index.isin(missing_df.index))]])
        save_iris_code_df(iris_df)
    return iris_df


iris_df = get_iris_df()
iris_code_df = get_iris_code_df()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Data Preparation Script",
        description="Prepare the data for later training and evaluating",
    )
    parser.add_argument("-i", "--infer-count", default=0, help="Number of rows to infer iris code")
    args = parser.parse_args()
    infer_count = int(args.infer_count)

    print(iris_code_df.shape)
    print(iris_code_df.sample(1))

    def print_infer_stats(iris_code_df: pd.DataFrame):
        inferred_count = (~iris_code_df["Output_Metadata"].isna()).sum()
        remaining_count = iris_code_df["Output_Metadata"].isna().sum()
        print(f"Inferred|Remaining Images: {inferred_count}|{remaining_count}")

    print_infer_stats(iris_code_df)

    if infer_count > 0:
        iris_code_df = infer_iris_code(iris_code_df, infer_count)
        print_infer_stats(iris_code_df)
        print(iris_code_df[~iris_code_df["Output_Metadata"].isna()].sample(1))
