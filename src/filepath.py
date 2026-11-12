import os

KAGGLE_DIR = "./kaggle"
KAGGLE_IRIS_CSV_PATH = os.path.join(KAGGLE_DIR, "iris_thousands.csv")

DATA_DIR = "./data"
IRIS_DF_PATH = os.path.join(DATA_DIR, "iris.feather")
IRIS_CODE_DF_PATH = os.path.join(DATA_DIR, "iris-code.feather")

TEMP_DIR = "./.temp"

if not os.path.exists(KAGGLE_DIR):
    import kagglehub

    kagglehub.dataset_download("sondosaabed/casia-iris-thousand", output_dir=KAGGLE_DIR)

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)
