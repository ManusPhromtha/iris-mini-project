# iris-mini-project

## Idea

1. Infer iris code from the image
2. If hamming distance is in a indecisive threshold pass it to CNN model
3. CNN model takes the image and encode it to vector (can be done in parallel with step 1)
4. Check if the vector is similar to the previous registered vector
5. Decide to accept or reject the verification

This will reduce the need for SVM and also increase scalability of the system. (See [referenced research](https://ieeexplore.ieee.org/abstract/document/10354096))

## Setup

I use uv but you can simply install the dependencies listed in `pyproject.toml`.  
You also need to run `pip install --no-deps open-iris` too.

## How to run

Run every script from the root directory so the filepath will not break.

The code has not been cleaned up yet, many functions may need to be modified to train another model.  
The only documentation here may also be out of sync with the codebase.

### Data Preparation

The kaggle dataset will be automatically downloaded (code in `filepath.py`)

Running `data_prep.py` will generate `iris.feather` which simply fix some path in the dataset.  
And `iris-code.feather` which contains the iris code of each image.  
This might take a long time, you can download the pre-computed data from [Google Drive](https://drive.google.com/drive/folders/15HNYLA9SvSonDe-qK25tZrBjgKucVWPz?usp=sharing) to skip waiting time.

```sh
# -n --infer-count | number of rows to infer iris code
uv run src/data_prep.py -n 100
```

### Model Training

Running `model_prep.py` will split the data for training and evaluating.

```sh
# -n --label-count | number of label to include in the next stage
# dataset has 2000 labels, recommend small dataset first
uv run src/model_prep.py -n 100
```

After that run `model_train.py` to train on training data.  
The model information will be in `model_cnn.py`

```sh
uv run src/model_cnn.py
```

```sh
# -c --checkpoint-path | Path to load/save training state (omit to load/save nothing)
# -m --model-path | Path to load/save best model (omit to load/save nothing)
# -n --epoch-count | Number of epochs to train the model
uv run src/model_train.py -c data/model/custom.checkpoint -m data/model/custom.model -n 10
```

Modify `model_cnn.py` and experiment with different model.

### Model Evaluation

Run `model_test.py` to encode the image into vector and store into the specified path

```sh
# -m --model-path | Path to load model
# -v --vector-path | Path to save the image vector of training dataset
uv run src/model_test.py -m data/model/custom.model -v data/model/custom.vector
```

Run `model_eval_iris.py` to calculate the hamming distance of all pair combination in test dataset.

```sh
# -n --count | Number of rows to calculate hamming distance
# this might take a while depending on the test dataset
uv run src/model_eval_iris.py -n 1000
```

Run `model_eval.py` to evaluate the model on all pair combination in test dataset.  
And also evaluate iris code distance based on a specified threshold on all pair combination

```sh
# -v --vector-path | Path to load the image vector of training dataset
# -mt --model-threshold | Model prediction threshold (0 to 1)
# -ht --hamming-threshold | Hamming distance prediction threshold (recommend around 0.3 to 0.4)
uv run src/model_eval.py -v data/model/custom.vector -mt 0.5 -ht 0.35
```

## Resources from professor

AI Frontier Subject
iris verification: ศึกษาและพัฒนาระบบยืนยันตัวตนด้วยลายม่านตา โดยใช้ baseline ที่เตรียมไว้ให้ (ดูรายละเอียดจากไฟล์ readme) ร่วมกับ CASIA-Iris-Thousand dataset หรือจะใช้ dataset อื่นก็ได้ (เพื่อความง่าย แนะนำให้ใช้ data ที่ capture จากกล้องย่าน NIR) และตั้งต้นจาก open iris ของ world coin ก็ได้ครับ

Resources:
<https://drive.google.com/drive/folders/1o5n7pTQaPUqQauUZvAsGvZ0l3dTGadgv?usp=sharing>

<https://www.kaggle.com/datasets/sondosaabed/casia-iris-thousand>

<https://github.com/worldcoin/open-iris>

<https://world.org/blog/engineering/iris-recognition-inference-system>

ตัวอย่างไอเดียสำหรับ improvement ดูได้จาก

<https://ieeexplore.ieee.org/abstract/document/10354096>

<https://link.springer.com/chapter/10.1007/978-3-031-46781-3_13>

<https://ieeexplore.ieee.org/abstract/document/10665383>

<https://link.springer.com/chapter/10.1007/978-981-96-4606-7_13>

<https://link.springer.com/chapter/10.1007/978-981-96-4606-7_14>
