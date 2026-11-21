import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from model_prep import LabelImageGroupDataset, create_dataset, create_dataset_loader, get_data
from model_train import batch_predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_test(loss_fn):
    with tqdm(enumerate(TestDatasetLoader), total=len(TestDatasetLoader)) as pbar:
        losses = []
        accuracies = []
        for batchIdx, batchData in pbar:
            loss, accuracy = batch_predict(batchData, model, loss_fn)

            accuracies.append(accuracy)
            losses.append(loss.item())

            pbar.set_description(
                f"Accuracy: ({accuracy:.4f}|{np.mean(accuracies):.4f}) , Loss: ({loss:.4f}|{np.mean(losses):.4f})"
            )

    print(f"Average loss: {np.mean(losses)}")


def encode_vector(TestDataset: LabelImageGroupDataset):
    vectors = []
    groups_labels = []
    vectors_labels = []

    with tqdm(enumerate(TestDatasetLoader), total=len(TestDatasetLoader)) as pbar:
        for batchIdx, batchData in pbar:
            group_images, group_labels, group_vector_labels = batchData
            group_images, group_labels = group_images.to(device), group_labels.to(device)

            images = torch.flatten(group_images, end_dim=1)
            labels = torch.flatten(group_labels, end_dim=1)
            vector_labels = torch.flatten(group_vector_labels, end_dim=1)

            # encode to vector
            outputs = model(images)
            for output in outputs:
                vectors.append(output.detach().cpu().numpy())

            for label in labels:
                groups_labels.append(label.item())

            for label in TestDataset.get_label_str([int(label.item()) for label in vector_labels]):
                vectors_labels.append(str(label))

    return vectors, groups_labels, vectors_labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Model Evaluation Script",
        description="Evaluate the model with test data from previous stage",
    )
    parser.add_argument("model_type", default="custom", help="Type of model")
    parser.add_argument("-m", "--model-path", help="Path to pytorch model file")
    parser.add_argument("-v", "--vector-path", default="", help="Path to save model vector")
    args = parser.parse_args()
    model_type = str(args.model_type)
    model_path = str(args.model_path)
    vector_path = str(args.vector_path)

    from model_cnn import get_model

    model_params = torch.load(model_path, weights_only=True, map_location=device)

    model = get_model(model_type, pretrained=False)
    loss_fn = nn.BCELoss()
    model.load_state_dict(model_params)
    model = model.to(device)
    model.eval()
    torch.no_grad()

    TrainDataset, ValidDataset, TestDataset = create_dataset(*get_data())

    _, _, TestDatasetLoader = create_dataset_loader(TrainDataset, ValidDataset, TestDataset, batch_size=4)

    batch_test(loss_fn=loss_fn)

    vectors, group_labels, vectors_labels = encode_vector(TestDataset)
    print(len(vectors), len(group_labels), len(vectors_labels))

    if vector_path != "":
        df = pd.DataFrame.from_dict({"vectors": vectors, "label": group_labels, "id": vectors_labels})
        df = df.set_index("id")
        df.to_feather(vector_path)
        print(df.sample(2))
