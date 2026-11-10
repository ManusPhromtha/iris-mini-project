import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_predict(batchData, model: nn.Module, loss_fn):
    group_images, group_labels = batchData
    group_images, group_labels = group_images.to(device), group_labels.to(device)

    images = torch.flatten(group_images, end_dim=1)
    labels = torch.flatten(group_labels, end_dim=1)

    # encode to vector
    outputs = model(images)

    # TODO: can we optimize this?
    pair_comb = torch.combinations(torch.arange(images.size()[0]), 2)
    pair_pred = torch.empty(pair_comb.size()[0], dtype=float)
    pair_true = torch.empty(pair_comb.size()[0], dtype=float)
    for idx in torch.arange(pair_comb.size()[0]):
        a, b = pair_comb[idx]
        pair_pred[idx] = torch.clamp((F.cosine_similarity(outputs[a], outputs[b], dim=0) + 1) / 2, min=0, max=1)
        pair_true[idx] = labels[a].eq(labels[b])

    loss = loss_fn(pair_pred, pair_true)
    accuracy = ((pair_pred.ge(0.5) == pair_true).sum() / len(pair_comb)) * 100

    return loss, accuracy


def train(model: nn.Module, loss_fn, optimizer, scheduler):
    model.to(device)

    train_losses = []
    valid_losses = []

    for epoch in range(1, 30):
        print(f"Epoch: {epoch}")

        # TRAIN
        model.train()
        with tqdm(enumerate(TrainDatasetLoader), total=len(TrainDatasetLoader)) as pbar:
            losses = []
            accuracies = []
            for batchIdx, batchData in pbar:
                # zero the parameter gradients
                optimizer.zero_grad()

                loss, accuracy = batch_predict(batchData, model, loss_fn)

                # Compute gradient, update parameters
                loss.backward()
                optimizer.step()

                accuracies.append(accuracy)
                losses.append(loss.item())

                pbar.set_description(
                    f"Accuracy: ({accuracy:.4f}|{np.mean(accuracies):.4f}) , Loss: ({loss:.4f}|{np.mean(losses):.4f})"
                )

            train_losses.append(np.mean(losses))

        # VALIDATE
        model.eval()
        with tqdm(enumerate(ValidDatasetLoader), total=len(ValidDatasetLoader)) as pbar:
            losses = []
            accuracies = []
            for batchIdx, batchData in pbar:
                loss, accuracy = batch_predict(batchData, model, loss_fn)

                accuracies.append(accuracy)
                losses.append(loss.item())

                pbar.set_description(
                    f"Accuracy: ({accuracy:.4f}|{np.mean(accuracies):.4f}) , Loss: ({loss:.4f}|{np.mean(losses):.4f})"
                )

            scheduler.step(np.mean(losses))
            valid_losses.append(np.mean(losses))


if __name__ == "__main__":
    from model_cnn import custom_model
    from model_prep import getDataset

    BATCH_SIZE = 4

    TrainDataset, ValidDataset, TestDataset = getDataset(100)

    TrainDatasetLoader = DataLoader(TrainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    ValidDatasetLoader = DataLoader(ValidDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    TestDatasetLoader = DataLoader(TestDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(custom_model.parameters(), lr=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    print(f"Running on {device}")
    train(custom_model, loss_fn, optimizer=optimizer, scheduler=scheduler)
