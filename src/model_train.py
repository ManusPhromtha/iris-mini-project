import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
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


def train(
    model: nn.Module,
    loss_fn,
    optimizer,
    scheduler,
    start_epoch: int = 1,
    epoch_count: int = 10,
    model_path: str = "",
    checkpoint_path: str = "",
    train_losses: list | None = None,
    valid_losses: list | None = None,
):
    model.to(device)

    train_losses = train_losses if train_losses is not None else []
    valid_losses = valid_losses if valid_losses is not None else []

    for epoch in range(start_epoch, start_epoch + epoch_count):
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
                    f"Accuracy: ({accuracy:.4f}|{np.mean(accuracies):.4f}), Loss: ({loss:.4f}|{np.mean(losses):.4f})"
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

        if checkpoint_path != "":
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_losses": train_losses,
                "valid_losses": valid_losses,
            }
            torch.save(state, checkpoint_path)

        if model_path != "" and (epoch == 1 or valid_losses[-1] < min(valid_losses[:-1])):
            print("Saving new best model...")
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Data Preparation Script",
        description="Prepare the data for later training and evaluating",
    )
    parser.add_argument("-c", "--checkpoint-path", default="", help="Path to pytorch train checkpoint file")
    parser.add_argument("-m", "--model-path", default="", help="Path to pytorch model file")
    args = parser.parse_args()
    checkpoint_path = str(args.checkpoint_path)
    model_path = str(args.model_path)

    import os

    from model_cnn import CNNModel
    from model_prep import getDatasetLoader

    TrainDatasetLoader, ValidDatasetLoader, TestDatasetLoader = getDatasetLoader(label_count=100, batch_size=4)

    custom_model = CNNModel()
    loss_fn = nn.BCELoss()

    if checkpoint_path != "" and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        custom_model.load_state_dict(checkpoint["model"])
        optimizer = optim.Adam(custom_model.parameters(), lr=0.05)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

        print(f"Loaded epoch {checkpoint['epoch']} model, best validation loss: {min(valid_losses)}")
    else:
        optimizer = optim.Adam(custom_model.parameters(), lr=0.05)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

        start_epoch = 1
        train_losses = []
        valid_losses = []

    print(f"Running on {device}")
    train(
        custom_model,
        loss_fn,
        start_epoch=start_epoch,
        epoch_count=10,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        train_losses=train_losses,
        valid_losses=valid_losses,
    )
