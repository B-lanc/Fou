import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

import settings
from models.Fou import Fou, MaskFou
from dataset import NormalDataset, NormalShuffleDataset, BinaryShuffleDataset

import os
import json
import time
import sys


if __name__ == "__main__":
    LOAD_MODEL = False
    saved = {
        "channels": settings.channels,
        "n_vocals": settings.n_vocals,
        "n_noise": settings.n_noise,
        "alpha_vocals": settings.alpha_vox,
        "alpha_noise": settings.alpha_noi,
        "kernel_size": settings.kernel,
        "stride": settings.stride,
        "n_fft": settings.n_fft,
        "hop_size": settings.hop_size,
        "output_min": settings.output_min,
        "lr": settings.lr,
        "bs": settings.batch_size,
        "loss": settings.loss,
        "scaled": settings.scaled,
        "activation": settings.activation,
        "patience": settings.epoch_patience,
        "norm": settings.norm,
        "drop_rate": settings.drop_rate,
        "train_loss": [],
        "val_loss": [],
        "train_time": [],
        "val_time": [],
    }
    for key, val in saved.items():
        print(key, ":", val)

    checkpoint_dir = os.path.join(
        settings.checkpoint_dir, "FOU", settings.training_tag
    )
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    else:
        ans = input("Tag already exists, y to overwrite, n to cancel [y/N]:")
        if ans.lower() not in ["y", "yes"]:
            sys.exit()

    model = MaskFou(
        settings.channels,
        settings.kernel,
        settings.stride,
        settings.n_fft,
        settings.hop_size,
        settings.output_min,
        scaled=settings.scaled,
        activation=settings.activation,
        norm=settings.norm,
        dropout_rate=settings.drop_rate,
    )
    if settings.cuda:
        model.cuda()
    optimizer = Adam(params=model.parameters(), lr=settings.lr)

    if settings.loss.lower() == "mse":
        criterion = nn.MSELoss()
    elif settings.loss.lower() == "l1":
        criterion = nn.L1Loss()
    else:
        raise Exception("Loss function not implemented")

    n_params = sum([p.numel() for p in model.parameters()])
    print(f"Amount of parameters: {n_params}")
    print(f"Number of inputs: {model.input_samples}")
    print(f"Number of outputs: {model.output_samples}")

    train_dataset = BinaryShuffleDataset(
        settings.shuffle_train_hdf,
        model.input_samples,
        model.output_samples,
        settings.n_vocals,
        settings.n_noise,
        settings.alpha_vox,
        settings.alpha_noi,
    )
    val_dataset = NormalDataset(
        settings.val_hdf, model.input_samples, model.output_samples
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=4
    )

    state = {
        "epoch": 0,
        "bad_epoch": 0,
        "best_loss": np.inf,
        "best_loss_epoch": 0,
    }

    if LOAD_MODEL is not False:
        with open(os.path.join(checkpoint_dir, "logs.json"), "r") as f:
            saved = json.load(f)
        check = torch.load(os.path.join(checkpoint_dir, "model.pth"))
        model.load_state_dict(check["model_state_dict"])
        optimizer.load_state_dict(check["optimizer_state_dict"])
        state = check["state"]

        print(f"LOADING MODEL FROM EPOCH {state['epoch']}")
    print("TRAINING START")
    while state["bad_epoch"] < settings.epoch_patience:
        train_dataset.shuffle()
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=settings.batch_size,
            shuffle=True,
            num_workers=4,
        )
        model.train()
        time_start = time.time()
        avg_loss = 0
        with tqdm(total=len(train_dataset) // settings.batch_size) as pbar:
            for idx, (x, y) in enumerate(train_dataloader):
                if settings.cuda:
                    x = x.cuda()
                    y = y.cuda()
                optimizer.zero_grad()
                preds = model(x)
                loss = criterion(preds, y)
                avg_loss += loss.item()
                loss.backward()

                optimizer.step()
                pbar.update(1)
        avg_loss = avg_loss / (len(train_dataset) // settings.batch_size)
        diff_time = time.time() - time_start
        saved["train_loss"].append(avg_loss)
        saved["train_time"].append(diff_time)
        print(
            f"Epoch {state['epoch']} training. Time:{diff_time}, Train Loss:{avg_loss}"
        )

        # VALIDATION
        model.eval()
        time_start = time.time()
        avg_loss = 0
        for idx, (x, y) in enumerate(val_dataloader):
            if settings.cuda:
                x = x.cuda()
                y = y.cuda()
            preds = model(x)
            loss = criterion(preds, y)
            avg_loss += loss.item()
        avg_loss = avg_loss / (len(val_dataset) // settings.batch_size)

        diff_time = time.time() - time_start
        saved["val_loss"].append(avg_loss)
        saved["val_time"].append(diff_time)

        if avg_loss < state["best_loss"]:
            state["best_loss"] = avg_loss
            state["best_loss_epoch"] = state["epoch"]
            state["bad_epoch"] = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "state": state,
                },
                os.path.join(checkpoint_dir, "model.pth"),
            )
            with open(os.path.join(checkpoint_dir, "logs.json"), "w") as f:
                json.dump(saved, f, indent=2)
        else:
            state["bad_epoch"] += 1

        print(
            f"Epoch {state['epoch']} validation. Time:{diff_time}, Val Loss:{avg_loss}, cons:{state['bad_epoch']}"
        )
        state["epoch"] += 1

    with open(os.path.join(checkpoint_dir, "logs.json"), "w") as f:
        json.dump(saved, f, indent=2)
    print(f"Training finished at epoch {state['epoch']}")
