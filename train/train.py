from typing import List

import torch.utils.data as data
import torch.nn as nn
import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from tqdm import tqdm
import statistics
import math

from input.input import CoinDataset

def train_loop(
        ds: CoinDataset,
        model: nn.Module,
        device,
        optimizer: Optimizer,
        loss_fn: _Loss = nn.MSELoss,
        splits: List[float] = [0.8, 0.05, 0.15],
        batch_size = 32,
        epochs = 200
):
    train, validate, test = torch.utils.data.random_split(ds, splits)

    train_loader = data.DataLoader(train, shuffle=False, batch_size=batch_size)
    validation_loader = data.DataLoader(validate, shuffle=False, batch_size=batch_size)
    test_loader = data.DataLoader(test, shuffle=False, batch_size=32)

    print("Beginning training loop.")
    print("Model:")
    print(model)
    print("Optimizer: {}".format(optimizer))
    print("Loss function: {}".format(loss_fn))
    print("Batch size: {}".format(batch_size))
    print("Splits: {}".format(splits))
    print("Epochs: {}".format(epochs))

    for epoch in range(epochs):
        print("Starting epoch {} ...".format(epoch))

        print("Starting training for epoch {} ...".format(epoch))

        train_errors = []
        model.train()
        for X_batch, y_batch in tqdm(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            y_pred = y_pred.squeeze()
            loss = loss_fn(y_pred, y_batch)
            train_errors.append(math.sqrt(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Completed training for epoch {}.".format(epoch))

        validation_errors = []

        print("Starting validation for epoch {} ...".format(epoch))

        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in tqdm(validation_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                y_pred = y_pred.squeeze()
                rmse = math.sqrt(loss_fn(y_pred, y_batch))
                validation_errors.append(rmse)

        avg_train_error = statistics.fmean(train_errors)
        avg_validation_error = statistics.fmean(validation_errors)

        print("Completed validation for epoch {}. Avg training RMSE: {} (per batch: {}), Avg validation RMSE: {} (per batch: {})".format(
            epoch, avg_train_error/batch_size, avg_train_error, avg_validation_error/batch_size, avg_validation_error))

        print("Done with epoch {}.".format(epoch))

    test_errors = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            y_pred = y_pred.squeeze()
            rmse = math.sqrt(loss_fn(y_pred, y_batch))
            test_errors.append(rmse)

    avg_test_error = statistics.fmean(test_errors)
    print(
        "Completed test. Avg test RMSE: {} (per batch: {})".format(
            avg_test_error / batch_size, avg_test_error
        )
    )

    print("Completed training loop.")

