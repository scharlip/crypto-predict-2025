import os
from typing import List

import torch.utils.data as data
import torch.nn as nn
import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from tqdm.auto import tqdm
import statistics
import math
import gc
from datetime import datetime

from backtest import backtest
from input.coindataset import CoinDataset
from models.BaseModel import BaseModel


def train_loop(
        ds: CoinDataset,
        model: BaseModel,
        optimizer: Optimizer,
        model_save_dir: str,
        base_log_dir: str,
        loss_fn: _Loss = nn.MSELoss,
        splits: List[float] = [0.8, 0.05, 0.15],
        batch_size = 32,
        epochs = 20,
        epochs_per_backtest = 5
):
    train, validate, test = torch.utils.data.random_split(ds, splits)

    train_loader = data.DataLoader(train, shuffle=False, batch_size=batch_size)
    validation_loader = data.DataLoader(validate, shuffle=False, batch_size=batch_size)
    test_loader = data.DataLoader(test, shuffle=False, batch_size=batch_size)

    timestamp_str = datetime.now().strftime("%-m/%-d/%Y %-I:%M:%S %p")
    print("Beginning training loop. ({})".format(timestamp_str))
    print("Model:")
    print(model)
    print("Optimizer: {}".format(optimizer))
    print("Loss function: {}".format(loss_fn))
    print("Batch size: {}".format(batch_size))
    print("Splits: {}".format(splits))
    print("Epochs: {}".format(epochs))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device: {}".format(device))

    backtests_run = 0

    log_dir = "{}/{}".format(base_log_dir, model.descriptor_string())
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(epochs):
        print("Starting epoch {} ...".format(epoch))

        timestamp_str = datetime.now().strftime("%-m/%-d/%Y %-I:%M:%S %p")
        print("Starting training for epoch {} ... ({})".format(epoch, timestamp_str))

        train_losses = []
        model.train()
        for X_batch, y_batch in tqdm(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            y_pred = y_pred.squeeze()
            loss = loss_fn()(y_pred, y_batch)
            train_losses.append(math.sqrt(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del X_batch, y_batch, y_pred
            gc.collect()
            torch.cuda.empty_cache()

        timestamp_str = datetime.now().strftime("%-m/%-d/%Y %-I:%M:%S %p")
        print("Completed training for epoch {}.".format(epoch, timestamp_str))

        validation_losses = []

        timestamp_str = datetime.now().strftime("%-m/%-d/%Y %-I:%M:%S %p")
        print("Starting validation for epoch {} ... ({})".format(epoch, timestamp_str))

        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in tqdm(validation_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                y_pred = y_pred.squeeze()
                rmse = math.sqrt(loss_fn()(y_pred, y_batch))
                validation_losses.append(rmse)

                del X_batch, y_batch, y_pred
                gc.collect()
                torch.cuda.empty_cache()

        avg_train_loss = statistics.fmean(train_losses)
        avg_validation_loss = statistics.fmean(validation_losses)

        timestamp_str = datetime.now().strftime("%-m/%-d/%Y %-I:%M:%S %p")
        print("Completed validation for epoch {} ({}). Avg training RMSE: {} (per batch: {}), Avg validation RMSE: {} (per batch: {})".format(
            epoch, timestamp_str, avg_train_loss/batch_size, avg_train_loss, avg_validation_loss/batch_size, avg_validation_loss))

        backtest.spot_check(ds=ds, model=model, log_file_name="{}/spot_check_epoch_{}.txt".format(log_dir, epoch))

        print("Saving model ... (dir: {})".format(model_save_dir))
        model.save_model(model_save_dir, epoch=epoch, batch_size=batch_size, avg_train_loss=avg_train_loss, avg_validation_loss=avg_validation_loss)

        print("Done with epoch {}.".format(epoch))

        if epoch > 0 and epochs_per_backtest is not None and epoch % epochs_per_backtest == 0:
            timestamp_str = datetime.now().strftime("%-m/%-d/%Y %-I:%M:%S %p")
            print("Running backtest #{} ... ({})".format(backtests_run + 1, timestamp_str))
            backtest.run_backtest(ds, model, log_file_name="{}/backtest_epoch_{}.txt".format(log_dir, epoch))
            timestamp_str = datetime.now().strftime("%-m/%-d/%Y %-I:%M:%S %p")
            print("Backtest #{} completed.".format(backtests_run + 1, timestamp_str))
            backtests_run += 1

        gc.collect()
        torch.cuda.empty_cache()

    test_errors = []

    timestamp_str = datetime.now().strftime("%-m/%-d/%Y %-I:%M:%S %p")
    print("Starting test ... ({})".format(timestamp_str))
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            y_pred = y_pred.squeeze()
            rmse = math.sqrt(loss_fn()(y_pred, y_batch))
            test_errors.append(rmse)

    avg_test_error = statistics.fmean(test_errors)
    timestamp_str = datetime.now().strftime("%-m/%-d/%Y %-I:%M:%S %p")
    print(
        "Completed test ({}). Avg test RMSE: {} (per batch: {})".format(
            timestamp_str,
            avg_test_error / batch_size, avg_test_error
        )
    )

    if epochs_per_backtest is not None and backtests_run == 0:
        timestamp_str = datetime.now().strftime("%-m/%-d/%Y %-I:%M:%S %p")
        print("Haven't run a backtest yet. Running final backtest ... ({})".format(timestamp_str))
        backtest.run_backtest(ds, model, log_file_name="{}/backtest_final.txt".format(log_dir))
        backtests_run += 1
        timestamp_str = datetime.now().strftime("%-m/%-d/%Y %-I:%M:%S %p")
        print("Final backtest completed. ({})".format(timestamp_str))

    del model, X_batch, y_batch, y_pred

    timestamp_str = datetime.now().strftime("%-m/%-d/%Y %-I:%M:%S %p")
    print("Completed training loop. ({})".format(timestamp_str))

