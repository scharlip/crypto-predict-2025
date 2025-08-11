import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import statistics
import math

from common.common import CoinType, Exchange
from input.input import CoinDataset
from models.models import SimpleLSTMModel

ds = CoinDataset(coin_type=CoinType.ETH,exchange=Exchange.Coinbase)
train, validate, test = torch.utils.data.random_split(ds, [0.8, 0.05, 0.15])

device = torch.device("cpu")

torch.set_num_threads(8)

model = SimpleLSTMModel(input_size=ds.lookback_window_size, device=device).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
train_loader = data.DataLoader(train, shuffle=False, batch_size=32)
validation_loader = data.DataLoader(validate, shuffle=False, batch_size=32)

n_epochs = 200

for epoch in range(n_epochs):
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

    print("Completed validation for epoch {}. Avg training RMSE/batch: {}, Avg validation RMSE/batch: {}".format(
        epoch, avg_train_error, avg_validation_error))

    print("Done with epoch {}.".format(epoch))

# TODO: run test
