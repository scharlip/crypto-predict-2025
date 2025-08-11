import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import statistics

from common.common import CoinType, Exchange
from input.input import CoinDataset
from models.models import SimpleLSTMModel

ds = CoinDataset(coin_type=CoinType.ETH,exchange=Exchange.Coinbase)
ds.generate_midpoint_windows()
#train, validate, test = torch.utils.data.random_split(ds, [0.8, 0.05, 0.15])

total_samples = len(ds)
train_samples = int(total_samples * 0.8)
validation_samples = int(total_samples * 0.05)
test_samples = int(total_samples * 0.15)

train = torch.utils.data.Subset(ds, range(train_samples))
validate = torch.utils.data.Subset(ds, range(train_samples, train_samples + validation_samples))
test = torch.utils.data.Subset(ds, range(train_samples + validation_samples, total_samples))

'''
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available and will be used.")
else:
    device = torch.device("cpu")
    print("MPS backend not available, falling back to CPU.")
'''

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

    errors = []
    model.train()
    for X_batch, y_batch in tqdm(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)
        y_pred = y_pred.squeeze()
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Completed training for epoch {}.".format(epoch))

    errors = []

    print("Starting validation for epoch {} ...".format(epoch))

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in tqdm(validation_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            y_pred = y_pred.squeeze()
            rmse = np.sqrt(loss_fn(y_pred, y_batch))
            errors.append(rmse)

    avg_error = statistics.fmean(errors)

    print("Completed validation for epoch {}. Avg RMSE: {}".format(epoch, avg_error))

    print("Done with epoch {}.".format(epoch))

# TODO: run test
