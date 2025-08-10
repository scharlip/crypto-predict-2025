import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch
import numpy as np

from input.input import read_csv, CoinType, Exchange, construct_midpoint_dataset
from models.models import SimpleLSTMModel

df = read_csv(CoinType.ETH, Exchange.Coinbase)
dataset = construct_midpoint_dataset(df.head(1000), 10)

model = SimpleLSTMModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(dataset.X_train, dataset.y_train), shuffle=True, batch_size=8)

n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    if epoch % 100 != 0:
        continue

    model.eval()

    with torch.no_grad():
        y_pred = model(dataset.X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, dataset.y_train))
        y_pred = model(dataset.X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, dataset.y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
