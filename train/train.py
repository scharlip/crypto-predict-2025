import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch
from tqdm import tqdm

from common.common import CoinType, Exchange
from input.input import CoinDataset
from models.models import SimpleLSTMModel

ds = CoinDataset(coin_type=CoinType.ETH,exchange=Exchange.Coinbase)
ds.generate_midpoint_windows()
train, validate, test = torch.utils.data.random_split(ds, [0.8, 0.1, 0.1])

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

model = SimpleLSTMModel(input_size=60, device=device).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
train_loader = data.DataLoader(train, shuffle=True, batch_size=8)

n_epochs = 200
for epoch in range(n_epochs):
    print("Starting epoch {} ...".format(epoch))
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

    print("Done with epoch {}.".format(epoch))

    #TODO: run cross validation

# TODO: run test

'''
    with torch.no_grad():
        y_pred = model(dataset.X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, dataset.y_train))
        y_pred = model(dataset.X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, dataset.y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
'''
